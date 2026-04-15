#!/usr/bin/env python3
"""
monitor_node.py

ROS2 monitoring and metrics node for the MPPI racing controller.
Replaces the simpler lap_logger with full metrics collection:

  - Cross-track error (nearest point on the planner's reference trajectory)
  - Collision detection via LiDAR minimum range (N consecutive readings)
  - Lap timing via start/finish line crossing (sign-change dot-product method)
  - Live Rerun visualization (if rerun-sdk is installed)
  - /batch/status publisher for eventual batch orchestration

Subscriptions:
  /ego_racecar/odom           nav_msgs/Odometry
  /planner_traj               autoware_auto_planning_msgs/Trajectory
  /scan                       sensor_msgs/LaserScan
  /drive                      ackermann_msgs/AckermannDriveStamped

Publishes:
  /batch/status               std_msgs/String  (running | collision | laps_complete | timeout)

CSV output (auto-enumerated): resources/logs/run_NNN.csv
  columns: timestamp_s, x, y, yaw_rad, speed_mps, steering_angle_rad, cross_track_error_m
"""

import csv
import glob
import json
import math
import os
import re
import signal
import rerun as rr
import numpy as np

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from autoware_auto_planning_msgs.msg import Trajectory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_run_number(logs_dir: str) -> int:
    """Return the next unused run number by scanning existing run_NNN.csv files."""
    existing = glob.glob(os.path.join(logs_dir, "run_*.csv"))
    nums = []
    for p in existing:
        m = re.search(r"run_(\d+)\.csv$", p)
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def _yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


# ── Node ──────────────────────────────────────────────────────────────────────

class MonitorNode(Node):
    def __init__(self):
        super().__init__("monitor_node")

        # ── Parameters ───────────────────────────────────────────
        self.declare_parameter("target_laps", 1)
        self.declare_parameter("timeout_seconds", 300.0)
        self.declare_parameter("collision_threshold_m", 0.12)
        self.declare_parameter("collision_consecutive_readings", 3)
        self.declare_parameter("min_lap_distance", 30.0)
        self.declare_parameter("rerun_output", "")
        self.declare_parameter("log_output", "")
        self.declare_parameter("metrics_output_path", "")

        self._target_laps      = self.get_parameter("target_laps").value
        self._timeout_s        = self.get_parameter("timeout_seconds").value
        self._coll_thresh      = self.get_parameter("collision_threshold_m").value
        self._coll_n           = self.get_parameter("collision_consecutive_readings").value
        self._min_lap_dist     = self.get_parameter("min_lap_distance").value
        self._rerun_output     = self.get_parameter("rerun_output").value
        self._log_output       = self.get_parameter("log_output").value
        self._metrics_output   = self.get_parameter("metrics_output_path").value

        # ── Output file ──────────────────────────────────────────
        if self._log_output:
            # Explicit path provided by batch_runner — write directly there.
            self._csv_path = self._log_output
            os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
        else:
            # Fallback: auto-numbered file next to the script's resources dir.
            logs_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "resources", "logs")
            )
            os.makedirs(logs_dir, exist_ok=True)
            run_num = _next_run_number(logs_dir)
            self._csv_path = os.path.join(logs_dir, f"run_{run_num:03d}.csv")

        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp_s", "x", "y", "yaw_rad",
            "speed_mps", "steering_angle_rad", "cross_track_error_m",
        ])

        self.get_logger().info(f"Monitor node started — output: {self._csv_path}")

        # ── Vehicle state cache ───────────────────────────────────
        self._x: float = 0.0
        self._y: float = 0.0
        self._yaw: float = 0.0
        self._speed: float = 0.0
        self._steering: float = 0.0

        # ── Reference trajectory (x,y array from /planner_traj) ──
        self._ref_xy: np.ndarray | None = None   # shape (N, 2)

        # ── Lap detection — sign-change finish line ───────────────
        # Finish line is defined at start pose, perpendicular to initial heading.
        # A lap is counted when the signed projection of (car - start) onto the
        # initial heading vector transitions from negative to non-negative.
        self._start_x: float | None = None
        self._start_y: float | None = None
        self._finish_cos: float = 0.0   # cos(yaw0)
        self._finish_sin: float = 0.0   # sin(yaw0)
        self._prev_dot: float = 0.0
        self._dist_travelled: float = 0.0
        self._prev_x: float | None = None
        self._prev_y: float | None = None
        self._armed: bool = False
        self._lap_count: int = 0
        self._lap_start_time: float | None = None

        # ── CTE tracking ─────────────────────────────────────────
        self._cte_values: list = []
        self._total_distance: float = 0.0

        # ── Lap times list ────────────────────────────────────────
        self._lap_times: list = []

        # ── Collision detection ───────────────────────────────────
        self._coll_streak: int = 0

        # ── Active flag ───────────────────────────────────────────
        self._active: bool = True
        self._final_status: str = ""

        # ── Rerun ─────────────────────────────────────────────────
        rr.init("mppi_monitor", recording_id=self._csv_path)
        if self._rerun_output:
            os.makedirs(os.path.dirname(self._rerun_output), exist_ok=True)
            rr.save(self._rerun_output)
            self.get_logger().info(f"Rerun recording → {self._rerun_output}")
        else:
            self.get_logger().warn("rerun_output not set — Rerun data will not be saved")

        # ── Publisher ─────────────────────────────────────────────
        self._status_pub = self.create_publisher(String, "/batch/status", 10)
        self._publish_status("running")

        # ── Subscriptions ─────────────────────────────────────────
        self.create_subscription(Odometry,              "/ego_racecar/odom", self._odom_cb,  10)
        self.create_subscription(Trajectory,            "/planner_traj",     self._traj_cb,  10)
        self.create_subscription(LaserScan,             "/scan",             self._scan_cb,  10)
        self.create_subscription(AckermannDriveStamped, "/drive",            self._drive_cb, 10)

        # ── Timeout timer ─────────────────────────────────────────
        self._timeout_timer = self.create_timer(self._timeout_s, self._on_timeout)

    # ── Status publisher ─────────────────────────────────────────

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    # ── Callbacks ────────────────────────────────────────────────

    def _drive_cb(self, msg: AckermannDriveStamped) -> None:
        self._speed    = msg.drive.speed
        self._steering = msg.drive.steering_angle

    def _traj_cb(self, msg: Trajectory) -> None:
        pts = msg.points
        if not pts:
            return
        self._ref_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in pts])

        rr.log("/world/reference_path",
        rr.LineStrips2D([self._ref_xy.tolist()]))

    def _scan_cb(self, msg: LaserScan) -> None:
        if not self._active:
            return

        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if not valid:
            return

        if min(valid) < self._coll_thresh:
            self._coll_streak += 1
        else:
            self._coll_streak = 0

        if self._coll_streak >= self._coll_n:
            self.get_logger().error(
                f"Collision detected — min range {min(valid):.3f} m for "
                f"{self._coll_streak} consecutive readings"
            )
            rr.log("/events/collision", rr.TextLog("Collision detected"))
            self._finish("collision")

    def _odom_cb(self, msg: Odometry) -> None:
        if not self._active:
            return

        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y
        yaw = _yaw_from_quaternion(msg.pose.pose.orientation)
        t   = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self._x   = x
        self._y   = y
        self._yaw = yaw

        # ── Latch start pose ──────────────────────────────────────
        if self._start_x is None:
            self._start_x     = x
            self._start_y     = y
            self._finish_cos  = math.cos(yaw)
            self._finish_sin  = math.sin(yaw)
            self._prev_dot    = 0.0
            self._prev_x      = x
            self._prev_y      = y
            self._lap_start_time = t
            return

        # ── Accumulate distance ───────────────────────────────────
        step = math.hypot(x - self._prev_x, y - self._prev_y)
        self._dist_travelled += step
        self._prev_x = x
        self._prev_y = y

        if not self._armed and self._dist_travelled >= self._min_lap_dist:
            self._armed = True
            self.get_logger().info(
                f"Finish line armed after {self._dist_travelled:.1f} m"
            )

        # ── Finish line crossing ──────────────────────────────────
        dot = ((x - self._start_x) * self._finish_cos
               + (y - self._start_y) * self._finish_sin)

        if self._armed and self._prev_dot < 0.0 and dot >= 0.0:
            lap_time = t - self._lap_start_time
            self._lap_count += 1
            self._lap_times.append(round(lap_time, 4))
            self.get_logger().info(
                f"Lap {self._lap_count} complete — {lap_time:.2f} s"
            )
            rr.log("/metrics/lap_time", rr.Scalars(lap_time))
            rr.log("/events/lap_crossing", rr.TextLog(f"Lap {self._lap_count} — {lap_time:.2f} s"))
            self._lap_start_time  = t
            self._dist_travelled  = 0.0
            self._armed           = False   # re-arm for next lap

            if self._lap_count >= self._target_laps:
                self._finish("laps_complete")
                return

        self._prev_dot = dot

        # ── CTE ───────────────────────────────────────────────────
        cte = 0.0
        if self._ref_xy is not None:
            diffs = self._ref_xy - np.array([x, y])
            cte   = float(np.min(np.hypot(diffs[:, 0], diffs[:, 1])))
        self._cte_values.append(cte)
        self._total_distance += step

        # ── Write CSV row ─────────────────────────────────────────
        self._csv_writer.writerow([
            f"{t:.6f}", f"{x:.4f}", f"{y:.4f}", f"{yaw:.6f}",
            f"{self._speed:.4f}", f"{self._steering:.6f}", f"{cte:.4f}",
        ])

        # ── Rerun ─────────────────────────────────────────────────
        rr.set_time("time", duration=t)
        rr.log("/world/car/position", rr.Points2D([[x, y]]))
        rr.log("/metrics/cross_track_error", rr.Scalars([cte]))
        rr.log("/metrics/speed",             rr.Scalars([self._speed]))
        rr.log("/metrics/steering_angle",    rr.Scalars([math.degrees(self._steering)]))

    # ── Termination ───────────────────────────────────────────────

    def _on_timeout(self) -> None:
        if self._active:
            self.get_logger().warn(f"Timeout after {self._timeout_s:.0f} s")
            self._finish("timeout")

    def _finish(self, status: str) -> None:
        if not self._active:
            return
        self._active = False
        self._final_status = status
        self._csv_file.flush()
        self._csv_file.close()

        # ── Write metrics JSON ────────────────────────────────────
        metrics: dict = {
            "status":    status,
            "collision": status == "collision",
        }
        if self._cte_values:
            metrics["mean_cte_m"] = round(float(np.mean(self._cte_values)), 4)
            metrics["max_cte_m"]  = round(float(np.max(self._cte_values)),  4)
        else:
            metrics["mean_cte_m"] = None
            metrics["max_cte_m"]  = None

        metrics["lap_times"]       = self._lap_times
        metrics["best_lap_time_s"] = round(min(self._lap_times), 4) if self._lap_times else None
        metrics["total_distance_m"] = round(self._total_distance, 2)

        metrics_path = self._metrics_output or str(self._csv_path).replace(".csv", "_metrics.json")
        try:
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            self.get_logger().info(f"Metrics written → {metrics_path}")
        except Exception as exc:
            self.get_logger().error(f"Failed to write metrics: {exc}")

        self.get_logger().info(f"Run ended ({status}) — log saved to {self._csv_path}")
        # Publish status now, then again after a short delay to guarantee delivery
        # before the node shuts down.  Raising SystemExit immediately would kill
        # the node before the message leaves the DDS queue.
        self._publish_status(status)
        self.create_timer(0.5, self._shutdown_callback)

    def _shutdown_callback(self) -> None:
        # Second publish in case the first message was dropped during startup
        self._publish_status(self._final_status)
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = MonitorNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
