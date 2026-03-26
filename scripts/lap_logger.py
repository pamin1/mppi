#!/usr/bin/env python3
"""
Lap Logger Node

Subscribes to /drive (AckermannDriveStamped) and /ego_racecar/odom (Odometry).
Logs commanded speed and steering angle to an enumerated CSV starting immediately
on node launch.  Logging ends when the car returns to its starting position after
having travelled at least MIN_LAP_DISTANCE metres — a loose one-lap detector that
does not require a precise finish line.

Output filename : resources/logs/lap_NNN.csv  (NNN auto-increments)
Output CSV columns: timestamp_s, speed_mps, steering_angle_rad, x, y, yaw_rad
"""

import csv
import glob
import math
import os
import re

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def _next_lap_number(logs_dir: str) -> int:
    """Return the next unused lap number by scanning existing lap_NNN.csv files."""
    existing = glob.glob(os.path.join(logs_dir, "lap_*.csv"))
    nums = []
    for p in existing:
        m = re.search(r"lap_(\d+)\.csv$", p)
        if m:
            nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


# How close (metres) the car must be to the start position to trigger lap-end.
LAP_FINISH_RADIUS = 2.5
# Minimum distance (metres) to travel before the finish check is armed.
# Prevents false-positive detection right after launch.
MIN_LAP_DISTANCE = 30.0


class LapLogger(Node):
    def __init__(self):
        super().__init__("lap_logger")

        # ── State ────────────────────────────────────────────────
        self._start_x: float | None = None
        self._start_y: float | None = None
        self._prev_x: float | None = None
        self._prev_y: float | None = None
        self._distance_travelled: float = 0.0
        self._armed: bool = False          # True once MIN_LAP_DISTANCE exceeded
        self._logging: bool = True
        self._rows: list[tuple] = []       # buffered before flush
        # Latest pose — updated by odom, stamped into each drive row
        self._pose_x: float = 0.0
        self._pose_y: float = 0.0
        self._pose_yaw: float = 0.0

        # ── Output file ──────────────────────────────────────────
        logs_dir = os.path.join(
            os.path.dirname(__file__), "..", "resources", "logs"
        )
        logs_dir = os.path.normpath(logs_dir)
        os.makedirs(logs_dir, exist_ok=True)

        lap_num = _next_lap_number(logs_dir)
        self._lap_num = lap_num
        self._csv_path = os.path.join(logs_dir, f"lap_{lap_num:03d}.csv")

        self.get_logger().info(f"Lap logger started — output: {self._csv_path}")

        # ── Subscriptions ────────────────────────────────────────
        self.create_subscription(
            AckermannDriveStamped,
            "/drive",
            self._drive_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            "/ego_racecar/odom",
            self._odom_callback,
            10,
        )

    # ── Callbacks ────────────────────────────────────────────────

    def _drive_callback(self, msg: AckermannDriveStamped) -> None:
        if not self._logging:
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._rows.append((
            t,
            msg.drive.speed,
            msg.drive.steering_angle,
            self._pose_x,
            self._pose_y,
            self._pose_yaw,
        ))

    def _odom_callback(self, msg: Odometry) -> None:
        if not self._logging:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        # Cache for drive rows
        self._pose_x = x
        self._pose_y = y
        self._pose_yaw = yaw

        # Latch start position on first message
        if self._start_x is None:
            self._start_x = x
            self._start_y = y
            self._prev_x = x
            self._prev_y = y
            return

        # Accumulate distance
        dx = x - self._prev_x
        dy = y - self._prev_y
        self._distance_travelled += math.hypot(dx, dy)
        self._prev_x = x
        self._prev_y = y

        # Arm the finish detector once enough ground has been covered
        if not self._armed and self._distance_travelled >= MIN_LAP_DISTANCE:
            self._armed = True
            self.get_logger().info(
                f"Lap detector armed after {self._distance_travelled:.1f} m"
            )

        # Check for lap completion
        if self._armed:
            dist_to_start = math.hypot(x - self._start_x, y - self._start_y)
            if dist_to_start <= LAP_FINISH_RADIUS:
                self._finish_lap()

    # ── Lap finish ───────────────────────────────────────────────

    def _finish_lap(self) -> None:
        self._logging = False
        self.get_logger().info(
            f"Lap complete — {self._distance_travelled:.1f} m travelled, "
            f"{len(self._rows)} drive commands recorded"
        )
        self._write_csv()
        self.get_logger().info(f"Log saved to: {self._csv_path}")
        # Trigger a clean shutdown of this node only
        raise SystemExit(0)

    def _write_csv(self) -> None:
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_s", "speed_mps", "steering_angle_rad", "x", "y", "yaw_rad"])
            writer.writerows(self._rows)


def main(args=None):
    rclpy.init(args=args)
    node = LapLogger()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
