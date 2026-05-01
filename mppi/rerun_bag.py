#!/usr/bin/env python3
"""
Rerun viewer for F1TENTH MPPI rosbag2 (.db3) files.

Reads a ROS2 bag (sqlite3 backend) using the `rosbags` library (no ROS install
needed) and logs the data into Rerun for interactive visualization.

Visualised topics
─────────────────
  /ego_racecar/odom    → 2-D pose trail + velocity time-series
  /scan                → laser-scan rays (LineStrips2D)
  /drive               → steering & speed commands (time-series)
  /planner_traj        → planned trajectory (LineStrips2D)
  /visual_raceline     → reference raceline (LineStrips2D)
  /mppi/top_k_paths    → sampled MPPI rollouts (LineStrips2D)
  /local_costmap       → local costmap (Image)
  /map                 → static map (Image, logged once)

Usage
─────
  pip install rerun-sdk rosbags numpy
  python rerun_bag_viewer.py /path/to/bag_directory

The bag directory must contain metadata.yaml + the .db3 file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

# ──────────────────────────────────────────────────────────────────────
# Custom message definitions (not in the default Humble typestore)
# ──────────────────────────────────────────────────────────────────────

ACKERMANN_DRIVE_MSG = """\
float32 steering_angle
float32 steering_angle_velocity
float32 speed
float32 acceleration
float32 jerk
"""

ACKERMANN_DRIVE_STAMPED_MSG = """\
std_msgs/Header header
ackermann_msgs/AckermannDrive drive
"""

# autoware_auto_planning_msgs/msg/TrajectoryPoint
TRAJECTORY_POINT_MSG = """\
builtin_interfaces/Duration time_from_start
geometry_msgs/Pose pose
float32 longitudinal_velocity_mps
float32 lateral_velocity_mps
float32 acceleration_mps2
float32 heading_rate_rps
float32 front_wheel_angle_rad
float32 rear_wheel_angle_rad
"""

# autoware_auto_planning_msgs/msg/Trajectory
TRAJECTORY_MSG = """\
std_msgs/Header header
autoware_auto_planning_msgs/TrajectoryPoint[] points
"""

# bond/msg/Status  (just need enough to deserialize, we skip it)
BOND_STATUS_MSG = """\
std_msgs/Header header
string id
string instance_id
bool active
float32 heartbeat_timeout
float32 heartbeat_period
"""


def register_custom_types(typestore):
    """Register non-standard message types with the rosbags typestore."""
    add_types = {}
    add_types.update(get_types_from_msg(
        ACKERMANN_DRIVE_MSG, 'ackermann_msgs/msg/AckermannDrive'))
    add_types.update(get_types_from_msg(
        ACKERMANN_DRIVE_STAMPED_MSG, 'ackermann_msgs/msg/AckermannDriveStamped'))
    add_types.update(get_types_from_msg(
        TRAJECTORY_POINT_MSG, 'autoware_auto_planning_msgs/msg/TrajectoryPoint'))
    add_types.update(get_types_from_msg(
        TRAJECTORY_MSG, 'autoware_auto_planning_msgs/msg/Trajectory'))
    add_types.update(get_types_from_msg(
        BOND_STATUS_MSG, 'bond/msg/Status'))
    typestore.register(add_types)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def quat_to_yaw(q) -> float:
    """Extract yaw from a geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def set_time(timestamp_ns: int):
    """Set the Rerun timeline from a ROS nanosecond timestamp."""
    rr.set_time(timeline="ros_time", timestamp=timestamp_ns * 1e-9)


# ──────────────────────────────────────────────────────────────────────
# Per-topic handlers
# ──────────────────────────────────────────────────────────────────────

def log_odom(msg, ts):
    set_time(ts)
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    yaw = quat_to_yaw(q)

    # Transform the robot frame into the world frame
    rr.log("world/robot", rr.Transform3D(
        translation=[p.x, p.y, 0.0],
        rotation=rr.RotationAxisAngle(axis=[0, 0, 1], angle=yaw),
    ))

    # Heading arrow (now in robot-local coords, i.e. always points +x)
    rr.log("world/robot/heading", rr.Arrows2D(
        origins=[[0.0, 0.0]],
        vectors=[[0.3, 0.0]],
        colors=[0, 255, 0],
    ))

    # Time-series
    rr.log("timeseries/velocity/linear", rr.Scalars(msg.twist.twist.linear.x))
    rr.log("timeseries/velocity/angular", rr.Scalars(msg.twist.twist.angular.z))


def log_scan(msg, ts):
    set_time(ts)
    angles = np.arange(msg.angle_min, msg.angle_max, msg.angle_increment)
    ranges = np.array(msg.ranges, dtype=np.float32)

    # Clamp invalid values
    valid = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
    angles = angles[: len(ranges)]  # safety
    valid = valid[: len(angles)]

    xs = ranges[valid] * np.cos(angles[valid])
    ys = ranges[valid] * np.sin(angles[valid])
    pts = np.column_stack([xs, ys])

    # Draw rays from origin to each scan point
    origins = np.zeros_like(pts)
    lines = np.stack([origins, pts], axis=1)  # (N, 2, 2)
    rr.log("world/robot/scan", rr.LineStrips2D(
        lines, colors=[255, 100, 100], radii=0.005))


def log_drive(msg, ts):
    set_time(ts)
    rr.log("timeseries/drive/speed", rr.Scalars(msg.drive.speed))
    rr.log("timeseries/drive/steering_angle", rr.Scalars(msg.drive.steering_angle))


def log_planner_traj(msg, ts):
    set_time(ts)
    pts = []
    for pt in msg.points:
        p = pt.pose.position
        pts.append([p.x, p.y])
    if len(pts) >= 2:
        rr.log("world/planner_traj", rr.LineStrips2D(
            [pts], colors=[0, 200, 255], radii=0.02))


def log_raceline(msg, ts):
    set_time(ts)
    pts = [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
    if len(pts) >= 2:
        rr.log("world/raceline", rr.LineStrips2D(
            [pts], colors=[255, 255, 0], radii=0.015))


def log_top_k_paths(msg, ts):
    set_time(ts)
    strips = []
    for marker in msg.markers:
        if len(marker.points) < 2:
            continue
        strip = [[p.x, p.y] for p in marker.points]
        strips.append(strip)
    if strips:
        rr.log("world/mppi_rollouts", rr.LineStrips2D(
            strips, colors=[180, 100, 255], radii=0.008))

def log_steering_modes(msg, ts):
    set_time(ts)
    for marker in msg.markers:
        if marker.action == 3:  # DELETEALL
            continue
        if marker.type != 0:    # not ARROW
            continue
        if len(marker.points) < 2:
            continue

        start = marker.points[0]
        end = marker.points[1]
        mode_id = marker.id

        is_best = (marker.color.g > 0.5 and marker.color.r < 0.5)
        color = [0, 255, 0] if is_best else [255, 0, 0]

        # Arrow from origin to gap endpoint (in laser frame, child of robot)
        rr.log(f"world/robot/modes/arrow_{mode_id}", rr.Arrows2D(
            origins=[[start.x, start.y]],
            vectors=[[end.x - start.x, end.y - start.y]],
            colors=[color],
        ))

        # std_dev is packed into scale.x
        rr.log(f"timeseries/modes/std_dev/mode_{mode_id}", rr.Scalars(marker.scale.x))

def log_occupancy_grid(msg, ts, entity: str):
    """Log an OccupancyGrid as a greyscale image."""
    set_time(ts)
    w, h = msg.info.width, msg.info.height
    res = msg.info.resolution
    origin = msg.info.origin

    yaw = quat_to_yaw(origin.orientation)

    # Place the grid at its origin in world coords, scaled by resolution
    rr.log(entity, rr.Transform3D(
        translation=[origin.position.x, origin.position.y, 0.0],
        rotation=rr.RotationAxisAngle(axis=[0, 0, 1], angle=yaw),
        scale=[res, res, 1.0],
    ))

    data = np.array(msg.data, dtype=np.int8).astype(np.int16).reshape((h, w))
    img = np.where(data < 0, 127, 255 - data * 255 // 100).astype(np.uint8)
    rr.log(f"{entity}/grid", rr.Image(img))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

TOPIC_HANDLERS = {
    "/ego_racecar/odom":    log_odom,
    "/scan":                log_scan,
    "/drive":               log_drive,
    "/planner_traj":        log_planner_traj,
    "/visual_raceline":     log_raceline,
    "/mppi/top_k_paths":    log_top_k_paths,
    "/mppi/steering_modes": log_steering_modes,
}

# Topics we handle with a special signature
COSTMAP_TOPICS = {
    "/local_costmap": "world/local_costmap",
    "/map":           "world/map",
}


def build_blueprint() -> rrb.Blueprint:
    """Build a sensible default Rerun layout."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="World", origin="world"),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    name="Velocity",
                    origin="timeseries/velocity",
                ),
                rrb.TimeSeriesView(
                    name="Drive Commands",
                    origin="timeseries/drive",
                ),
            ),
            column_shares=[3, 1],
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize an F1TENTH MPPI rosbag2 in Rerun")
    parser.add_argument("bag", type=Path,
                        help="Path to the rosbag2 directory (contains metadata.yaml)")
    parser.add_argument("--spawn", action="store_true", default=True,
                        help="Spawn the Rerun viewer (default)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save to .rrd file instead of spawning viewer")
    args = parser.parse_args()

    bag_path = args.bag.resolve()
    if not bag_path.exists():
        print(f"Error: {bag_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # --- Rerun init ---
    rr.init("f1tenth_mppi_bag_viewer")
    if args.save:
        rr.save(args.save)
    else:
        rr.spawn()

    rr.send_blueprint(build_blueprint())

    # --- Typestore setup ---
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    register_custom_types(typestore)

    # --- Read bag ---
    topics_of_interest = set(TOPIC_HANDLERS.keys()) | set(COSTMAP_TOPICS.keys())

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections
                       if c.topic in topics_of_interest]

        total = sum(1 for c in reader.connections
                    if c.topic in topics_of_interest)
        print(f"Reading {total} connections across {len(connections)} topics …")

        for i, (conn, timestamp, rawdata) in enumerate(
                reader.messages(connections=connections)):
            msg = reader.deserialize(rawdata, conn.msgtype)

            if conn.topic in TOPIC_HANDLERS:
                TOPIC_HANDLERS[conn.topic](msg, timestamp)
            elif conn.topic in COSTMAP_TOPICS:
                log_occupancy_grid(msg, timestamp, COSTMAP_TOPICS[conn.topic])

            if i % 5000 == 0 and i > 0:
                print(f"  processed {i} messages …")

    print("Done — Rerun viewer should be open.")


if __name__ == "__main__":
    main()