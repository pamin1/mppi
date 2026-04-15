#!/usr/bin/env python3
import os
import pandas as pd
import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from tf2_ros import TransformListener, Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf_transformations
from ament_index_python.packages import get_package_share_directory


class GlobalTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("global_trajectory_publisher")

        self.map_frame = "map"
        self.map = None
        self.odom = None
        self.lookahead = 75

        self.declare_parameter('control_dt', 0.02)
        self.control_dt = self.get_parameter('control_dt').value

        # load path
        share_dir = get_package_share_directory("mppi")

        self.declare_parameter('trajectory_path', 'Spielberg_map_optimized')
        fn = self.get_parameter('trajectory_path').value

        self.get_logger().info(f"Loading trajectory: {fn}")

        df = pd.read_csv(fn)
        self.x = df["x"].to_numpy()
        self.y = df["y"].to_numpy()
        self.speed = df["vx"].to_numpy()
        self.yaw = df["yaw"].to_numpy()
        self.kappa = df["kappa"].to_numpy()
        self.N = df.shape[0]
        self.get_logger().info(f"Got {self.N} points")

        # publishers
        self.trajectory_pub = self.create_publisher(Trajectory, "/planner_traj", 10)
        self.viz_pub = self.create_publisher(Path, "/visual_raceline", 10)

        # subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            "/ego_racecar/odom",  # Adjust topic name if needed
            self.odom_callback,
            10,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 1
        )

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to periodically publish path
        self.timer = self.create_timer(0.1, self.update_traj)

    def odom_callback(self, msg):
        self.odom = msg

    def map_callback(self, msg):
        self.map = msg

    def update_traj(self):
        if self.odom is None:
            self.get_logger().info("Odometry is not ready")
            return
        try:
            now = self.get_clock().now()
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame, "ego_racecar/base_link", rclpy.time.Time()
            )

            x = trans.transform.translation.x
            y = trans.transform.translation.y

            q = trans.transform.rotation
            _, _, yaw = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

            speed = self.odom.twist.twist.linear.x

            dx = self.x - x
            dy = self.y - y

            dist_sq = dx**2 + dy**2
            # min_idx = (np.argmin(dist_sq) + self.N - 25) % self.N
            min_idx = np.argmin(dist_sq)

            # Temporal resampling: find the raceline point corresponding to
            # t = i * control_dt for each output index i, so refTrajectory[i]
            # matches what the MPPI kernel expects at time i * dt.
            window_size = min(self.N, 500)
            idx_window = [(min_idx + j) % self.N for j in range(window_size)]

            wx = self.x[idx_window]
            wy = self.y[idx_window]
            wv = np.maximum(self.speed[idx_window], 0.1)

            ds = np.sqrt(np.diff(wx)**2 + np.diff(wy)**2)
            dt_segs = ds / wv[:-1]
            t_cumulative = np.concatenate([[0.0], np.cumsum(dt_segs)])

            target_times = np.arange(1, self.lookahead + 1) * self.control_dt
            wi = np.searchsorted(t_cumulative, target_times)
            wi = np.clip(wi, 0, window_size - 1)

            ri = [idx_window[w] for w in wi]
            xx = self.x[ri]
            yy = self.y[ri]
            vv = self.speed[ri]
            hh = self.yaw[ri]
            kk = self.kappa[ri]

            traj_msg = Trajectory()
            traj_msg.header.stamp = self.get_clock().now().to_msg()
            traj_msg.header.frame_id = self.map_frame

            path_msg = Path()
            path_msg.header.frame_id = self.map_frame
            path_msg.header.stamp = self.get_clock().now().to_msg()

            for i in range(self.lookahead):
                point = TrajectoryPoint()
                point.pose.position.x = xx[i]
                point.pose.position.y = yy[i]

                # 1. Correct Heading (Yaw)
                qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0, 0, hh[i])
                point.pose.orientation.x = qx
                point.pose.orientation.y = qy
                point.pose.orientation.z = qz
                point.pose.orientation.w = qw

                # 2. Correct Speed and Yaw Rate
                point.longitudinal_velocity_mps = vv[i]
                point.heading_rate_rps = vv[i] * kk[i]

                # 3. Handle Steering (Front Wheel Angle)
                # If you don't have optimal steering in your CSV, set it to 0.0
                # and let MPPI calculate the effort, or estimate it via L * kappa
                L = 0.33  # wheel base

                max_steering = np.deg2rad(30.0)  # F1TENTH typical limit
                steering = np.clip(np.arctan(L * kk[i]), -max_steering, max_steering)
                point.front_wheel_angle_rad = steering

                traj_msg.points.append(point)

                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = xx[i]
                pose.pose.position.y = yy[i]
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 0.0
                path_msg.poses.append(pose)

            self.trajectory_pub.publish(traj_msg)
            self.viz_pub.publish(path_msg)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Failed to get transform: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GlobalTrajectoryPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
