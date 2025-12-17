#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import pandas as pd
import numpy as np
from autoware_auto_planning_msgs.msg import Trajectory, TrajectoryPoint
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
import tf_transformations


class GlobalTrajectoryPublisher(Node):
    def __init__(self):
        super().__init__("global_trajectory_publisher")

        # traj and visualization pubs
        self.trajectory_pub = self.create_publisher(Trajectory, "/planner_traj", 10)
        self.viz_pub = self.create_publisher(Path, '/visual_raceline', 10)
        
        # load path
        self.csv_path = f"{os.getcwd()}/src/mppi/resources/Spielberg_map_optimized.csv"
        self.load_and_publish()

    def load_and_publish(self):
        try:
            df = pd.read_csv(self.csv_path)
            self.get_logger().info(f"Loaded {len(df)} points from {self.csv_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load CSV: {e}")
            return

        msg = Trajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for _, row in df.iterrows():
            point = TrajectoryPoint()

            point.pose.position.x = float(row["x"])
            point.pose.position.y = float(row["y"])

            q = tf_transformations.quaternion_from_euler(0, 0, row["yaw"])
            point.pose.orientation.x = q[0]
            point.pose.orientation.y = q[1]
            point.pose.orientation.z = q[2]
            point.pose.orientation.w = q[3]

            point.longitudinal_velocity_mps = float(row["vx"])

            # if these get added to the path optimization later on
            if "acceleration" in row:
                point.acceleration_mps2 = float(row["acceleration"])
            if "kappa" in row:
                point.front_wheel_angle_rad = np.arctan(0.33 * row["kappa"])

            msg.points.append(point)

        self.trajectory_pub.publish(msg)
        self.get_logger().info("Global trajectory published.")

        viz_msg = Path()
        viz_msg.header.frame_id = "map"
        viz_msg.header.stamp = self.get_clock().now().to_msg()

        for _, row in df.iterrows():
            pose = PoseStamped()
            pose.header = viz_msg.header
            pose.pose.position.x = float(row["x"])
            pose.pose.position.y = float(row["y"])
            viz_msg.poses.append(pose)

        self.viz_pub.publish(viz_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalTrajectoryPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
