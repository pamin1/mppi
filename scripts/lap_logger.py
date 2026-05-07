#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Int16

class LapLogger(Node):
    def __init__(self):
        super().__init__('lap_logger')

        self.declare_parameter('sx', 0.0)
        self.declare_parameter('sy', 0.0)
        self.declare_parameter('stheta', 0.0)

        self.sx = self.get_parameter('sx').value
        self.sy = self.get_parameter('sy').value
        self.stheta = self.get_parameter('stheta').value
        self.stheta_rad = math.radians(self.stheta)

        self.get_logger().info(f"sx: {self.sx}, sy: {self.sy}, stheta: {self.stheta}")

        self.crossing_radius = 1.0
        self.near_start = True
        self.toggle_count = 0
        self.lap_count = 0
        self.lap_start_time = self.get_clock().now()

        cos_t = math.cos(-self.stheta)
        sin_t = math.sin(-self.stheta)
        self.start_rot = np.array([[cos_t, -sin_t], [sin_t,  cos_t]])

        self.odom_sub = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        
        self.lap_pub = self.create_publisher(Int16, "/laps", 10)

    def odom_cb(self, msg):
        elapsed_since_start = (self.get_clock().now() - self.lap_start_time).nanoseconds / 1e9
        if elapsed_since_start < 5.0:
            return
        
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        dx = x - self.sx
        dy = y - self.sy
        dist = math.sqrt(dx * dx + dy * dy)

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # Check heading is within 45° of start heading
        heading_diff = abs(math.atan2(math.sin(yaw - self.stheta_rad), math.cos(yaw - self.stheta_rad)))

        is_near = dist < 3.0 and heading_diff < math.pi / 4.0

        if is_near and not self.near_start:
            self.near_start = True
            self.toggle_count += 1
        elif not is_near and self.near_start:
            self.near_start = False
            self.toggle_count += 1

        new_laps = self.toggle_count // 2
        if new_laps > self.lap_count:
            now = self.get_clock().now()
            elapsed = (now - self.lap_start_time).nanoseconds / 1e9
            self.lap_count = new_laps
            self.lap_start_time = now
            self.get_logger().info(f'Lap {self.lap_count} completed — {elapsed:.2f}s')


def main(args=None):
    rclpy.init(args=args)
    node = LapLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()