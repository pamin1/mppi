#!/usr/bin/env python3

import math
import os
import signal
import csv

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int16


class LapLogger(Node):
    def __init__(self):
        super().__init__('lap_logger')

        self.declare_parameter('sx', 0.0)
        self.declare_parameter('sy', 0.0)
        self.declare_parameter('stheta', 0.0)
        self.declare_parameter('target_laps', 3)
        self.declare_parameter('output_file', '/workspace/lap_results.csv')

        self.sx = self.get_parameter('sx').value
        self.sy = self.get_parameter('sy').value
        self.stheta = self.get_parameter('stheta').value
        self.stheta_rad = math.radians(self.stheta)
        self.target_laps = self.get_parameter('target_laps').value
        self.output_file = self.get_parameter('output_file').value

        # Lap tracking
        self.near_start = True
        self.toggle_count = 0
        self.lap_count = 0
        self.lap_start_time = self.get_clock().now()
        self.start_time = self.get_clock().now()
        self.total_speed = 0
        self.total_msgs = 0
        self.lap_distance = 0.0
        self.last_x = None
        self.last_y = None

        # Collision tracking — rising edge only
        self.was_colliding = False
        self.total_collisions = 0
        self.lap_collisions = 0

        # Results storage
        self.lap_results = []

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.collision_sub = self.create_subscription(
            Bool, '/collision', self.collision_cb, 10)

        # Publisher
        self.lap_pub = self.create_publisher(Int16, '/lap_count', 10)

        self.get_logger().info(
            f'Lap logger started — sx={self.sx}, sy={self.sy}, '
            f'stheta={self.stheta}, target={self.target_laps}')

    def collision_cb(self, msg):
        if msg.data and not self.was_colliding:
            self.total_collisions += 1
            self.lap_collisions += 1
            self.get_logger().warn(f'Collision #{self.total_collisions} (lap collisions: {self.lap_collisions})')
            self.was_colliding = True
        elif not msg.data:
            self.was_colliding = False

    def odom_cb(self, msg):
        # Grace period — ignore first 5 seconds
        elapsed_since_start = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed_since_start < 5.0:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        dx = x - self.sx
        dy = y - self.sy
        dist = math.sqrt(dx * dx + dy * dy)

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        # extract the speed at this time step
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.total_speed += math.sqrt(vx * vx + vy * vy)
        self.total_msgs += 1

        # Heading difference normalized to [-pi, pi]
        heading_diff = abs(math.atan2(math.sin(yaw - self.stheta_rad),
                                       math.cos(yaw - self.stheta_rad)))

        is_near = dist < 3.0 and heading_diff < math.pi / 4.0

        # total distance traveled
        if self.last_x is not None:
            ddx = x - self.last_x
            ddy = y - self.last_y
            self.lap_distance += math.sqrt(ddx * ddx + ddy * ddy)
        self.last_x = x
        self.last_y = y

        if is_near and not self.near_start:
            self.near_start = True
            self.toggle_count += 1
        elif not is_near and self.near_start:
            self.near_start = False
            self.toggle_count += 1

        new_laps = self.toggle_count // 2
        if new_laps > self.lap_count:
            now = self.get_clock().now()
            lap_time = (now - self.lap_start_time).nanoseconds / 1e9

            self.lap_count = new_laps
            self.get_logger().info(
                f'Lap {self.lap_count} — {lap_time:.2f}s, '
                f'{self.lap_collisions} collisions')

            self.lap_results.append({
                'lap': self.lap_count,
                'time': round(lap_time, 2),
                'collisions': self.lap_collisions,
                'avg_speed': round(self.total_speed / max(self.total_msgs, 1), 2),
                'distance': round(self.lap_distance, 2),
            })

            # Reset per-lap counters
            self.lap_start_time = now
            self.lap_collisions = 0
            self.total_speed = 0.0
            self.total_msgs = 0
            self.lap_distance = 0

            lap_msg = Int16()
            lap_msg.data = self.lap_count
            self.lap_pub.publish(lap_msg)

            if self.lap_count >= self.target_laps:
                self.write_results()
                self.get_logger().info(
                    f'Target {self.target_laps} laps reached — shutting down')
                os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)

            # self.write_results()


    def write_results(self):
        try:
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['lap', 'time', 'collisions', 'avg_speed', 'distance'])
                writer.writeheader()
                writer.writerows(self.lap_results)
            self.get_logger().info(f'Results written to {self.output_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to write results: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LapLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()