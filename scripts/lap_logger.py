#!/usr/bin/env python3

import math
import os
import signal
import csv
import datetime

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int16
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class LapLogger(Node):
    def __init__(self):
        super().__init__('lap_logger')

        # --- Parameters ---
        self.declare_parameter('sx', 0.0)
        self.declare_parameter('sy', 0.0)
        self.declare_parameter('stheta', 0.0)
        self.declare_parameter('target_laps', 3)

        self.sx = self.get_parameter('sx').value
        self.sy = self.get_parameter('sy').value
        self.stheta = self.get_parameter('stheta').value
        self.stheta_rad = math.radians(self.stheta)
        self.target_laps = 1 + self.get_parameter('target_laps').value
        self.output_file = os.path.join(os.getcwd(), 'data', f'test_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

        # --- Finish-line geometry ---
        self.finish_cos = math.cos(self.stheta_rad)
        self.finish_sin = math.sin(self.stheta_rad)
        self.finish_perp_x = math.cos(self.stheta_rad + math.pi / 2)
        self.finish_perp_y = math.sin(self.stheta_rad + math.pi / 2)
        self.finish_line_half_width = 4.0
        self.min_lap_time = 5.0

        # --- Crossing state ---
        self.prev_signed_dist = None
        self.prev_x = None
        self.prev_y = None

        # --- Lap tracking ---
        self.lap_count = 0
        now = self.get_clock().now()
        self.start_time = now
        self.lap_start_time = now
        self.total_speed = 0.0
        self.total_msgs = 0
        self.lap_distance = 0.0
        self.last_x = None
        self.last_y = None

        # --- Collision tracking (rising edge) ---
        self.was_colliding = False
        self.total_collisions = 0
        self.lap_collisions = 0

        # --- Results ---
        self.lap_results = []

        # --- ROS interfaces ---
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_cb, 10)
        self.collision_sub = self.create_subscription(
            Bool, '/collision', self.collision_cb, 10)
        self.lap_pub = self.create_publisher(Int16, '/lap_count', 10)
        self.finish_line_pub = self.create_publisher(Marker, '/finish_line', 10)
        self.create_timer(1.0, self.publish_finish_line)

        self.get_logger().info(
            f'Lap logger started — sx={self.sx}, sy={self.sy}, '
            f'stheta={self.stheta}, target_laps={self.target_laps - 1}, '
            f'output={self.output_file}')

    def collision_cb(self, msg):
        if msg.data and not self.was_colliding:
            self.total_collisions += 1
            self.lap_collisions += 1
            self.get_logger().warn(
                f'Collision #{self.total_collisions} '
                f'(lap collisions: {self.lap_collisions})')
            self.was_colliding = True
        elif not msg.data:
            self.was_colliding = False

    def odom_cb(self, msg):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed < 5.0:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        # Accumulate speed
        self.total_speed += math.sqrt(vx * vx + vy * vy)
        self.total_msgs += 1

        # Accumulate distance
        if self.last_x is not None:
            self.lap_distance += math.hypot(x - self.last_x, y - self.last_y)
        self.last_x = x
        self.last_y = y

        # --- Finish-line crossing detection ---
        dx = x - self.sx
        dy = y - self.sy
        signed_dist = dx * self.finish_cos + dy * self.finish_sin

        crossed = False
        if self.prev_signed_dist is not None:
            if (self.prev_signed_dist < 0.0) != (signed_dist < 0.0):
                # Interpolate to find crossing point
                total = abs(signed_dist - self.prev_signed_dist)
                alpha = abs(self.prev_signed_dist) / total if total > 1e-6 else 0.5
                cx = self.prev_x + alpha * (x - self.prev_x)
                cy = self.prev_y + alpha * (y - self.prev_y)
                lateral = abs((cx - self.sx) * self.finish_perp_x +
                              (cy - self.sy) * self.finish_perp_y)
                crossed = lateral < self.finish_line_half_width

        self.prev_signed_dist = signed_dist
        self.prev_x = x
        self.prev_y = y

        if not crossed:
            return

        # --- Cooldown ---
        now = self.get_clock().now()
        lap_time = (now - self.lap_start_time).nanoseconds / 1e9
        if lap_time < self.min_lap_time:
            return

        # --- Register lap ---
        self.lap_count += 1
        self.get_logger().info(
            f'Lap {self.lap_count} — {lap_time:.2f}s, '
            f'{self.lap_collisions} collisions')

        if self.lap_count > 1:
            self.lap_results.append({
                'lap': self.lap_count - 1,
                'time': round(lap_time, 2),
                'collisions': self.lap_collisions,
                'avg_speed': round(
                    self.total_speed / max(self.total_msgs, 1), 2),
                'distance': round(self.lap_distance, 2),
            })

        # Reset per-lap counters
        self.lap_start_time = now
        self.lap_collisions = 0
        self.total_speed = 0.0
        self.total_msgs = 0
        self.lap_distance = 0.0

        lap_msg = Int16()
        lap_msg.data = self.lap_count
        self.lap_pub.publish(lap_msg)

        if self.lap_count >= self.target_laps:
            self.write_results()
            self.get_logger().info(
                f'Target {self.target_laps - 1} laps reached — shutting down')
            os.killpg(os.getpgid(os.getpid()), signal.SIGTERM)

    def publish_finish_line(self):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'finish_line'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.color.g = 1.0
        marker.color.a = 1.0

        w = self.finish_line_half_width
        p1 = Point(x=self.sx + w * self.finish_perp_x,
                    y=self.sy + w * self.finish_perp_y, z=0.0)
        p2 = Point(x=self.sx - w * self.finish_perp_x,
                    y=self.sy - w * self.finish_perp_y, z=0.0)
        marker.points = [p1, p2]
        self.finish_line_pub.publish(marker)

    def write_results(self):
        try:
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['lap', 'time', 'collisions',
                                   'avg_speed', 'distance'])
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