#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import Header

from scipy.ndimage import distance_transform_edt


class LocalCostmapNode(Node):
    def __init__(self):
        super().__init__('local_costmap_node')

        # --- Local map parameters ---
        self.declare_parameter('local_map.size', 10.0)
        self.declare_parameter('local_map.resolution', 0.05)
        self.declare_parameter('local_map.inflation_radius', 0.3)
        self.declare_parameter('local_map.lethal_cost', 100)
        self.declare_parameter('local_map.inscribed_cost', 99)
        self.declare_parameter('local_map.decay_power', 2.0)
        self.declare_parameter('local_map.publish_rate_hz', 20.0)

        # --- Topics ---
        self.declare_parameter('topics.scan', '/scan')
        self.declare_parameter('topics.odom', '/ego_racecar/odom')
        self.declare_parameter('topics.costmap', '/local_costmap')

        # --- Frames ---
        self.declare_parameter('frames.costmap', 'ego_racecar/base_link')

        # --- Local map parameters ---
        self.costmap_size = self.get_parameter('local_map.size').value
        self.resolution = self.get_parameter('local_map.resolution').value
        self.inflation_radius = self.get_parameter('local_map.inflation_radius').value
        self.lethal_cost = self.get_parameter('local_map.lethal_cost').value
        self.inscribed_cost = self.get_parameter('local_map.inscribed_cost').value
        self.decay_power = self.get_parameter('local_map.decay_power').value
        publish_rate = self.get_parameter('local_map.publish_rate_hz').value

        # --- Topics ---
        scan_topic = self.get_parameter('topics.scan').value
        odom_topic = self.get_parameter('topics.odom').value
        costmap_topic = self.get_parameter('topics.costmap').value

        # --- Frames ---
        self.costmap_frame = self.get_parameter('frames.costmap').value

        # Grid dimensions
        self.grid_width = int(self.costmap_size / self.resolution)
        self.grid_height = int(self.costmap_size / self.resolution)
        self.origin_offset = self.costmap_size / 2.0  # robot at center

        # Precompute inflation kernel radius in cells
        self.inflation_cells = int(math.ceil(self.inflation_radius / self.resolution))

        # State
        self.latest_scan: LaserScan | None = None
        self.latest_odom: Odometry | None = None

        # --- Subscribers ---
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self._scan_cb, scan_qos
        )
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self._odom_cb, scan_qos
        )

        # --- Publisher ---
        self.costmap_pub = self.create_publisher(OccupancyGrid, costmap_topic, 10)

        # --- Timer ---
        period = 1.0 / publish_rate
        self.timer = self.create_timer(period, self._publish_costmap)

        self.get_logger().info(
            f'Local costmap node started: {self.grid_width}x{self.grid_height} '
            f'@ {self.resolution}m/cell, inflation={self.inflation_radius}m'
        )

    # Callbacks
    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    # Costmap generation
    def _publish_costmap(self):
        if self.latest_scan is None:
            return

        scan = self.latest_scan
        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)

        # --- Ray-cast scan points into grid ---
        angles = np.arange(
            scan.angle_min,
            scan.angle_min + len(scan.ranges) * scan.angle_increment,
            scan.angle_increment,
        )
        ranges = np.array(scan.ranges, dtype=np.float32)

        # Mask out invalid ranges
        valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        angles = angles[: len(ranges)]  # safety trim
        valid = valid[: len(angles)]

        r = ranges[valid]
        a = angles[valid]

        # Scan points in base_link frame (x forward, y left)
        xs = r * np.cos(a)
        ys = r * np.sin(a)

        # Convert to grid coords (origin at bottom-left, robot at center)
        gxs = ((xs + self.origin_offset) / self.resolution).astype(np.int32)
        gys = ((ys + self.origin_offset) / self.resolution).astype(np.int32)

        # Bounds check
        in_bounds = (
            (gxs >= 0) & (gxs < self.grid_width) &
            (gys >= 0) & (gys < self.grid_height)
        )
        gxs = gxs[in_bounds]
        gys = gys[in_bounds]

        # Mark obstacle cells as lethal
        grid[gys, gxs] = self.lethal_cost

        if self.inflation_cells > 0:
            grid = self._inflate(grid)

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.costmap_frame

        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height

        msg.info.origin = Pose()
        msg.info.origin.position.x = -self.origin_offset
        msg.info.origin.position.y = -self.origin_offset
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # Row-major, y then x
        msg.data = grid.flatten().tolist()

        self.costmap_pub.publish(msg)

    def _inflate(self, grid: np.ndarray) -> np.ndarray:
        """
        Inflate lethal cells using EDT (Euclidean Distance Transform).
        Produces a smooth cost falloff from lethal -> 0 over inflation_radius.
        """
        # Binary obstacle mask: 1 = free, 0 = obstacle (EDT convention)
        obstacle_mask = grid < self.lethal_cost
        dist = distance_transform_edt(obstacle_mask) * self.resolution  # in meters

        # Compute inflated costs: lethal at obstacle, decaying outward
        inflated = np.where(
            dist == 0.0,
            self.lethal_cost,
            np.where(
                dist < self.inflation_radius,
                np.clip(
                    self.inscribed_cost
                    * (1.0 - (dist / self.inflation_radius) ** self.decay_power),
                    0,
                    self.inscribed_cost,
                ).astype(np.int8),
                0,
            ),
        ).astype(np.int8)

        return inflated


def main(args=None):
    rclpy.init(args=args)
    node = LocalCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()