import os
import threading
from time import time
from typing import NamedTuple

import numpy as np
import cv2
import jax
import jax.numpy as jnp
from jax import jit, vmap

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import tf2_ros
import tf_transformations
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from autoware_auto_planning_msgs.msg import Trajectory
from geometry_msgs.msg import TransformStamped, PoseStamped
from visualization_msgs.msg import MarkerArray

# from .TimeUtil import TimeUtil

# --- Configuration & Constants ---
class CarParams:
    """Physical parameters for the racecar model."""
    mu, C_Sf, C_Sr = 1.0489, 4.718, 5.4562
    lf, lr, h, mass, Iz = 0.15875, 0.17145, 0.074, 3.74, 0.04712
    s_min, s_max = -0.9189, 0.9189
    a_min, a_max = -10.0, 9.51
    v_max = 20.0
    alpha_sat = 0.19
    Fx_split = 0.45

class MPPIConfig:
    """Hyperparameters for the MPPI controller."""
    T = 30           # Horizon length
    dt = 0.05        # Time step
    samples = 1500   # Number of trajectories to sample
    m = 2            # Control dimensions (steering_rate, acceleration)

class SDF(NamedTuple):
    grid: jnp.ndarray
    origin: jnp.ndarray
    inv_res: jnp.float32

# --- JAX Physics & Cost Functions ---

@jit
def wrap(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi

@jit
def get_ref_point(traj, x_curr):
    """Finds the closest point on the reference trajectory via interpolation."""
    def dist_sq(i):
        return jnp.sum((traj[i, :2] - x_curr[:2])**2)
    
    idx_range = jnp.arange(len(traj) - 1)
    dists = vmap(dist_sq)(idx_range)
    i = jnp.argmin(dists)
    
    # Linear interpolation factor 't'
    p1, p2 = traj[i], traj[i+1]
    vec_p1p2 = p2[:2] - p1[:2]
    vec_p1curr = x_curr[:2] - p1[:2]
    t = jnp.clip(jnp.dot(vec_p1p2, vec_p1curr) / (jnp.sum(vec_p1p2**2) + 1e-6), 0, 1)
    
    return p1 * (1 - t) + p2 * t

@jit
def vehicle_dynamics(x, u):
    """Bicycle model with lateral slip and tire dynamics."""
    # x: [x, y, steer, vx, vy, yaw, yaw_rate] | u: [steer_rate, ax]
    steer, vx, vy, yaw, r = x[2], x[3], x[4], x[5], x[6]
    vx_safe = jnp.maximum(vx, 0.1)

    # Forces
    Fx_total = CarParams.mass * (u[1] - vy * r)
    Fx_f, Fx_r = CarParams.Fx_split * Fx_total, (1 - CarParams.Fx_split) * Fx_total
    
    # Slip angles and Lateral Forces
    alpha_f = jnp.arctan2(vy + CarParams.lf * r, vx_safe) - steer
    alpha_r = jnp.arctan2(vy - CarParams.lr * r, vx_safe)
    
    def fy_tire(alpha, C):
        return jnp.where(jnp.abs(alpha) <= CarParams.alpha_sat, -C * alpha, -C * CarParams.alpha_sat * jnp.sign(alpha))
    
    Fy_f, Fy_r = fy_tire(alpha_f, CarParams.C_Sf), fy_tire(alpha_r, CarParams.C_Sr)

    # Accelerations
    dvx = (Fx_f * jnp.cos(steer) - Fy_f * jnp.sin(steer) + Fx_r) / CarParams.mass + vy * r
    dvy = (Fy_f * jnp.cos(steer) + Fx_f * jnp.sin(steer) + Fy_r) / CarParams.mass - vx * r
    dr = (CarParams.lf * (Fy_f * jnp.cos(steer) + Fx_f * jnp.sin(steer)) - CarParams.lr * Fy_r) / CarParams.Iz

    return jnp.array([vx * jnp.cos(yaw) - vy * jnp.sin(yaw), vx * jnp.sin(yaw) + vy * jnp.cos(yaw), u[0], dvx, dvy, r, dr])

@jit
def rollout_and_cost(x0, u_seq, ref_traj, Q, R, sdf):
    """Predicts future states and computes the total cost of a control sequence."""
    def step(x, u):
        x_next = x + vehicle_dynamics(x, u) * MPPIConfig.dt
        x_next = x_next.at[2].set(jnp.clip(x_next[2], CarParams.s_min, CarParams.s_max))
        
        ref = get_ref_point(ref_traj, x_next)
        err = ref - x_next
        err = err.at[5].set(wrap(err[5])) # Yaw error
        
        cost = err.T @ Q @ err + u.T @ R @ u
        return x_next, cost

    final_state, costs = jax.lax.scan(step, x0, u_seq)
    return jnp.sum(costs)

# --- ROS2 Controller Node ---

class MppiController(Node):
    def __init__(self):
        super().__init__('mppi_controller')
        self._init_params()
        self._init_pubs_subs()
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1/50.0, self.update_control)
        
        self.u_nom = jnp.zeros((MPPIConfig.T, MPPIConfig.m))
        self.key = jax.random.PRNGKey(int(time()))
        self.sdf_data = None
        self.planner_traj, self.odom = None, None
        self.last_steering = 0.0

    def _init_params(self):
        # Weight matrices for the cost function
        self.Q = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])) 
        self.R = jnp.diag(jnp.array([1.0, 1.0])) # [steer_rate, ax]
        self.temp = 0.3
        self.noise_std = jnp.array([3.6, 3.1]) # Steering rate and accel noise

    def _init_pubs_subs(self):
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.create_subscription(Trajectory, '/planner_traj', lambda m: setattr(self, 'planner_traj', m), 1)
        self.create_subscription(Odometry, '/ego_racecar/noisy_odom', lambda m: setattr(self, 'odom', m), 1)
        self.create_subscription(OccupancyGrid, '/sim_map', self.map_callback, 10)

    def map_callback(self, msg):
        """Processes the occupancy grid into a Signed Distance Field (SDF)."""
        data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        occ = (data > 0).astype(np.uint8) * 255
        free = (data <= 0).astype(np.uint8) * 255
        sdf_map = (cv2.distanceTransform(free, cv2.DIST_L2, 5) - 
                   cv2.distanceTransform(occ, cv2.DIST_L2, 5)) * msg.info.resolution
        
        self.sdf_data = SDF(
            grid=jax.device_put(jnp.flipud(sdf_map)),
            origin=jnp.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            inv_res=1.0 / msg.info.resolution
        )

    def update_control(self):
        if self.planner_traj is None or self.odom is None or self.sdf_data is None:
            return

        try:
            # 1. State Estimation
            t = self.tf_buffer.lookup_transform('sim_map', 'ego_racecar/base_link', rclpy.time.Time())
            pos = t.transform.translation
            rot = t.transform.rotation
            _, _, heading = tf_transformations.euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
            
            x0 = jnp.array([pos.x, pos.y, self.last_steering, 
                            self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, 
                            heading, self.odom.twist.twist.angular.z])

            # 2. Reference Path Pre-processing
            ref_points = jnp.array([[p.pose.position.x, p.pose.position.y, 0, p.longitudinal_velocity_mps, 0, 0, 0] 
                                    for p in self.planner_traj.points])

            # 3. MPPI Sampling
            self.key, subkey = jax.random.split(self.key)
            noise = jax.random.normal(subkey, (MPPIConfig.samples, MPPIConfig.T, MPPIConfig.m)) * self.noise_std
            u_candidates = jnp.clip(self.u_nom + noise, 
                                    jnp.array([CarParams.s_min, CarParams.a_min]), 
                                    jnp.array([CarParams.s_max, CarParams.a_max]))

            # 4. Cost Evaluation (Vmap-accelerated)
            costs = vmap(lambda u: rollout_and_cost(x0, u, ref_points, self.Q, self.R, self.sdf_data))(u_candidates)
            
            # 5. Softmax-weighting and Synthesis
            weights = jax.nn.softmax(-(costs - jnp.min(costs)) / self.temp)
            delta_u = jnp.sum(weights[:, None, None] * noise, axis=0)
            
            self.u_nom = jnp.clip(self.u_nom + delta_u, 
                                  jnp.array([CarParams.s_min, CarParams.a_min]), 
                                  jnp.array([CarParams.s_max, CarParams.a_max]))

            # 6. Command Execution
            u_apply = self.u_nom[0]
            self.last_steering = np.clip(self.last_steering + (1/50.0) * u_apply[0], CarParams.s_min, CarParams.s_max)
            target_vel = np.clip(x0[3] + (1/50.0) * u_apply[1], 0.0, CarParams.v_max)

            self.publish_drive(self.last_steering, target_vel)
            self.u_nom = jnp.roll(self.u_nom, -1, axis=0).at[-1].set(0.0)

        except Exception as e:
            self.get_logger().error(f"Control Loop Error: {e}")

    def publish_drive(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MppiController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()