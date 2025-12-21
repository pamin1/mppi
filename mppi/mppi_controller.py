import os
import threading
from time import time
from math import isnan
from typing import NamedTuple

import numpy as np
import cv2
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix

import jax
import jax.numpy as jnp
from jax import jit, vmap

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from laser_geometry import LaserProjection
import sensor_msgs_py.point_cloud2 as pc2

from sklearn.cluster import DBSCAN

import tf2_ros
from tf2_ros import TransformListener, Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

import tf_transformations

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import TransformStamped, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from autoware_auto_planning_msgs.msg import Trajectory
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan

from .TimeUtil import TimeUtil

global BASEDIR
BASEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# comment out to force CPU mode if needed
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

tu = TimeUtil(True)

mu = 1.0489
C_Sf = 4.718
C_Sr = 5.4562
lf = 0.15875
lr = 0.17145
h = 0.074
mass = 3.74
Iz = 0.04712
s_min = -0.9189
s_max = 0.9189
sv_min = -3.2
sv_max = 3.2
v_switch = 7.319
a_min = -10.0
a_max = 9.51
v_min = -5.0
v_max = 20.0
width = 0.31
length = 0.58
height = 0.22
g = 9.81
alpha_sat = 0.19
Fx_split = 0.45

T = 30
dt = 0.05
m = 2
samples = 1500

class SDF(NamedTuple):
    grid: jnp.ndarray
    origin: jnp.ndarray
    inv_res: jnp.float32
    
class MppiController(Node):
    def __init__(self):
        super().__init__('mppi_controller')
        self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.map_frame = 'sim_map'

        self.last_steering = 0
        self.control_freq = 50
        
        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # subscribers
        self.create_subscription(Trajectory, '/planner_traj', self.traj_callback, 1)
        self.create_subscription(Odometry, '/ego_racecar/noisy_odom', self.odom_callback, 1)
        # self.create_subscription(LaserScan, '/scan_filt', self.laser_callback, 10)
        self.create_subscription(OccupancyGrid, '/sim_map', self.map_callback, self.qos)
        
        # publishers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.rollouts_pub = self.create_publisher(MarkerArray, '/visual_mppi_samples', 1)
        self.visual_pub = self.create_publisher(Path, '/visual_mppi', 1)
        
        # command timer
        self.timer = self.create_timer(1/self.control_freq, self.update_control)
        self.planner_traj = None
        self.odom = None
        self.clusters = None

        # variables
        self.map = None
        self.point_cloud = None
        self.sdf = None
        self.res, self.ox, self.oy, self.gx, self.gy = None, None, None, None, None
        self.w, self.h = None, None
        self._evt = threading.Event()
        self.good = []

        # Cached JAX-device copy of the signed-distance field
        self._sdf_jax = None

        # noise scale
        steer_rate_lim = 3.6
        ax_lim = 9.5
        self.u_noise_scale = np.array([steer_rate_lim, ax_lim/3])

        self.u_nom = np.zeros((T, m), dtype=np.float32)
        self.key = jax.random.PRNGKey(int(time()*1e9))

        # Simulation/control discretisation
        self.dt = dt  # global constant defined at top of file, used inside decode()

        self.declare_parameters(
            namespace='',
            parameters=[
                ('mppi_weights.x', 1.0),
                ('mppi_weights.y', 1.0),
                ('mppi_weights.steering', 1.0),
                ('mppi_weights.u', 1.0),
                ('mppi_weights.v', 1.0),
                ('mppi_weights.yaw', 1.0),
                ('mppi_weights.yaw_rate', 1.0),
                ('mppi_weights.R_steering', 1.0),
                ('mppi_weights.R_acceleration', 1.0),
                ('mppi_weights.temperature', 0.3),
            ]
        )
        self.Q = jnp.diag(jnp.array([
            self.get_parameter('mppi_weights.x').value,
            self.get_parameter('mppi_weights.y').value, 
            self.get_parameter('mppi_weights.steering').value,
            self.get_parameter('mppi_weights.u').value,
            self.get_parameter('mppi_weights.v').value,
            self.get_parameter('mppi_weights.yaw').value,
            self.get_parameter('mppi_weights.yaw_rate').value,
        ]))
        self.R = jnp.diag(jnp.array([
            self.get_parameter('mppi_weights.R_steering').value,
            self.get_parameter('mppi_weights.R_acceleration').value,
        ]))
        
        self.temperature = self.get_parameter('mppi_weights.temperature').value
        self.get_logger().info(f"Loaded cost weights:\n{self.Q}")
        
        # Log JAX device information
        jax_devices = jax.devices()
        self.get_logger().info(f"JAX using device: {jax_devices[0]} (backend: {jax.default_backend()})")
        if jax.default_backend() == 'gpu':
            self.get_logger().info("GPU acceleration enabled for MPPI!")
        else:
            self.get_logger().warn("Running on CPU")

    def traj_callback(self, msg: Trajectory):
        self.planner_traj = msg
        
    def odom_callback(self, msg: Odometry):
        self.odom = msg
        
    def update_control(self):
        if (self.planner_traj is None or self.odom is None):
            self.get_logger().info('planner_traj and odom not ready')
            return
        try:
            t0 = time()
            tu.s('total')
            
            # Lookup latest transform from global frame to base_link
            tu.s('transform')
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame, 'ego_racecar/base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            _, _, heading = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            vx = self.odom.twist.twist.linear.x
            vy = self.odom.twist.twist.linear.y
            yaw_rate = self.odom.twist.twist.angular.z
            
            x0 = np.array([x,y, self.last_steering, vx, vy, heading, yaw_rate],dtype=np.float32) # getting the current state
            tu.e('transform')

            tu.s('state_prep')
            # decode planner local path
            # find point closest to vehicle
            def decode(point, x0, prev_point=None):
                x = point.pose.position.x
                y = point.pose.position.y
                qx = point.pose.orientation.x
                qy = point.pose.orientation.y
                qz = point.pose.orientation.z
                qw = point.pose.orientation.w
                # Use the orientation of the trajectory point (not the vehicle)
                _, _, yaw = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])
                u = point.longitudinal_velocity_mps
                if prev_point is not None:
                    _, _, prev_yaw = tf_transformations.euler_from_quaternion([
                        prev_point.pose.orientation.x,
                        prev_point.pose.orientation.y,
                        prev_point.pose.orientation.z,
                        prev_point.pose.orientation.w,
                    ])
                    # Handle yaw wraparound
                    dyaw = np.arctan2(np.sin(yaw - prev_yaw), np.cos(yaw - prev_yaw))
                    yaw_rate = dyaw / self.dt
                else:
                    yaw_rate = 0.0
                # Approximate the feed-forward steering required to follow the path curvature
                L = lf + lr
                if prev_point is not None and u > 0.5:
                    ds = np.hypot(x - prev_point.pose.position.x,
                                   y - prev_point.pose.position.y) + 1e-6
                    kappa = dyaw / ds
                    steering = -np.arctan(L * kappa)
                    v = np.sqrt(a_max / kappa)
                else:
                    steering = 0.0
                    v = 0.0
            
                return np.array([x, y, steering, u, v, yaw, yaw_rate])

            # Build reference trajectory from ALL planner points so nearest-point logic works
            traj_list = []
            prev_p = None
            for p in self.planner_traj.points:
                traj_list.append(decode(p, x0, prev_p))
                prev_p = p
            # Convert to ndarray for JAX cost evaluation
            traj = np.asarray(traj_list, dtype=np.float32)
            tu.e('state_prep')

            tu.s('mppi_prep')
            # state x: [x, y, steering, u, v, heading, yaw_rate]
            # control u: [ steering_rate, ax]
            
            # ensure nominal sequence length matches current horizon
            if self.u_nom.shape[0] != T:
                self.u_nom = np.zeros((T, m), dtype=np.float32)
            
            tu.e('mppi_prep')
            
            tu.s('mppi_compute')
            try:
                # sample noise, split keys to maintain randomness
                self.key, k_sample = jax.random.split(self.key)
                noise = jax.random.normal(k_sample, shape=(samples, T, m)) * self.u_noise_scale # TODO: try other noise distributions
                
                # candidate trajectories centred on the nominal sequence
                u_nom = jnp.asarray(self.u_nom)
                u_vec = jnp.clip(u_nom + noise,
                            jnp.array([s_min, a_min]),
                            jnp.array([s_max, a_max]))
                u_vec = jnp.asarray(u_vec.reshape((-1,T,m)))

                # Most expensive operation - rollout and cost evaluation
                sdf = self.getSDF()
                cost_vec = np.array(vmap(lambda u:rollout_cost(traj,x0,u, self.Q, self.R, sdf))(u_vec).block_until_ready())
                cost_vec = jnp.nan_to_num(cost_vec, posinf=1e9, neginf=1e9)
                
            except Exception as e:
                self.get_logger().error(f'JAX computation failed: {e}')
                cost_vec = np.ones(samples) * 1e9
            tu.e('mppi_compute')

            tu.s('control_synthesis')
            # synthesize control
            cost_min = np.min(cost_vec)
            delta_J = cost_vec - cost_min  # ≥ 0

            # Characteristic cost scale – prevents λ from being too small/large
            cost_spread = np.mean(delta_J) + 1e-6  # always > 0
            lam = self.temperature * cost_spread      # effective λ (Williams et al.)

            # Stable normalised exponentials (softmax-style implementation)
            scaled = -delta_J / lam
            # Shift by max for numerical stability
            scaled -= np.max(scaled)
            weight_vec = np.exp(scaled)
            weight_vec /= np.sum(weight_vec)

            # path-integral control update
            delta_u = np.sum(weight_vec.reshape(-1,1,1) * np.array(noise), axis=0)
            delta_u = jnp.clip(
                delta_u,
                jnp.array([s_min, a_min]),
                jnp.array([s_max, a_max]))
            self.u_nom = self.u_nom + delta_u
            
            # keep the whole seq inside limits
            self.u_nom = jnp.clip(
                self.u_nom,
                jnp.array([s_min, a_min]),
                jnp.array([s_max, a_max]))
            
            # SMPPI safe shielding
            
            
            u_apply = self.u_nom[0].copy()

            # shift sequence forward for next iteration
            self.u_nom = np.roll(self.u_nom, -1, axis=0)
            self.u_nom[-1] = 0.0

            steering = self.last_steering + (1/self.control_freq)*u_apply[0]
            steering = np.clip(steering, s_min, s_max)

            # Prevent the controller from commanding reverse motion
            target_speed = vx + (1/self.control_freq) * u_apply[1]
            target_speed = np.clip(target_speed, 0.0, v_max)
            tu.e('control_synthesis')
            
            if (isnan(steering) or isnan(target_speed)):
                self.get_logger().warn('nan detected, setting to zero')
                steering = 0.0
                target_speed = 0.0

            tu.s('publish')
            # publish command
            ackermann_msg = AckermannDriveStamped()
            ackermann_msg.header.frame_id = ''
            ackermann_msg.header.stamp = self.get_clock().now().to_msg()
            ackermann_msg.drive.steering_angle = steering
            ackermann_msg.drive.speed = float(target_speed)
            self.drive_pub.publish(ackermann_msg)
            tu.e('publish')
            
            tu.e('total')
            self.last_steering = steering

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to get transform: {e}')

    def clip(self, vec):
        return jnp.clip(vec,
                        jnp.array([s_min, a_min]),
                        jnp.array([s_max, a_max]))

    def compile(self):
        # jax jit compile the first time we run the functions
        self.get_logger().info('compiling jax')
        t0 = time()
        traj = jnp.zeros((100,4))
        x0 = jnp.zeros(7)
        u_vec = jnp.zeros((10,T,m))
        # cost_vec = vmap(lambda u:rollout_cost(traj,x0,u, self.weights, self.clusters))(u_vec).block_until_ready()
        state_traj = rollout(x0, u_vec[0])
        get_interpolated_traj_point(traj, x0)
        rollout_cost(traj, x0, u_vec[0], self.weights, self.clusters)
        eval_cost(traj,state_traj, u_vec[0], self.weights, self.clusters)
        dynamics(x0,u_vec[0,0])
        self.get_logger().info(f'done, took {time()-t0:.2f} seconds')
        return

    def stop(self):
        # publish zero speed command
        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.frame_id = ''
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()
        ackermann_msg.drive.steering_angle = 0.0
        ackermann_msg.drive.speed = 0.0
        self.drive_pub.publish(ackermann_msg)
        self.get_logger().info('send stop command')
        return
    
    def map_callback(self, map: OccupancyGrid):
        if map is not None:
            self.get_logger().info(f'map received: {map.info.width}x{map.info.height}')
        self.h, self.w = map.info.height, map.info.width
        self.res = map.info.resolution
        
        data = np.asarray(map.data, dtype=np.int8).reshape(self.h, self.w)
        
        occ = data > 0
        free = ~occ
        
        occ = occ.astype(np.uint8) * 255
        free = free.astype(np.uint8) * 255
        
        dfree = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        docc  = cv2.distanceTransform(occ, cv2.DIST_L2, 5)
        sdf = (dfree - docc) * self.res  # metres, +ve = free
        self.sdf = np.flipud(sdf)
        self.get_logger().info(f'{self.sdf.shape}')

        scale = 1.0 / (8.0 * self.res)   # Sobel normalisation
        self.gx = cv2.Sobel(self.sdf, cv2.CV_32F, 1, 0, ksize=3, scale=scale)
        self.gy = cv2.Sobel(self.sdf, cv2.CV_32F, 0, 1, ksize=3, scale=scale)
        self.ox = map.info.origin.position.x
        self.oy = map.info.origin.position.y
        self._evt.set()
        self.get_logger().info(f'SDF ready')
    
    def getSDF(self):
        # Lazily (re)create the JAX-device copy only when the map has changed
        if self._sdf_jax is None:
            grid = jax.device_put(jnp.asarray(self.sdf, dtype=jnp.float32))
            origin = jax.device_put(jnp.array([self.ox, self.oy], dtype=jnp.float32))
            inv_res = jnp.float32(1.0 / self.res)
            self._sdf_jax = SDF(grid, origin, inv_res)
        return self._sdf_jax
    
    def ready(self):
        self._evt.wait()
        
    def visualize_mppi(self, state_traj):
        try:
            # simulate control and rollout traj
            path_msg = Path()
            path_msg.header.frame_id = self.map_frame
            path_msg.header.stamp = self.get_clock().now().to_msg()
            for state in state_traj[::10]:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = float(state[0])
                pose.pose.position.y = float(state[1])
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0  # Default orientation
                path_msg.poses.append(pose)
            self.visual_pub.publish(path_msg)
        except Exception as e:
            self.get_logger().warn(f'Visualization error (path): {e}')        
            
def wrap(x):
    return (x + np.pi) % (2*np.pi) - np.pi

@jit
def get_interpolated_traj_point(traj, x):
    min_dist = 1e9
    before = 0
    after = 0
    def find_dist(i):
        return (traj[i,0] - x[0])**2 + (traj[i,1] - x[1])**2 + (traj[i+1,0] - x[0])**2 + (traj[i+1,1] - x[1])**2
    idx_range = jnp.arange(0,traj.shape[0] - 1)
    dist_vec = vmap(find_dist)(idx_range)
    i = jnp.argmin(dist_vec)
    #interpolate
    # let before = A, after = B, x(car state) = P
    # t = AB dot AP / |AB|^2
    AB = (traj[i+1,0] - traj[i,0], traj[i+1,1] - traj[i,1])
    AP = (x[0] - traj[i,0], x[1] - traj[i,1])
    denom = AB[0]**2 + AB[1]**2 + 1e-12
    t = (AB[0]*AP[0] + AB[1]*AP[1]) / (denom + 1e-6)
    # jax.debug.print("t: {t}, denom: {denom}", t=t, denom=denom)
    retval = traj[i]*(1-t) + traj[i+1]*t
    return retval ,i

@jit
def rollout_cost(traj, x0, u, Q, R, sdf):
    state_traj = rollout(x0, u)
    # Correct order: simulated trajectory first, reference path second
    return eval_cost(state_traj, traj, u, Q, R, sdf)

@jit
def eval_cost(state_traj, ref_traj, u_traj, Q, R, sdf):
    alpha = 1/dt
    C = 0.0
    cost = 0.0
    T = u_traj.shape[0]

    for i in range(T):
        # Calculate error w.r.t. the nearest point on the reference path
        xk = state_traj[i]
        ref_state, _ = get_interpolated_traj_point(ref_traj, xk)
        e = ref_state - xk
        
        # Wrap heading error to [-pi, pi]
        e = e.at[5].set(wrap(e[5]))  # Index 5 is heading
        e = e.at[2].set(wrap(e[2]))
        
        cost += e.T @ Q @ e
        
        # Control cost
        u = u_traj[i]
        cost += u.T @ R @ u
        
        # DCBF, pair xk, xk+1
        xk1 = state_traj[i+1]
        hk = h(xk, sdf)
        hk1 = h(xk1, sdf)
        
        viol = alpha * hk - hk1
        pen = C * jnp.maximum(viol, 0.0)
        
        # cost += pen
        
    # Terminal cost with reduced weight
    ref_terminal, _ = get_interpolated_traj_point(ref_traj, state_traj[T])
    e_terminal = ref_terminal - state_traj[T]
    e_terminal = e_terminal.at[5].set(wrap(e_terminal[5]))
    e_terminal = e_terminal.at[2].set(wrap(e_terminal[2]))
    
    cost += e_terminal.T @ Q @ e_terminal
    
    return cost

@jit
def rollout(x0, u):
    def step(_x, _u):
        newx = _x + dynamics(_x,_u)*0.05
        newx = newx.at[2].set(jnp.clip(newx[2],s_min, s_max))
        return newx, newx
    _, state_traj = jax.lax.scan(step, x0, u)
    return jnp.vstack([x0[None,:],state_traj])

@jit
def _fy_piecewise_jax(C, alpha, alpha_sat):
    """Piece-wise linear lateral tyre force."""
    return jnp.where(jnp.abs(alpha) <= alpha_sat,
                     -C * alpha,
                     -C * alpha_sat * jnp.sign(alpha))

@jit
def dynamics(x,u):
    # state x: [x, y, steering, u, v, heading, yaw_rate]
    #           0  1    2        3  4   5          6
    # control u: [ steering_rate, ax]
    
    # longitudinal force
    # approximate longitudinal acceleration commanded by the speed input
    u_dot_cmd = u[1] # ax

    # include centripetal correction
    Fx_total = mass * (u_dot_cmd - x[4] * x[6]) # x[4] is v, x[6] is r

    # split longitudinal force front/rear
    Fx_f = Fx_split * Fx_total
    Fx_r = Fx_total - Fx_f
    u_safe = jnp.maximum(x[3], 0.1)

    # tyre slip angles
    # delta is steering angle, x[2]
    alpha_f = jnp.arctan2(x[4] + lf * x[6], u_safe) - x[2]
    alpha_r = jnp.arctan2(x[4] - lr * x[6], u_safe)

    # lateral tyre forces (piece-wise)
    Fy_f = _fy_piecewise_jax(C_Sf, alpha_f, alpha_sat)
    Fy_r = _fy_piecewise_jax(C_Sr, alpha_r, alpha_sat)

    # longitudinal dynamics
    u_dot = ((Fx_f * jnp.cos(x[2]) - Fy_f * jnp.sin(x[2]) + Fx_r)
            / mass + x[4] * x[6])

    # lateral dynamics
    v_dot = ((Fy_f * jnp.cos(x[2]) + Fx_f * jnp.sin(x[2]) + Fy_r)
            / mass - x[3] * x[6])

    # yaw dynamics
    r_dot = (lf * (Fy_f * jnp.cos(x[2]) + Fx_f * jnp.sin(x[2]))
            - lr * Fy_r) / Iz

    return jnp.array([
        x[3] * jnp.cos(x[5]) - x[4] * jnp.sin(x[5]),  # x_dot
        x[3] * jnp.sin(x[5]) + x[4] * jnp.cos(x[5]),  # y_dot
        u[0],  # steering_dot
        u_dot,
        v_dot,
        x[6],  # heading_dot = yaw_rate
        r_dot
    ])

def h(x: jnp.ndarray, sdf, d_safe=0.3):
    return sdf_query(sdf, x[:2]) - d_safe


def sdf_query(sdf: SDF, xy: jnp.ndarray):
    # xy = [x, y] in world coords (no y-flip needed if you flipped SDF at ingest)
    ix = (xy[0] - sdf.origin[0]) * sdf.inv_res
    iy = (xy[1] - sdf.origin[1]) * sdf.inv_res

    ix0 = jnp.floor(ix).astype(jnp.int32)
    iy0 = jnp.floor(iy).astype(jnp.int32)
    ix1 = jnp.clip(ix0 + 1, 0, sdf.grid.shape[1]-1)
    iy1 = jnp.clip(iy0 + 1, 0, sdf.grid.shape[0]-1)
    ix0 = jnp.clip(ix0, 0, sdf.grid.shape[1]-1)
    iy0 = jnp.clip(iy0, 0, sdf.grid.shape[0]-1)

    wx = ix - ix0.astype(ix.dtype)
    wy = iy - iy0.astype(iy.dtype)

    f00 = sdf.grid[iy0, ix0]; f10 = sdf.grid[iy0, ix1]
    f01 = sdf.grid[iy1, ix0]; f11 = sdf.grid[iy1, ix1]
    v0 = f00 * (1 - wx) + f10 * wx
    v1 = f01 * (1 - wy) + f11 * wy
    dist = v0 * (1 - wy) + v1 * wy
    return dist  # meters (since your SDF is in meters)
      
def main(args=None):
    rclpy.init(args=args)
    node = MppiController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()