from scipy import interpolate, optimize
import pandas as pd
import numpy as np
import os

# basic vehicle params -- needs to be tuned
MU = 1.0489
G = 9.81
ACCEL_MAX = 9.5  # m/s^2
DECEL_MAX = 10.0  # m/s^2
V_MAX = 20.0  # m/s
SAFETY_BUFFER = 0.70  # meters

# read out the csv, get the x and y coords
cwd = os.getcwd()
file_name = "Spielberg_map"

df = pd.read_csv(f"{cwd}/src/mppi/resources/{file_name}.csv")

# downsampling the positions to speed up the computations
DOWNSAMPLE_FACTOR = 20
x_sampled = df["x"].values[::DOWNSAMPLE_FACTOR]
y_sampled = df["y"].values[::DOWNSAMPLE_FACTOR]
w_left_sampled = df["w_left"].values[::DOWNSAMPLE_FACTOR]
w_right_sampled = df["w_right"].values[::DOWNSAMPLE_FACTOR]

# ensure loop closure
x_sampled = np.append(x_sampled, x_sampled[0])
y_sampled = np.append(y_sampled, y_sampled[0])
w_left_sampled = np.append(w_left_sampled, w_left_sampled[0])
w_right_sampled = np.append(w_right_sampled, w_right_sampled[0])

# arc lengths
ds = np.sqrt(np.diff(x_sampled) ** 2 + np.diff(y_sampled) ** 2)
s = np.cumsum(np.insert(ds, 0, 0))

# centerline normals -- slight issue with the sparse samples?
dx_center = np.diff(x_sampled)
dy_center = np.diff(y_sampled)
dx_center = np.append(dx_center, x_sampled[1] - x_sampled[0])
dy_center = np.append(dy_center, y_sampled[1] - y_sampled[0])

tangent_norm = np.sqrt(dx_center**2 + dy_center**2)
normals_x = -dy_center / tangent_norm
normals_y = dx_center / tangent_norm


def lateral_offsets_to_xy(d_offsets):
    """Convert lateral offsets to x,y coordinates"""
    d_full = np.append(d_offsets, d_offsets[0])

    x_raceline = x_sampled + d_full * normals_x
    y_raceline = y_sampled + d_full * normals_y

    return x_raceline, y_raceline


def objective(d_offsets):
    """Objective function: minimize squared curvature

    Args:
        d_offsets: lateral displacement from centerline at each waypoint (excluding last)
    """
    x_raceline, y_raceline = lateral_offsets_to_xy(d_offsets)

    x_spline = interpolate.CubicSpline(s, x_raceline, bc_type="periodic")
    y_spline = interpolate.CubicSpline(s, y_raceline, bc_type="periodic")

    # compute curvature
    s_samples = np.linspace(s[0], s[-1], 200)
    dx = x_spline(s_samples, 1)
    dy = y_spline(s_samples, 1)
    ddx = x_spline(s_samples, 2)
    ddy = y_spline(s_samples, 2)

    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2) ** (3 / 2)
    kappas = numerator / denominator

    return np.sum(kappas**2)


# Set up box bounds on lateral offsets
bounds = []
for i in range(len(x_sampled) - 1):  # Exclude last point (periodic)
    lower = -w_right_sampled[i] + SAFETY_BUFFER
    upper = w_left_sampled[i] - SAFETY_BUFFER
    bounds.append((lower, upper))

# Initial guess: start at centerline (d = 0 everywhere)
d0 = np.zeros(len(x_sampled) - 1)

print("Starting optimization...")
print(f"Number of optimization variables: {len(d0)}")

# Optimize using L-BFGS-B with bounds
result = optimize.minimize(
    objective,
    d0,
    method="L-BFGS-B",
    bounds=bounds,
    options={"maxiter": 500, "ftol": 1e-6, "disp": True},
)

print(f"\nOptimization completed! Success: {result.success}")
print(f"Final objective value: {result.fun}")

# Extract optimized lateral offsets
d_optimized = result.x

# Convert to x,y coordinates
x_optimized_coarse, y_optimized_coarse = lateral_offsets_to_xy(d_optimized)

# Create optimized splines from coarse waypoints
spline_x_opt = interpolate.CubicSpline(s, x_optimized_coarse, bc_type="periodic")
spline_y_opt = interpolate.CubicSpline(s, y_optimized_coarse, bc_type="periodic")

# Upsample to original resolution
OUTPUT_RESOLUTION = len(df["x"])
s_dense = np.linspace(s[0], s[-1], OUTPUT_RESOLUTION)

# Evaluate splines at dense points
x_optimized_smooth = spline_x_opt(s_dense)
y_optimized_smooth = spline_y_opt(s_dense)

# Compute curvature at dense points
dx = spline_x_opt(s_dense, 1)
dy = spline_y_opt(s_dense, 1)
ddx = spline_x_opt(s_dense, 2)
ddy = spline_y_opt(s_dense, 2)
numerator = np.abs(dx * ddy - dy * ddx)
denominator = (dx**2 + dy**2) ** (3 / 2)
kappa_values = numerator / denominator

# Compute distances between points for velocity solver
dx_points = np.diff(x_optimized_smooth)
dy_points = np.diff(y_optimized_smooth)
distances = np.sqrt(dx_points**2 + dy_points**2)

print("\n" + "=" * 60)
print("VELOCITY PROFILE COMPUTATION")
print("=" * 60)


def compute_velocity_profile(kappa, distances, v_max, accel_max, decel_max, mu, g):
    """
    Forward-backward velocity profile solver

    Args:
        kappa: curvature at each point [1/m]
        distances: distance between consecutive points [m]
        v_max: maximum vehicle speed [m/s]
        accel_max: maximum longitudinal acceleration [m/s^2]
        decel_max: maximum braking deceleration [m/s^2]
        mu: friction coefficient
        g: gravity [m/s^2]

    Returns:
        velocity profile [m/s]
    """
    n = len(kappa)
    v = np.zeros(n)

    # Step 1: Compute lateral acceleration limit from curvature
    # v_lat_max = sqrt(mu * g / |kappa|)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    v_lat_limit = np.sqrt(mu * g / (np.abs(kappa) + epsilon))
    v_lat_limit = np.minimum(v_lat_limit, v_max)

    # Step 2: Forward pass (acceleration limit)
    # v[i+1]^2 = v[i]^2 + 2 * a_max * ds[i]
    v[0] = v_lat_limit[0]  # Start at first point's lateral limit

    for i in range(n - 1):
        # Maximum speed from acceleration
        v_accel = np.sqrt(v[i] ** 2 + 2 * accel_max * distances[i])

        # Limited by lateral acceleration at next point
        v[i + 1] = min(v_accel, v_lat_limit[i + 1], v_max)

    # Step 3: Backward pass (braking limit)
    # v[i]^2 = v[i+1]^2 + 2 * a_brake * ds[i]
    # Process from end to start (with wraparound for closed loop)

    # Multiple backward passes to handle closed loop properly
    for lap in range(3):  # Usually converges in 2-3 laps
        for i in range(n - 2, -1, -1):
            # Maximum speed from braking
            v_brake = np.sqrt(v[i + 1] ** 2 + 2 * decel_max * distances[i])

            # Take minimum of current speed and braking-limited speed
            v[i] = min(v[i], v_brake)

        # Handle wraparound (last point -> first point)
        # Distance from last point back to first
        dx_wrap = x_optimized_smooth[0] - x_optimized_smooth[-1]
        dy_wrap = y_optimized_smooth[0] - y_optimized_smooth[-1]
        d_wrap = np.sqrt(dx_wrap**2 + dy_wrap**2)

        v_brake_wrap = np.sqrt(v[0] ** 2 + 2 * decel_max * d_wrap)
        v[-1] = min(v[-1], v_brake_wrap)
    return v


# Compute velocity profile
velocity_profile = compute_velocity_profile(
    kappa_values, distances, V_MAX, ACCEL_MAX, DECEL_MAX, MU, G
)

# Compute yaw angle from tangent vectors
yaw = np.arctan2(dy, dx)
# yaw = np.unwrap(yaw)

# Compute lap time
lap_time = 0.0
for i in range(len(distances)):
    if velocity_profile[i] > 0:
        lap_time += distances[i] / velocity_profile[i]

print(f"RESULTS")
print(f"Estimated lap time: {lap_time:.2f} seconds")
print(f"Average speed: {s_dense[-1] / lap_time:.2f} m/s")
print(f"Max curvature: {np.max(np.abs(kappa_values)):.4f} 1/m")
print(f"Max speed: {np.max(velocity_profile):.2f} m/s")
print(f"Min speed: {np.min(velocity_profile):.2f} m/s")

# exporting
final_df = pd.DataFrame(
    {
        "x": x_optimized_smooth,
        "y": y_optimized_smooth,
        "vx": velocity_profile,
        "yaw": yaw,
        "kappa": kappa_values,
    }
)
final_df.to_csv(f"{cwd}/src/mppi/resources/{file_name}_optimized_original.csv", index=False)

print(f"\nOutput saved to: {file_name}_optimized.csv")
print(f"Output resolution: {OUTPUT_RESOLUTION} points")
