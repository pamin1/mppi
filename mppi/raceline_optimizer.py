from scipy import interpolate, optimize
import pandas as pd
import numpy as np
import os

# basic vehicle params -- needs to be tuned
MU = 1.0489
G = 9.81
ACCEL_MAX = 9.5  # m/s^2
DECEL_MAX = 10.0  # m/s^2
V_MAX = 7.50  # m/s
SAFETY_BUFFER = 0.50  # meters
DOWNSAMPLE_FACTOR = 20


def optimize_raceline(
    input_csv_path,
    output_csv_path,
    v_max=V_MAX,
    accel_max=ACCEL_MAX,
    decel_max=DECEL_MAX,
    mu=MU,
    g=G,
    safety_buffer=SAFETY_BUFFER,
    downsample_factor=DOWNSAMPLE_FACTOR,
):
    df = pd.read_csv(input_csv_path)

    x_sampled = df["x"].values[::downsample_factor]
    y_sampled = df["y"].values[::downsample_factor]
    w_left_sampled = df["w_left"].values[::downsample_factor]
    w_right_sampled = df["w_right"].values[::downsample_factor]

    # ensure loop closure
    x_sampled = np.append(x_sampled, x_sampled[0])
    y_sampled = np.append(y_sampled, y_sampled[0])
    w_left_sampled = np.append(w_left_sampled, w_left_sampled[0])
    w_right_sampled = np.append(w_right_sampled, w_right_sampled[0])

    # arc lengths
    ds = np.sqrt(np.diff(x_sampled) ** 2 + np.diff(y_sampled) ** 2)
    s = np.cumsum(np.insert(ds, 0, 0))

    # centerline normals
    dx_center = np.diff(x_sampled)
    dy_center = np.diff(y_sampled)
    dx_center = np.append(dx_center, x_sampled[1] - x_sampled[0])
    dy_center = np.append(dy_center, y_sampled[1] - y_sampled[0])

    tangent_norm = np.sqrt(dx_center**2 + dy_center**2)
    normals_x = -dy_center / tangent_norm
    normals_y = dx_center / tangent_norm

    def lateral_offsets_to_xy(d_offsets):
        d_full = np.append(d_offsets, d_offsets[0])
        x_raceline = x_sampled + d_full * normals_x
        y_raceline = y_sampled + d_full * normals_y
        return x_raceline, y_raceline

    def objective(d_offsets):
        x_raceline, y_raceline = lateral_offsets_to_xy(d_offsets)
        x_spline = interpolate.CubicSpline(s, x_raceline, bc_type="periodic")
        y_spline = interpolate.CubicSpline(s, y_raceline, bc_type="periodic")

        s_samples = np.linspace(s[0], s[-1], 200)
        dx = x_spline(s_samples, 1)
        dy = y_spline(s_samples, 1)
        ddx = x_spline(s_samples, 2)
        ddy = y_spline(s_samples, 2)

        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2) ** (3 / 2)
        kappas = numerator / denominator

        return np.sum(kappas**2)

    bounds = [
        (-w_right_sampled[i] + safety_buffer, w_left_sampled[i] - safety_buffer)
        for i in range(len(x_sampled) - 1)
    ]
    d0 = np.zeros(len(x_sampled) - 1)

    print("Starting optimization...")
    print(f"Number of optimization variables: {len(d0)}")

    result = optimize.minimize(
        objective,
        d0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-6, "disp": True},
    )

    print(f"\nOptimization completed! Success: {result.success}")
    print(f"Final objective value: {result.fun}")

    d_optimized = result.x
    x_optimized_coarse, y_optimized_coarse = lateral_offsets_to_xy(d_optimized)

    spline_x_opt = interpolate.CubicSpline(s, x_optimized_coarse, bc_type="periodic")
    spline_y_opt = interpolate.CubicSpline(s, y_optimized_coarse, bc_type="periodic")

    OUTPUT_RESOLUTION = len(df["x"])
    s_dense = np.linspace(s[0], s[-1], OUTPUT_RESOLUTION)

    x_optimized_smooth = spline_x_opt(s_dense)
    y_optimized_smooth = spline_y_opt(s_dense)

    dx = spline_x_opt(s_dense, 1)
    dy = spline_y_opt(s_dense, 1)
    ddx = spline_x_opt(s_dense, 2)
    ddy = spline_y_opt(s_dense, 2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2) ** (3 / 2)
    kappa_values = numerator / denominator

    dx_points = np.diff(x_optimized_smooth)
    dy_points = np.diff(y_optimized_smooth)
    distances = np.sqrt(dx_points**2 + dy_points**2)

    velocity_profile = _compute_velocity_profile(
        kappa_values,
        distances,
        x_optimized_smooth,
        y_optimized_smooth,
        v_max,
        accel_max,
        decel_max,
        mu,
        g,
    )

    yaw = np.arctan2(dy, dx)

    lap_time = sum(
        distances[i] / velocity_profile[i]
        for i in range(len(distances))
        if velocity_profile[i] > 0
    )

    print(f"RESULTS")
    print(f"Estimated lap time: {lap_time:.2f} seconds")
    print(f"Average speed: {s_dense[-1] / lap_time:.2f} m/s")
    print(f"Max curvature: {np.max(np.abs(kappa_values)):.4f} 1/m")
    print(f"Max speed: {np.max(velocity_profile):.2f} m/s")
    print(f"Min speed: {np.min(velocity_profile):.2f} m/s")

    final_df = pd.DataFrame(
        {
            "x": x_optimized_smooth,
            "y": y_optimized_smooth,
            "vx": velocity_profile,
            "yaw": yaw,
            "kappa": kappa_values,
        }
    )
    final_df.to_csv(output_csv_path, index=False)
    print(f"\nOutput saved to: {output_csv_path}")
    print(f"Output resolution: {OUTPUT_RESOLUTION} points")

    return final_df

def optimize_baseline(input_csv_path, output_csv_path, v_max=V_MAX, accel_max=ACCEL_MAX,
                      decel_max=DECEL_MAX, mu=MU, g=G, **kwargs):
    df = pd.read_csv(input_csv_path)
    x = df["x"].values
    y = df["y"].values

    # arc lengths (include wrap-around for periodic spline)
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    d_wrap = np.sqrt((x[0] - x[-1])**2 + (y[0] - y[-1])**2)
    s = np.cumsum(np.insert(np.append(ds, d_wrap), 0, 0))

    # periodic splines for smooth curvature
    spline_x = interpolate.CubicSpline(s, np.append(x, x[0]), bc_type="periodic")
    spline_y = interpolate.CubicSpline(s, np.append(y, y[0]), bc_type="periodic")

    s_dense = np.linspace(s[0], s[-1], len(x))
    dx = spline_x(s_dense, 1)
    dy = spline_y(s_dense, 1)
    ddx = spline_x(s_dense, 2)
    ddy = spline_y(s_dense, 2)

    kappa = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    yaw = np.arctan2(dy, dx)
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    velocity = _compute_velocity_profile(kappa, distances, x, y, v_max, accel_max, decel_max, mu, g)

    lap_time = np.sum(distances / velocity[:-1])
    print(f"Estimated lap time: {lap_time:.2f}s")
    print(f"Avg speed: {s[-1] / lap_time:.2f} m/s")
    print(f"Speed range: {velocity.min():.2f} - {velocity.max():.2f} m/s")
    print(f"Max curvature: {kappa.max():.4f} 1/m")

    final_df = pd.DataFrame({"x": x, "y": y, "vx": velocity, "yaw": yaw, "kappa": kappa})
    final_df.to_csv(output_csv_path, index=False)
    print(f"Saved to {output_csv_path} ({len(x)} points)")

    return final_df

def _compute_velocity_profile(
    kappa, distances, x_smooth, y_smooth, v_max, accel_max, decel_max, mu, g
):
    n = len(kappa)
    v = np.zeros(n)

    epsilon = 1e-6
    v_lat_limit = np.sqrt(mu * g / (np.abs(kappa) + epsilon))
    v_lat_limit = np.minimum(v_lat_limit, v_max)

    v[0] = v_lat_limit[0]
    for i in range(n - 1):
        v_accel = np.sqrt(v[i] ** 2 + 2 * accel_max * distances[i])
        v[i + 1] = min(v_accel, v_lat_limit[i + 1], v_max)

    for _ in range(3):
        for i in range(n - 2, -1, -1):
            v_brake = np.sqrt(v[i + 1] ** 2 + 2 * decel_max * distances[i])
            v[i] = min(v[i], v_brake)

        dx_wrap = x_smooth[0] - x_smooth[-1]
        dy_wrap = y_smooth[0] - y_smooth[-1]
        d_wrap = np.sqrt(dx_wrap**2 + dy_wrap**2)
        v_brake_wrap = np.sqrt(v[0] ** 2 + 2 * decel_max * d_wrap)
        v[-1] = min(v[-1], v_brake_wrap)

    return v