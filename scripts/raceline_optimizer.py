from scipy import interpolate
import pandas as pd
import numpy as np
import os

# basic vehicle params -- needs to be tuned
MU = 1.0489
G = 9.81
ACCEL_MAX = 9.5
DECEL_MAX = 10.0
V_MAX = 20.0

# read out the csv, get the x and y coords
cwd = os.getcwd()
file_name = "Spielberg_map"
df = pd.read_csv(f"{cwd}/src/mppi/resources/{file_name}.csv")
x = np.append(df["x"].values, df["x"].values[0])
y = np.append(df["y"].values, df["y"].values[0])

# calculate cumulative distance 's'
ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
s = np.cumsum(np.insert(ds, 0, 0))

# splines
spline_x = interpolate.CubicSpline(s, x, bc_type="periodic")
spline_y = interpolate.CubicSpline(s, y, bc_type="periodic")

s_smooth = np.linspace(0, s[-1], 2000)
x_s = spline_x(s_smooth)
y_s = spline_y(s_smooth)

# curvatures
dx = spline_x.derivative(1)(s_smooth)
dy = spline_y.derivative(1)(s_smooth)
ddx = spline_x.derivative(2)(s_smooth)
ddy = spline_y.derivative(2)(s_smooth)
curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5

# velocity profiling
v_limit = np.sqrt((MU * G) / (curvature + 1e-6))
v_limit = np.clip(v_limit, 0, V_MAX)

# forward/acceleration pass
v_profile = np.zeros_like(s_smooth)
v_profile[0] = v_limit[0]
for i in range(len(s_smooth) - 1):
    dist = s_smooth[i + 1] - s_smooth[i]
    v_max_accel = np.sqrt(v_profile[i] ** 2 + 2 * ACCEL_MAX * dist)
    v_profile[i + 1] = min(v_limit[i + 1], v_max_accel)

# backwards/braking pass
for i in range(len(s_smooth) - 2, -1, -1):
    dist = s_smooth[i + 1] - s_smooth[i]
    v_max_decel = np.sqrt(v_profile[i + 1] ** 2 + 2 * DECEL_MAX * dist)
    v_profile[i] = min(v_profile[i], v_max_decel)

yaw = np.arctan2(dy, dx)

# exporting
final_df = pd.DataFrame(
    {"x": x_s, "y": y_s, "vx": v_profile, "yaw": yaw, "kappa": curvature}
)
final_df.to_csv(f"{cwd}/src/mppi/resources/{file_name}_optimized.csv", index=False)
