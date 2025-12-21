import cv2
import numpy as np
import pandas as pd
import os

# map params
map_name = "Spielberg_map"
RESOLUTION = 0.05796  # meters per pixel
ORIGIN_X = -84.85359914210505  # meters
ORIGIN_Y = -36.30299725862132  # meters

cwd = os.getcwd()
map_path = f"{cwd}/src/f1tenth_gym_ros/maps/{map_name}.png"
im = cv2.imread(map_path)
if im is None:
    raise FileNotFoundError(f"Could not find map at {map_path}")

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# typical indices for a single closed loop track
c_outer = contours[2].reshape(-1, 2)
c_inner = contours[3].reshape(-1, 2)

baseline_world = []
baseline_pixels = []
img_h = im.shape[0]

# spatial midpoint
for p_out in c_outer:
    dists = np.linalg.norm(c_inner - p_out, axis=1)
    p_in = c_inner[np.argmin(dists)]

    mid_px = (p_out + p_in) / 2.0
    half_width_px = np.linalg.norm(p_out - mid_px)

    baseline_pixels.append(mid_px.astype(int))

    # world coord for ROS
    x_world = (mid_px[0] * RESOLUTION) + ORIGIN_X
    y_world = ((img_h - mid_px[1]) * RESOLUTION) + ORIGIN_Y
    width_m = half_width_px * RESOLUTION

    baseline_world.append([x_world, y_world, width_m, width_m])

# csv exporting
df = pd.DataFrame(baseline_world, columns=["x", "y", "w_left", "w_right"])
df = df.drop_duplicates(subset=["x", "y"]).iloc[::4, :].reset_index(drop=True)
df.to_csv(f"{cwd}/src/mppi/resources/{map_name}.csv", index=False)
print(f"Success: Exported {len(df)} points to {cwd}/src/mppi/resources/{map_name}.csv")

# visualization
viz = im.copy()
cv2.drawContours(viz, [contours[2]], -1, (0, 255, 0), 2)  # outer = green
cv2.drawContours(viz, [contours[3]], -1, (255, 0, 0), 2)  # inner = blue
pts = np.array(baseline_pixels, np.int32).reshape((-1, 1, 2))
cv2.polylines(viz, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
for i, pt in enumerate(baseline_pixels):
    if i % 20 == 0:
        cv2.circle(viz, tuple(pt), 3, (0, 255, 255), -1)  # highlight baseline pixels

cv2.imshow("Green=Outer, Blue=Inner, Red=Baseline Path", viz)
cv2.waitKey(0)
cv2.destroyAllWindows()
