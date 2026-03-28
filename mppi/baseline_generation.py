import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────
# When called by batch_runner.py the following env vars are set:
#   BATCH_MAP_NAME    — map stem, e.g. "Spielberg_map"
#   BATCH_MAP_PATH    — absolute path without extension, e.g. ".../maps/Spielberg_map"
#   BATCH_RESOLUTION  — meters per pixel (float string)
#   BATCH_ORIGIN_X    — world origin X in metres (float string)
#   BATCH_ORIGIN_Y    — world origin Y in metres (float string)
#   BATCH_OUTPUT_CSV  — absolute path where baseline.csv should be written
# When run standalone the hardcoded defaults below are used instead.

map_name   = os.environ.get("BATCH_MAP_NAME", "Spielberg_map")
RESOLUTION = float(os.environ.get("BATCH_RESOLUTION", "0.05796"))
ORIGIN_X   = float(os.environ.get("BATCH_ORIGIN_X",   "-84.85359914210505"))
ORIGIN_Y   = float(os.environ.get("BATCH_ORIGIN_Y",   "-36.30299725862132"))

cwd = os.getcwd()

# Derive the PNG path: prefer BATCH_MAP_PATH (no extension), fall back to cwd layout
_batch_map_path = os.environ.get("BATCH_MAP_PATH", "")
if _batch_map_path:
    map_path = _batch_map_path + ".png"
else:
    map_path = f"{cwd}/src/f1tenth_gym_ros/maps/{map_name}.png"

# Output CSV path
_batch_output_csv = os.environ.get("BATCH_OUTPUT_CSV", "")
output_csv = _batch_output_csv if _batch_output_csv else f"{cwd}/src/mppi/resources/{map_name}.csv"

# ── Image loading ─────────────────────────────────────────────────────────────

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

# ── Spatial midpoint computation ──────────────────────────────────────────────

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

# ── CSV export ────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df = pd.DataFrame(baseline_world, columns=["x", "y", "w_left", "w_right"])
df = df.drop_duplicates(subset=["x", "y"]).iloc[::4, :].reset_index(drop=True)
df.to_csv(output_csv, index=False)
print(f"Success: Exported {len(df)} points to {output_csv}")

# ── Visualization ─────────────────────────────────────────────────────────────
# Batch mode: save a single matplotlib figure when BATCH_FIGURES_DIR is set.
# Interactive mode: show an OpenCV window (only when DISPLAY is available and
#                   BATCH_OUTPUT_CSV was not given).

_figures_dir = os.environ.get("BATCH_FIGURES_DIR", "")
_headless    = not os.environ.get("DISPLAY") or bool(_batch_output_csv)

if _figures_dir:
    def _px_to_world(pts_px):
        xs = pts_px[:, 0] * RESOLUTION + ORIGIN_X
        ys = (img_h - pts_px[:, 1]) * RESOLUTION + ORIGIN_Y
        return xs, ys

    x_outer, y_outer = _px_to_world(c_outer)
    x_inner, y_inner = _px_to_world(c_inner)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_outer, y_outer, color="steelblue", lw=1.0, label="Left wall")
    ax.plot(x_inner, y_inner, color="firebrick", lw=1.0, label="Right wall")
    ax.plot(df["x"], df["y"], color="forestgreen", lw=1.2, label="Centerline")
    ax.set_aspect("equal")
    ax.set_title(f"Baseline — {map_name}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(_figures_dir, "baseline_contours.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Baseline figure saved to {out_path}")

elif not _headless:
    viz = im.copy()
    cv2.drawContours(viz, [contours[2]], -1, (0, 255, 0), 2)  # outer = green
    cv2.drawContours(viz, [contours[3]], -1, (255, 0, 0), 2)  # inner = blue
    pts = np.array(baseline_pixels, np.int32).reshape((-1, 1, 2))
    cv2.polylines(viz, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    for i, pt in enumerate(baseline_pixels):
        if i % 20 == 0:
            cv2.circle(viz, tuple(pt), 3, (0, 255, 255), -1)

    cv2.imshow("Green=Outer, Blue=Inner, Red=Baseline Path", viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
