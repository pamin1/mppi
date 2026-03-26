"""
profile_comparison.py

Compares the speed and steering profiles commanded by the MPPI controller
(recorded by lap_logger) against the reference profiles produced by the
raceline optimizer.

X-axis: lap fraction [0, 1] — both sources are normalised to this so their
profiles are directly comparable regardless of timing or resolution.

Inputs
------
Optimized raceline : src/mppi/resources/<MAP>_optimized.csv
                     columns: x, y, vx, yaw, kappa
Lap log            : src/mppi/resources/logs/lap_NNN.csv
                     columns: timestamp_s, speed_mps, steering_angle_rad

Outputs (per lap)
-----------------
src/mppi/resources/images/lap_NNN/comparison_speed.png
src/mppi/resources/images/lap_NNN/comparison_steering.png

Usage
-----
# Use the most-recent lap log
python3 profile_comparison.py

# Use a specific lap number
python3 profile_comparison.py --lap 3
"""

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
WHEELBASE = 0.33   # m
MAP_NAME  = "Spielberg_map"

cwd       = os.getcwd()
resources = os.path.join(cwd, "src/mppi/resources")
# Logs are written by the installed node into the install tree
logs_dir  = os.path.join(cwd, "install/mppi/lib/resources/logs")


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> int:
    """Return the lap number to process (1-based)."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--lap", type=int, default=None,
        help="Lap number to analyse (e.g. 3 → lap_003.csv). "
             "Defaults to the highest-numbered lap available.",
    )
    args = parser.parse_args()

    existing = sorted(glob.glob(os.path.join(logs_dir, "lap_*.csv")))
    if not existing:
        print(f"ERROR: no lap CSVs found in {logs_dir}/", file=sys.stderr)
        sys.exit(1)

    if args.lap is not None:
        return args.lap

    # Default: latest by number
    nums = []
    for p in existing:
        m = re.search(r"lap_(\d+)\.csv$", p)
        if m:
            nums.append(int(m.group(1)))

    if not nums:
        names = [os.path.basename(p) for p in existing]
        print(
            f"ERROR: found lap CSVs but none follow the lap_NNN.csv naming scheme: {names}\n"
            "Re-run the simulator with the updated lap_logger to produce an enumerated log.",
            file=sys.stderr,
        )
        sys.exit(1)

    return max(nums)


# ── Load optimized raceline ───────────────────────────────────────────────────
def load_optimizer_profile():
    opt_path = os.path.join(resources, f"{MAP_NAME}_optimized.csv")
    opt = pd.read_csv(opt_path)

    x   = opt["x"].to_numpy()
    y   = opt["y"].to_numpy()
    vx  = opt["vx"].to_numpy()
    yaw = opt["yaw"].to_numpy()
    kap = opt["kappa"].to_numpy()

    ds   = np.hypot(np.diff(x), np.diff(y))
    s    = np.concatenate([[0.0], np.cumsum(ds)])
    frac = s / s[-1]

    # Recover steering sign from the signed curvature of the (x, y) path.
    # np.gradient uses central differences and returns the same length as the
    # input, avoiding the off-by-one and wrap issues of diff(unwrap(yaw)).
    dx_c  = np.gradient(x)
    dy_c  = np.gradient(y)
    ddx_c = np.gradient(dx_c)
    ddy_c = np.gradient(dy_c)
    sign  = np.sign(dx_c * ddy_c - dy_c * ddx_c)
    steer = np.arctan(WHEELBASE * kap) * sign

    return frac, vx, steer, x, y, yaw


# ── Load lap log ──────────────────────────────────────────────────────────────
def load_log_profile(lap_num: int):
    log_path = os.path.join(logs_dir, f"lap_{lap_num:03d}.csv")
    if not os.path.exists(log_path):
        print(f"ERROR: {log_path} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Using log: {log_path}")
    log = pd.read_csv(log_path)

    t     = log["timestamp_s"].to_numpy()
    vx    = log["speed_mps"].to_numpy()
    steer = log["steering_angle_rad"].to_numpy()
    x     = log["x"].to_numpy()
    y     = log["y"].to_numpy()
    yaw   = log["yaw_rad"].to_numpy()

    # Approximate arc length by integrating speed over time
    dt   = np.diff(t)
    ds   = vx[:-1] * dt
    s    = np.concatenate([[0.0], np.cumsum(ds)])
    frac = s / s[-1]

    return frac, vx, steer, x, y, yaw


# ── Plotting ──────────────────────────────────────────────────────────────────
def save_comparison(lap_num: int) -> None:
    frac_opt, vx_opt, steer_opt, x_opt, y_opt, yaw_opt = load_optimizer_profile()
    frac_log, vx_log, steer_log, x_log, y_log, yaw_log = load_log_profile(lap_num)

    lap_tag    = f"lap_{lap_num:03d}"
    images_dir = os.path.join(resources, "images", lap_tag)
    os.makedirs(images_dir, exist_ok=True)

    # Speed
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frac_opt, vx_opt, color="orangered", lw=1.4, label="Optimizer reference")
    ax.plot(frac_log, vx_log, color="steelblue", lw=1.0, alpha=0.85,
            label=f"Controller ({lap_tag})")
    ax.set_title(f"Speed Profile Comparison — {MAP_NAME} / {lap_tag}")
    ax.set_xlabel("Lap fraction")
    ax.set_ylabel("Speed [m/s]")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(images_dir, "comparison_speed.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Steering
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frac_opt, np.degrees(steer_opt), color="orangered", lw=1.4,
            label="Optimizer reference")
    ax.plot(frac_log, np.degrees(steer_log), color="steelblue", lw=1.0, alpha=0.85,
            label=f"Controller ({lap_tag})")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_title(f"Steering Angle Profile Comparison — {MAP_NAME} / {lap_tag}")
    ax.set_xlabel("Lap fraction")
    ax.set_ylabel("Steering angle [deg]")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(images_dir, "comparison_steering.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Path (x, y)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_opt, y_opt, color="orangered", lw=1.4, label="Optimizer reference")
    ax.plot(x_log, y_log, color="steelblue", lw=1.0, alpha=0.85,
            label=f"Controller ({lap_tag})")
    ax.set_aspect("equal")
    ax.set_title(f"Path Comparison — {MAP_NAME} / {lap_tag}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(images_dir, "comparison_path.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Heading
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(frac_opt, np.degrees(np.unwrap(yaw_opt)), color="orangered", lw=1.4,
            label="Optimizer reference")
    ax.plot(frac_log, np.degrees(np.unwrap(yaw_log)), color="steelblue", lw=1.0,
            alpha=0.85, label=f"Controller ({lap_tag})")
    ax.set_title(f"Heading Comparison — {MAP_NAME} / {lap_tag}")
    ax.set_xlabel("Lap fraction")
    ax.set_ylabel("Heading [deg]")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(images_dir, "comparison_heading.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    lap_num = _parse_args()
    save_comparison(lap_num)
