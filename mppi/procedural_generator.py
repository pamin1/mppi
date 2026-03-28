#!/usr/bin/env python3
"""
generate_track.py — Procedurally generate racetrack maps for F1TENTH.

Outputs:
  - {name}.png          occupancy grid image (white=free, black=wall, gray=unknown)
  - {name}.yaml         ROS2 map_server metadata

Usage:
    python3 generate_track.py --name my_track --seed 42
    python3 generate_track.py --name wide_oval --preset oval
    python3 generate_track.py --name tight_circuit --num-points 12 --width 1.8
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
from scipy import interpolate


def generate_control_points(
    num_points: int = 8,
    radius_mean: float = 30.0,
    radius_var: float = 15.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate random control points around a rough circle.
    Returns array of shape (num_points, 2).
    """
    if rng is None:
        rng = np.random.default_rng()

    angles = np.sort(rng.uniform(0, 2 * np.pi, num_points))

    # Ensure minimum angular separation to avoid self-intersections
    min_sep = 2 * np.pi / (num_points * 2)
    for i in range(1, len(angles)):
        if angles[i] - angles[i - 1] < min_sep:
            angles[i] = angles[i - 1] + min_sep
    # Renormalize to [0, 2pi]
    angles = angles * (2 * np.pi / (angles[-1] + min_sep))

    radii = radius_mean + rng.uniform(-radius_var, radius_var, num_points)
    radii = np.clip(radii, radius_mean * 0.4, radius_mean * 1.6)

    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    return np.column_stack([x, y])


def preset_oval(scale: float = 30.0) -> np.ndarray:
    """Oval track control points."""
    t = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    x = scale * 1.5 * np.cos(t)
    y = scale * np.sin(t)
    return np.column_stack([x, y])


def preset_figure8(scale: float = 25.0) -> np.ndarray:
    """Figure-8 shaped track (lemniscate)."""
    t = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    x = scale * np.cos(t)
    y = scale * np.sin(t) * np.cos(t)
    return np.column_stack([x, y])


def smooth_track(
    control_points: np.ndarray, num_samples: int = 1000
) -> np.ndarray:
    """
    Fit a periodic cubic spline through control points and sample densely.
    Returns array of shape (num_samples, 2).
    """
    # Close the loop
    pts = np.vstack([control_points, control_points[0]])

    # Parameterize by cumulative arc length
    ds = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    s = np.concatenate([[0], np.cumsum(ds)])

    # Fit periodic splines
    spline_x = interpolate.CubicSpline(s, pts[:, 0], bc_type="periodic")
    spline_y = interpolate.CubicSpline(s, pts[:, 1], bc_type="periodic")

    s_dense = np.linspace(0, s[-1], num_samples, endpoint=False)
    x_dense = spline_x(s_dense)
    y_dense = spline_y(s_dense)

    return np.column_stack([x_dense, y_dense])


def compute_normals(centerline: np.ndarray) -> np.ndarray:
    """
    Compute unit normals at each centerline point (pointing left).
    Returns array of shape (N, 2).
    """
    # Forward differences with wraparound
    tangents = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    norms = np.sqrt(tangents[:, 0] ** 2 + tangents[:, 1] ** 2)
    norms = np.clip(norms, 1e-6, None)
    tangents = tangents / norms[:, None]

    # Rotate 90 degrees CCW for left-pointing normal
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals


def render_track(
    centerline: np.ndarray,
    track_width: float = 2.5,
    resolution: float = 0.05,
    margin: int = 50,
    wall_thickness: int = 3,
) -> tuple[np.ndarray, float, float]:
    """
    Render the track as an occupancy grid image.

    Returns:
        image:    HxW uint8 array (255=free, 0=wall, 205=unknown)
        origin_x: world x coordinate of pixel (0,0)
        origin_y: world y coordinate of pixel (0,0)
    """
    normals = compute_normals(centerline)
    half_w = track_width / 2.0

    # Inner and outer boundaries
    inner = centerline - normals * half_w
    outer = centerline + normals * half_w

    # Compute image bounds
    all_pts = np.vstack([inner, outer])
    x_min, y_min = all_pts.min(axis=0) - margin * resolution
    x_max, y_max = all_pts.max(axis=0) + margin * resolution

    width_px = int((x_max - x_min) / resolution) + 1
    height_px = int((y_max - y_min) / resolution) + 1

    # Start with unknown (gray)
    img = np.full((height_px, width_px), 205, dtype=np.uint8)

    def world_to_pixel(pts):
        px = ((pts[:, 0] - x_min) / resolution).astype(np.int32)
        py = ((y_max - pts[:, 1]) / resolution).astype(np.int32)  # flip y
        return np.column_stack([px, py])

    # Draw the track surface (white = free space)
    outer_px = world_to_pixel(outer)
    inner_px = world_to_pixel(inner)

    # Fill track area by drawing a filled polygon for each quad segment
    for i in range(len(centerline)):
        j = (i + 1) % len(centerline)
        quad = np.array([
            outer_px[i],
            outer_px[j],
            inner_px[j],
            inner_px[i],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, quad, 255)

    # Draw walls (black = occupied)
    outer_px_draw = outer_px.reshape(-1, 1, 2)
    inner_px_draw = inner_px.reshape(-1, 1, 2)
    cv2.polylines(img, [outer_px_draw], isClosed=True, color=0, thickness=wall_thickness)
    cv2.polylines(img, [inner_px_draw], isClosed=True, color=0, thickness=wall_thickness)

    origin_x = x_min
    origin_y = y_min

    return img, origin_x, origin_y


def write_map_yaml(
    name: str,
    output_dir: str,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> str:
    """Write the ROS2 map_server YAML file."""
    yaml_path = os.path.join(output_dir, f"{name}.yaml")
    png_path = f"{name}.png"

    content = (
        f"image: {png_path}\n"
        f"mode: trinary\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin_x}, {origin_y}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.196\n"
    )

    with open(yaml_path, "w") as f:
        f.write(content)

    return yaml_path


def find_start_pose(centerline: np.ndarray) -> tuple[float, float, float]:
    """
    Return (x, y, theta) for the start pose at the first centerline point,
    heading along the track direction.
    """
    x, y = centerline[0]
    dx = centerline[1, 0] - centerline[0, 0]
    dy = centerline[1, 1] - centerline[0, 1]
    theta = math.atan2(dy, dx)
    return float(x), float(y), float(theta)


def main():
    parser = argparse.ArgumentParser(description="Procedurally generate a racetrack map")
    parser.add_argument("--name", type=str, default="generated_track",
                        help="Track name (used for output filenames)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--num-points", type=int, default=8,
                        help="Number of random control points (more = more complex)")
    parser.add_argument("--radius", type=float, default=30.0,
                        help="Mean radius of the track [meters]")
    parser.add_argument("--radius-var", type=float, default=15.0,
                        help="Radius variation [meters]")
    parser.add_argument("--width", type=float, default=2.5,
                        help="Track width [meters]")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Map resolution [meters/pixel]")
    parser.add_argument("--samples", type=int, default=2000,
                        help="Number of centerline samples (smoothness)")
    parser.add_argument("--wall-thickness", type=int, default=3,
                        help="Wall thickness in pixels")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["oval", "figure8"],
                        help="Use a preset track shape instead of random")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Generate or load control points
    if args.preset == "oval":
        control_pts = preset_oval(args.radius)
        print(f"Using oval preset (scale={args.radius})")
    elif args.preset == "figure8":
        control_pts = preset_figure8(args.radius)
        print(f"Using figure-8 preset (scale={args.radius})")
    else:
        control_pts = generate_control_points(
            num_points=args.num_points,
            radius_mean=args.radius,
            radius_var=args.radius_var,
            rng=rng,
        )
        print(f"Generated {args.num_points} random control points (seed={args.seed})")

    # Smooth into a continuous track
    centerline = smooth_track(control_pts, num_samples=args.samples)
    print(f"Centerline: {len(centerline)} points")

    # Compute arc length
    ds = np.sqrt(np.sum(np.diff(centerline, axis=0) ** 2, axis=1))
    total_length = ds.sum()
    print(f"Track length: {total_length:.1f} m")

    # Render occupancy grid
    img, origin_x, origin_y = render_track(
        centerline,
        track_width=args.width,
        resolution=args.resolution,
        wall_thickness=args.wall_thickness,
    )
    print(f"Image size: {img.shape[1]}x{img.shape[0]} px")

    # Save PNG
    png_path = os.path.join(args.output_dir, f"{args.name}.png")
    cv2.imwrite(png_path, img)
    print(f"Saved: {png_path}")

    # Save YAML
    yaml_path = write_map_yaml(
        args.name, args.output_dir, args.resolution, origin_x, origin_y
    )
    print(f"Saved: {yaml_path}")

    # Print start pose
    sx, sy, stheta = find_start_pose(centerline)
    print(f"\nStart pose: x={sx:.2f}, y={sy:.2f}, theta={stheta:.4f} rad ({math.degrees(stheta):.1f}°)")
    print(f"\ntracks.yaml entry:")
    print(f"  - name: {args.name}")
    print(f"    map_path: /path/to/{args.name}")
    print(f"    start_pose: [{sx:.2f}, {sy:.2f}, {stheta:.4f}]")


if __name__ == "__main__":
    main()