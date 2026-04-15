#!/usr/bin/env python3
"""
batch_runner.py — Top-level MPPI batch testing orchestrator.

Iterates over tracks defined in tracks.yaml, runs the full preprocessing
pipeline (centerline extraction → raceline optimization), launches the ROS2
stack headlessly via batch_launch.py, waits for completion on /batch/status,
tears down the stack, and saves per-track metrics + a cross-track summary.

Usage:
    python3 batch_runner.py [--tracks tracks.yaml] [--config batch_config.yaml]

    # From the workspace root with the venv active:
    source ~/.venvs/mppi_batch/bin/activate
    python3 src/mppi/batch/batch_runner.py
"""

import argparse
import json
import logging
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
#
#   batch_runner.py is at:  sim/src/mppi/batch/batch_runner.py
#   BATCH_DIR               sim/src/mppi/batch/
#   MPPI_SRC                sim/src/mppi/
#   SIM_ROOT                sim/

BATCH_DIR        = Path(__file__).resolve().parent
MPPI_SRC         = BATCH_DIR.parent
SIM_ROOT         = MPPI_SRC.parent.parent          # …/sim/
RESULTS_BASE     = MPPI_SRC / "results"
RESOURCES_SRC    = MPPI_SRC / "resources"          # source-tree resources
INSTALLED_SHARE  = SIM_ROOT / "install" / "mppi" / "share" / "mppi"
INSTALLED_RES    = INSTALLED_SHARE / "resources"   # post-build installed resources

# Python executable — prefer the active venv, fall back to the running interpreter
_venv   = os.environ.get("VIRTUAL_ENV")
PYTHON  = Path(_venv) / "bin" / "python3" if _venv else Path(sys.executable)

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch_runner")


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ros_source() -> str:
    """Shell fragment that sources both ROS2 and the workspace install overlay."""
    setup = SIM_ROOT / "install" / "setup.bash"
    return f"source /opt/ros/humble/setup.bash && source {SIM_ROOT}/install/setup.bash"


# ── Subprocess helpers ─────────────────────────────────────────────────────────

def run_subprocess(cmd: list,*,cwd: Path,env: dict,timeout: int,label: str,) -> subprocess.CompletedProcess:
    log.info(f"  [{label}] {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout.strip():
        log.info(f"  [{label}] {result.stdout.strip()}")
    if result.returncode != 0:
        log.error(f"  [{label}] FAILED (exit {result.returncode}):\n{result.stderr.strip()}")
        raise RuntimeError(f"{label} failed (exit {result.returncode})")
    return result


# ── Per-track sim config ───────────────────────────────────────────────────────

def make_sim_config(track: dict) -> Path:
    """
    Write a per-track sim.yaml by cloning the base f1tenth_gym_ros config
    and patching:
      - map_path → this track's map
      - sx / sy / stheta → from tracks.yaml start_pose
      - kb_teleop → False  (headless batch run, no keyboard input)
    """
    base = INSTALLED_SHARE.parent.parent / "f1tenth_gym_ros" / "share" / "f1tenth_gym_ros" / "config" / "sim.yaml"
    if not base.exists():
        base = SIM_ROOT / "src" / "f1tenth_gym_ros" / "config" / "sim.yaml"
    if not base.exists():
        raise FileNotFoundError(f"sim.yaml not found (searched install and src trees)")

    cfg = load_yaml(base)
    params = cfg["bridge"]["ros__parameters"]

    params["map_path"] = track["map_path"]
    params["kb_teleop"] = False

    pose = track.get("start_pose", [0.0, 0.0, 0.0])
    params["sx"]     = float(pose[0])
    params["sy"]     = float(pose[1])
    params["stheta"] = float(pose[2])

    out = BATCH_DIR / f"sim_{track['name']}.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    log.info(f"  sim config → {out}")
    return out


# ── Stage 1: Centerline extraction ────────────────────────────────────────────

def run_centerline_extraction(track: dict, results_dir: Path) -> Path:
    """
    Call baseline_generation.py as a subprocess.

    The script currently has hardcoded map parameters.  We set environment
    variables (BATCH_MAP_NAME, BATCH_MAP_PATH, BATCH_RESOLUTION, BATCH_ORIGIN_X,
    BATCH_ORIGIN_Y, BATCH_OUTPUT_CSV) that a parameterized version of the script
    can read.  The current script ignores them and writes to
    resources/{map_name}.csv; in that case we copy the result to results_dir.
    """
    map_path = track["map_path"]
    map_name = Path(map_path).name
    baseline_csv = results_dir / "baseline.csv"

    # Read pixel→world transform from the ROS2 map YAML
    map_yaml = Path(map_path + ".yaml")
    if not map_yaml.exists():
        raise FileNotFoundError(f"Map YAML not found: {map_yaml}")
    map_meta   = load_yaml(map_yaml)
    resolution = float(map_meta.get("resolution", 0.05796))
    origin     = map_meta.get("origin", [0.0, 0.0, 0.0])

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    env = os.environ.copy()
    env.update({
        "BATCH_MAP_NAME":    map_name,
        "BATCH_MAP_PATH":    map_path,
        "BATCH_RESOLUTION":  str(resolution),
        "BATCH_ORIGIN_X":    str(origin[0]),
        "BATCH_ORIGIN_Y":    str(origin[1]),
        "BATCH_OUTPUT_CSV":  str(baseline_csv),
        "BATCH_FIGURES_DIR": str(figures_dir),
        "DISPLAY":           "",   # suppress any OpenCV GUI windows
        "MPLBACKEND":        "Agg",
    })

    log.info(f"  Extracting centerline for {map_name}...")
    run_subprocess(
        [PYTHON, MPPI_SRC / "mppi" / "baseline_generation.py"],
        cwd=SIM_ROOT,
        env=env,
        timeout=120,
        label="centerline",
    )

    # The script may write to resources/{map_name}.csv rather than baseline_csv;
    # copy it over when that's the case.
    fallback = RESOURCES_SRC / f"{map_name}.csv"
    if not baseline_csv.exists() and fallback.exists():
        shutil.copy(fallback, baseline_csv)
        log.info(f"  Copied {fallback.name} → results/")

    if not baseline_csv.exists():
        raise FileNotFoundError(f"Baseline CSV not produced: {baseline_csv}")

    log.info(f"  Baseline: {baseline_csv} ({baseline_csv.stat().st_size} B)")
    return baseline_csv


# ── Stage 2: Raceline optimization ────────────────────────────────────────────

def run_raceline_optimization(track: dict, baseline_csv: Path, results_dir: Path,
                              batch_cfg: dict | None = None) -> Path:
    """
    Call raceline_optimizer.py as a subprocess.

    Analogous to centerline extraction: we set environment variables for a
    parameterized version of the script and fall back to copying from the
    hardcoded output path resources/{map_name}_optimized.csv when needed.
    """
    map_name      = Path(track["map_path"]).name
    optimized_csv = results_dir / "optimized.csv"

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    cfg = batch_cfg or {}
    env = os.environ.copy()
    env.update({
        "BATCH_FILE_NAME":     map_name,
        "BATCH_INPUT_CSV":     str(baseline_csv),
        "BATCH_OUTPUT_CSV":    str(optimized_csv),
        "BATCH_FIGURES_DIR":   str(figures_dir),
        "MPLBACKEND":          "Agg",
        "BATCH_V_MAX":         str(cfg.get("v_max",          10.0)),
        "BATCH_ACCEL_MAX":     str(cfg.get("accel_max",      20.0)),
        "BATCH_DECEL_MAX":     str(cfg.get("decel_max",      10.0)),
        "BATCH_MU":            str(cfg.get("mu",             1.0489)),
        "BATCH_SAFETY_BUFFER": str(cfg.get("safety_buffer",  0.80)),
        "BATCH_WHEELBASE":     str(cfg.get("wheelbase",      0.33)),
    })

    log.info(f"  Optimizing raceline for {map_name}...")
    run_subprocess(
        [PYTHON, MPPI_SRC / "mppi" / "raceline_optimizer.py"],
        cwd=SIM_ROOT,
        env=env,
        timeout=600,
        label="optimizer",
    )

    fallback_csv     = RESOURCES_SRC / f"{map_name}_optimized.csv"
    # fallback_figures = RESOURCES_SRC / "images"

    if not optimized_csv.exists() and fallback_csv.exists():
        shutil.copy(fallback_csv, optimized_csv)
        log.info(f"  Copied {fallback_csv.name} → results/")

    # if fallback_figures.exists():
    #     for fig in fallback_figures.glob("*.png"):
    #         dest = figures_dir / fig.name
    #         if not dest.exists():
    #             shutil.copy(fig, dest)

    if not optimized_csv.exists():
        raise FileNotFoundError(f"Optimized CSV not produced: {optimized_csv}")

    log.info(f"  Optimized: {optimized_csv} ({optimized_csv.stat().st_size} B)")
    return optimized_csv


def _stage_optimized_csv(track: dict, optimized_csv: Path) -> None:
    """
    Copy optimized.csv into the ROS2 package resource directories so that
    path_planner.py (which reads from the installed share) can find it.

    Copies to both the source tree (for --symlink-install builds) and the
    installed share directory (for regular builds).

    NOTE: path_planner.py currently hard-codes 'Spielberg_map_optimized.csv'.
    Until it is updated to read a 'trajectory_file' ROS2 parameter, only
    Spielberg_map runs will use the correct trajectory automatically.  For
    other tracks, update path_planner.py to honour the trajectory_file param
    passed by batch_launch.py.
    """
    map_name  = Path(track["map_path"]).name
    dest_name = f"{map_name}_optimized.csv"

    for dest_dir in [RESOURCES_SRC, INSTALLED_RES]:
        if dest_dir.exists():
            dest = dest_dir / dest_name
            shutil.copy(optimized_csv, dest)
            log.info(f"  Staged {dest_name} → {dest_dir.relative_to(SIM_ROOT)}")


# ── Stage 3: ROS2 stack launch ────────────────────────────────────────────────

def _next_rrd_path(results_dir: Path) -> Path:
    """Return the next unused run_NNN.rrd path inside results_dir/reruns."""
    import re as _re
    rrd_dir = results_dir / "reruns"
    rrd_dir.mkdir(parents=True, exist_ok=True)
    existing = list(rrd_dir.glob("run_*.rrd"))
    nums = []
    for p in existing:
        m = _re.search(r"run_(\d+)\.rrd$", p.name)
        if m:
            nums.append(int(m.group(1)))
    n = max(nums, default=0) + 1
    return rrd_dir / f"run_{n:03d}.rrd"

def launch_ros_stack(track: dict, sim_config: Path, batch_cfg: dict, results_dir: Path) -> subprocess.Popen:
    map_name   = Path(track["map_path"]).name
    # path_planner.py appends ".csv" itself — pass the stem only
    trajectory = f"{map_name}_optimized"
    rrd_path   = _next_rrd_path(results_dir)
    log_path     = results_dir / "run_log.csv"
    metrics_path = results_dir / "metrics.json"

    log.info(f"trajectory:={trajectory} ")

    ros_cmd = (
        f"{ros_source()} && "
        f"ros2 launch mppi batch_launch.py "
        f"trajectory:={trajectory} "
        f"sim_config:={sim_config} "
        f"target_laps:={int(batch_cfg.get('target_laps', 3))} "
        f"timeout_seconds:={float(batch_cfg.get('timeout_seconds', 120))} "
        f"collision_threshold_m:={float(batch_cfg.get('collision_threshold_m', 0.12))} "
        f"collision_consecutive_readings:={int(batch_cfg.get('collision_consecutive_readings', 3))} "
        f"min_lap_distance:={float(batch_cfg.get('min_lap_distance', 30.0))} "
        f"rerun_output:={rrd_path} "
        f"log_output:={log_path} "
        f"metrics_output_path:={metrics_path}"
    )

    log.info(f"  Launching ROS2 stack for {map_name}...")

    # Kill any stray nodes from a previous interactive or batch session that
    # would conflict by publishing a second trajectory or status stream.
    _stale_nodes = ["/path_planner", "/monitor_node", "/bridge",
                    "/mppi_controller_node", "/map_server"]
    _kill_cmd = (
        f"{ros_source()} && "
        + " ; ".join(f"ros2 node kill {n} 2>/dev/null" for n in _stale_nodes)
        + " ; ros2 daemon stop && ros2 daemon start"
    )
    subprocess.run(_kill_cmd, shell=True, executable="/bin/bash",
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    proc = subprocess.Popen(
        ros_cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,   # new session → single SIGINT kills the whole tree
    )
    log.info(f"  Stack PID: {proc.pid}")
    return proc


# ── Stage 4: Completion monitoring ────────────────────────────────────────────

def wait_for_completion(timeout: float) -> str:
    """
    Stream /batch/status via `ros2 topic echo` and return the first
    non-'running' status value, or 'timeout' when the deadline expires.

    Expected values from monitor_node.py:
        'running'        — normal operation (ignored)
        'laps_complete'  — target lap count reached
        'collision'      — collision detected
        'timeout'        — monitor's internal timeout fired
    """
    log.info(f"  Watching /batch/status (deadline {timeout:.0f} s)...")

    # Give nodes time to initialise before we try to echo the topic
    time.sleep(1.0)

    echo_cmd = (
        f"{ros_source()} && "
        f"ros2 topic echo /batch/status std_msgs/msg/String"
    )
    echo_proc = subprocess.Popen(
        echo_cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        preexec_fn=os.setsid,
    )

    deadline = time.monotonic() + timeout
    final_status = "timeout"

    try:
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            ready, _, _ = select.select([echo_proc.stdout], [], [], min(2.0, remaining))
            if not ready:
                continue

            line = echo_proc.stdout.readline()
            if not line:
                # echo process exited
                break

            line = line.strip()
            log.debug(f"    topic echo: {line!r}")

            # ros2 topic echo YAML format:  data: 'running'
            if line.startswith("data:"):
                val = line.split(":", 1)[1].strip().strip("'\"")
                log.info(f"  /batch/status → {val!r}")
                if val and val != "running":
                    final_status = val
                    break
    finally:
        try:
            os.killpg(os.getpgid(echo_proc.pid), signal.SIGTERM)
            echo_proc.wait(timeout=3.0)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass

    log.info(f"  Completion status: {final_status}")
    return final_status


# ── Stage 5: Teardown ─────────────────────────────────────────────────────────

def teardown_ros_stack(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    log.info("  Tearing down ROS2 stack...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            log.warning("  Graceful shutdown timed out — SIGKILL")
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5.0)
    except (ProcessLookupError, OSError):
        pass
    log.info("  Stack torn down.")


# ── Stage 6: Results collection ───────────────────────────────────────────────

def collect_run_log(results_dir: Path, t_start: float) -> None:
    """
    The monitor node writes run_log.csv directly to results_dir via the
    log_output launch argument.  This function just verifies the file was
    written during the current run (mtime >= t_start).
    """
    dest = results_dir / "run_log.csv"
    if not dest.exists():
        log.warning("  run_log.csv not found — monitor may not have written it.")
        return

    epoch_threshold = time.time() - (time.monotonic() - t_start)
    if dest.stat().st_mtime < epoch_threshold:
        log.warning(
            f"  run_log.csv mtime predates run start — stale file, ignoring."
        )
        dest.unlink()
        return

    log.info(f"  Run log: {dest} ({dest.stat().st_size} B)")


def compute_metrics(results_dir: Path, status: str, elapsed: float) -> dict:
    """
    Load metrics from the monitor node's metrics.json as the primary source.
    If the file doesn't exist (node crashed before writing), return a fallback
    dict with collision=True and the elapsed time measured by batch_runner.
    """
    monitor_json = results_dir / "metrics.json"
    if monitor_json.exists():
        try:
            with open(monitor_json) as f:
                metrics = json.load(f)
            # Always use the authoritative status/collision from the monitor,
            # but override with batch_runner's elapsed (wall-clock truth).
            metrics["elapsed_seconds"] = round(elapsed, 2)
            # If batch_runner saw a different terminal status (e.g. status echo
            # timed out but monitor wrote "laps_complete"), trust the monitor.
            # If monitor wrote a status, keep it; otherwise use batch_runner's.
            if "status" not in metrics:
                metrics["status"] = status
                metrics["collision"] = status == "collision"
            log.info(f"  Metrics loaded from monitor JSON: {monitor_json}")
            return metrics
        except Exception as exc:
            log.warning(f"  Could not parse metrics.json: {exc}")

    # Fallback: monitor didn't write JSON (crashed or timed out hard)
    log.warning(f"  metrics.json not found in {results_dir} — using fallback")
    metrics: dict = {
        "status":          status,
        "collision":       status == "collision",
        "elapsed_seconds": round(elapsed, 2),
        "mean_cte_m":      None,
        "max_cte_m":       None,
        "lap_times":       [],
        "best_lap_time_s": None,
        "total_distance_m": None,
    }
    return metrics


# ── Per-track pipeline ────────────────────────────────────────────────────────

def run_track(track: dict, batch_cfg: dict, skip_preprocessing: bool = False) -> dict:
    subprocess.run(
        "pkill -9 -f 'path_planner.py' ; "
        "pkill -9 -f 'mppi_controller_node' ; "
        "pkill -9 -f 'gym_bridge' ; "
        "pkill -9 -f 'monitor_node.py' ; "
        "sleep 2",
        shell=True, executable="/bin/bash",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    name = track["name"]
    log.info(f"\n{'='*60}")
    log.info(f"  TRACK: {name}")
    log.info(f"{'='*60}")

    results_dir = RESULTS_BASE / name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale per-run outputs so compute_metrics never reads data from a
    # previous trial. Only run_log.csv and metrics.json are run-specific;
    # optimized.csv and baseline.csv are kept (they don't change between trials).
    for stale in ["run_log.csv", "metrics.json"]:
        p = results_dir / stale
        if p.exists():
            p.unlink()

    ros_proc = None
    status   = "error"
    t_start  = time.monotonic()

    try:
        if skip_preprocessing:
            # Raceline already computed — just re-stage it
            optimized_csv = results_dir / "optimized.csv"
            if not optimized_csv.exists():
                raise FileNotFoundError(
                    f"skip_preprocessing=True but optimized CSV not found: {optimized_csv}"
                )
            log.info(f"  Skipping preprocessing, reusing {optimized_csv}")
        else:
            # 1. Centerline extraction
            baseline_csv = run_centerline_extraction(track, results_dir)

            # 2. Raceline optimization
            optimized_csv = run_raceline_optimization(track, baseline_csv, results_dir, batch_cfg)

        # 3. Stage optimized CSV into ROS2 package resource dirs
        _stage_optimized_csv(track, optimized_csv)

        # 4. Write per-track sim.yaml (sets map_path, start pose, kb_teleop=False)
        sim_config = make_sim_config(track)

        # 5. Launch ROS2 stack
        ros_proc = launch_ros_stack(track, sim_config, batch_cfg, results_dir)

        # 6. Wait for completion signal on /batch/status
        timeout = float(batch_cfg.get("timeout_seconds", 120))
        status  = wait_for_completion(timeout)
        log.info(f"  {name}: status={status}  elapsed={time.monotonic() - t_start:.1f} s")

    except Exception as exc:
        log.error(f"  Pipeline error ({name}): {exc}", exc_info=True)
    finally:
        # 7. Tear down regardless of outcome
        teardown_ros_stack(ros_proc)

    elapsed = time.monotonic() - t_start

    # 8. Collect run log and save metrics
    collect_run_log(results_dir, t_start)

    metrics = compute_metrics(results_dir, status, elapsed)
    metrics["track"] = name

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  {name}: status={status}  elapsed={elapsed:.1f} s")

    return metrics


# ── Summary ───────────────────────────────────────────────────────────────────

def generate_summary(all_metrics: list) -> None:
    rows = [
        {
            "track_name":       m.get("track",            ""),
            "status":           m.get("status",           ""),
            "mean_cte_m":       m.get("mean_cte_m",       ""),
            "max_cte_m":        m.get("max_cte_m",        ""),
            "best_lap_time_s":  m.get("best_lap_time_s",  ""),
            "lap_times":        str(m.get("lap_times",    [])),
            "collision":        m.get("collision",        False),
            "elapsed_seconds":  m.get("elapsed_seconds",  ""),
        }
        for m in all_metrics
    ]
    path = RESULTS_BASE / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"Summary → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MPPI batch testing orchestrator")
    parser.add_argument(
        "--tracks",
        default=str(BATCH_DIR / "tracks.yaml"),
        help="Path to tracks.yaml",
    )
    parser.add_argument(
        "--config",
        default=str(BATCH_DIR / "batch_config.yaml"),
        help="Path to batch_config.yaml",
    )
    args = parser.parse_args()

    tracks_file = Path(args.tracks)
    config_file = Path(args.config)

    for f in [tracks_file, config_file]:
        if not f.exists():
            log.error(f"Config not found: {f}")
            sys.exit(1)

    tracks_data = load_yaml(tracks_file)
    batch_data  = load_yaml(config_file)

    tracks    = tracks_data.get("tracks", [])
    batch_cfg = batch_data.get("batch", {})

    if not tracks:
        log.error("No tracks defined in tracks.yaml")
        sys.exit(1)

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    log.info(f"Batch run: {len(tracks)} track(s)")
    log.info(
        f"  target_laps={batch_cfg.get('target_laps')}  "
        f"timeout={batch_cfg.get('timeout_seconds')} s"
    )

    all_metrics = []
    for t in tracks:
        metrics = run_track(t, batch_cfg)
        if metrics:
            all_metrics.append(metrics)

    generate_summary(all_metrics)

    log.info(f"\n{'='*60}")
    log.info("BATCH COMPLETE")
    log.info(f"Results: {RESULTS_BASE}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
