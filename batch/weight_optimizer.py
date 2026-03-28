#!/usr/bin/env python3
"""
weight_optimizer.py — Optuna-based cost weight optimizer for the MPPI controller.

Uses TPE to search the 7-dimensional cost weight space, minimizing mean CTE
across all tracks with a hard collision penalty. Each trial runs a full batch
via run_track() imported directly from batch_runner.py.

Usage:
    python3 src/mppi/batch/weight_optimizer.py --n-trials 100
    python3 src/mppi/batch/weight_optimizer.py --n-trials 50 --resume
    optuna-dashboard sqlite:///optuna/study.db
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import optuna
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
BATCH_DIR   = Path(__file__).resolve().parent
MPPI_SRC    = BATCH_DIR.parent
SIM_ROOT    = MPPI_SRC.parent.parent
OPTUNA_DIR  = MPPI_SRC / "optuna"
CONFIG_SRC  = MPPI_SRC / "config" / "cost_weights.yaml"
CONFIG_INST = SIM_ROOT / "install" / "mppi" / "share" / "mppi" / "config" / "cost_weights.yaml"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("weight_optimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Lazy import — batch_runner is in the same directory
sys.path.insert(0, str(BATCH_DIR))
from batch_runner import run_track, load_yaml  # noqa: E402


# ── Parameter space ────────────────────────────────────────────────────────────

def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "steer_dist":      trial.suggest_float("steer_dist",      0.01,   0.2,  log=True),
        "accel_dist":      trial.suggest_float("accel_dist",      0.1,    3.0),
        "q_xy":            trial.suggest_float("q_xy",            0.5,   20.0,  log=True),
        "q_vx":            trial.suggest_float("q_vx",            0.01,   5.0,  log=True),
        "r_accel":         trial.suggest_float("r_accel",         0.0001, 1.0,  log=True),
        "r_steering":      trial.suggest_float("r_steering",      0.001,  2.0,  log=True),
        "r_steering_rate": trial.suggest_float("r_steering_rate", 0.001,  5.0,  log=True),
    }


# ── Config YAML generation ─────────────────────────────────────────────────────

def build_cost_weights_yaml(params: dict) -> dict:
    return {
        "mppi_controller": {
            "ros__parameters": {
                "mppi": {
                    "samples":           8192,
                    "control_frequency": 50,
                    "horizon":           30,
                    "accel_dist":        params["accel_dist"],
                    "steer_dist":        params["steer_dist"],
                    "temperature":       1.0,
                    "alpha":             1.0,
                },
                "cost_weights": {
                    "q_x":            params["q_xy"],
                    "q_y":            params["q_xy"],
                    "q_heading":      0.0,
                    "q_vx":           params["q_vx"],
                    "q_vy":           0.0,
                    "q_yaw_rate":     0.0,
                    "r_accel":        params["r_accel"],
                    "r_steering":     params["r_steering"],
                    "r_steering_rate": params["r_steering_rate"],
                },
            }
        }
    }


def stage_config(config_path: Path) -> None:
    """Copy trial config to all locations the ROS2 stack reads from."""
    for dest in [CONFIG_SRC, CONFIG_INST]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, dest)
        log.debug(f"  Staged config → {dest}")


# ── Objective ──────────────────────────────────────────────────────────────────

def compute_objective(all_metrics: list, lambda_time: float = 0.1) -> float:
    """
    Returns infinity if any track collided.
    Otherwise: mean_cte + lambda_time * mean_lap_time.
    lambda_time=0.1 keeps CTE dominant.
    """
    for m in all_metrics:
        if m.get("collision", False):
            return float("inf")

    cte_values = [m["mean_cte_m"] for m in all_metrics if "mean_cte_m" in m]
    lap_values = [m["best_lap_time_s"] for m in all_metrics if "best_lap_time_s" in m]

    if not cte_values:
        return float("inf")

    mean_cte = float(np.mean(cte_values))
    mean_lap = float(np.mean(lap_values)) if lap_values else 0.0
    return mean_cte + lambda_time * mean_lap


# ── Trial function ─────────────────────────────────────────────────────────────

def make_objective(tracks: list, batch_cfg: dict, lambda_time: float = 0.1):
    """Factory that closes over tracks/batch_cfg and returns the Optuna objective."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)

        # Write per-trial config snapshot
        config = build_cost_weights_yaml(params)
        trial_config_dir = OPTUNA_DIR / "trial_configs"
        trial_config_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_config_dir / f"trial_{trial.number:03d}.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Stage into ROS2 package directories
        stage_config(config_path)
        log.info(f"Trial {trial.number}: params={params}")

        all_metrics: list[dict] = []
        for i, track in enumerate(tracks):
            metrics = run_track(track, batch_cfg)
            if metrics is None:
                metrics = {"collision": True, "track": track["name"]}
            all_metrics.append(metrics)

            # Report intermediate value for pruning
            intermediate = compute_objective(all_metrics, lambda_time=lambda_time)
            trial.report(intermediate, step=i + 1)
            if trial.should_prune():
                log.info(f"Trial {trial.number}: pruned after track {i + 1}/{len(tracks)}")
                raise optuna.TrialPruned()

        score = compute_objective(all_metrics, lambda_time=lambda_time)
        log.info(f"Trial {trial.number}: objective={score:.4f}")
        return score

    return objective


# ── Best config export ─────────────────────────────────────────────────────────

def save_best_config(study: optuna.Study) -> None:
    best = study.best_params
    config = build_cost_weights_yaml(best)
    out = OPTUNA_DIR / "best_config.yaml"
    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    log.info(f"Best config → {out}")

    summary = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": best,
    }
    summary_path = OPTUNA_DIR / "best_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Best summary → {summary_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MPPI cost weight optimizer")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials to run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study (default: create new or load existing)")
    parser.add_argument("--tracks", default=str(BATCH_DIR / "tracks.yaml"),
                        help="Path to tracks.yaml")
    parser.add_argument("--config", default=str(BATCH_DIR / "batch_config.yaml"),
                        help="Path to batch_config.yaml")
    parser.add_argument("--lambda-time", type=float, default=0.1,
                        help="Weight on lap time in objective (default 0.1)")
    args = parser.parse_args()

    tracks_file = Path(args.tracks)
    config_file = Path(args.config)

    for f in [tracks_file, config_file]:
        if not f.exists():
            log.error(f"Config not found: {f}")
            sys.exit(1)

    tracks_data = load_yaml(tracks_file)
    batch_data  = load_yaml(config_file)
    tracks      = tracks_data.get("tracks", [])
    batch_cfg   = batch_data.get("batch", {})

    if not tracks:
        log.error("No tracks defined in tracks.yaml")
        sys.exit(1)

    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)

    db_path  = OPTUNA_DIR / "study.db"
    storage  = f"sqlite:///{db_path}"

    if not args.resume and db_path.exists():
        log.info(f"Found existing study at {db_path}. Loading it (use --resume to continue).")

    study = optuna.create_study(
        study_name="mppi_cost_weights",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
        ),
    )

    log.info(
        f"Starting optimization: {args.n_trials} trials, "
        f"{len(tracks)} tracks each, lambda_time={args.lambda_time}"
    )

    objective = make_objective(tracks, batch_cfg, lambda_time=args.lambda_time)

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        log.info("Interrupted — saving best config so far.")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        save_best_config(study)
        log.info(
            f"Best trial: #{study.best_trial.number}  "
            f"value={study.best_value:.4f}  "
            f"params={study.best_params}"
        )
    else:
        log.warning("No completed trials — nothing to save.")

    log.info(f"Study DB: {db_path}")
    log.info("View results: optuna-dashboard sqlite:///" + str(db_path))


if __name__ == "__main__":
    main()
