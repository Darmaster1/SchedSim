"""
Model checkpointing and a validation-RMSE leaderboard for the continuous
training flywheel (`train.py`): refit every candidate burst-prediction model
on the accumulated pool, record each round's validation RMSE to a leaderboard
CSV, and atomically save the best model as the "latest" checkpoint that
`run_experiment.py` or a resumed training round can load.
"""
from __future__ import annotations

import csv
import os
import pickle
import time
from typing import Dict, List, Tuple

from .predictors import FittedPredictor, evaluate_predictor, fit_predictor, MODEL_FACTORIES
from .workload import Process

LEADERBOARD_FIELDS = ["round", "timestamp", "model", "pool_size", "val_rmse", "val_mae", "val_r2",
                       "is_best_this_round"]


def refit_all_models(pool_procs: List[Process], val_procs: List[Process],
                      checkpoint_dir: str = "checkpoints", round_idx: int = 0
                      ) -> Tuple[str, FittedPredictor, Dict[str, Dict[str, float]]]:
    """Refit every candidate model on `pool_procs`, score each on
    `val_procs`, write a timestamped checkpoint for each model plus a
    leaderboard row, and atomically promote the best model to
    `<checkpoint_dir>/predictor_latest.pkl`. Returns (best_name, best_fp, all_metrics)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    leaderboard_path = os.path.join(checkpoint_dir, "leaderboard.csv")
    leaderboard_exists = os.path.isfile(leaderboard_path)

    all_metrics: Dict[str, Dict[str, float]] = {}
    fitted: Dict[str, FittedPredictor] = {}
    for name in MODEL_FACTORIES:
        fp = fit_predictor(name, pool_procs, use_endcode=True)
        metrics = evaluate_predictor(fp, val_procs)
        fitted[name] = fp
        all_metrics[name] = metrics

        ckpt_path = os.path.join(checkpoint_dir, f"{name}_round{round_idx}.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump(fp, f)

    best_name = min(all_metrics, key=lambda n: all_metrics[n]["rmse"])
    best_fp = fitted[best_name]

    with open(leaderboard_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEADERBOARD_FIELDS)
        if not leaderboard_exists:
            writer.writeheader()
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        for name, m in all_metrics.items():
            writer.writerow({"round": round_idx, "timestamp": ts, "model": name,
                              "pool_size": len(pool_procs), "val_rmse": m["rmse"],
                              "val_mae": m["mae"], "val_r2": m["r2"],
                              "is_best_this_round": name == best_name})

    # Atomic promotion: write to a temp file then rename, so a reader never
    # sees a partially-written "latest" checkpoint.
    latest_path = os.path.join(checkpoint_dir, "predictor_latest.pkl")
    tmp_path = latest_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(best_fp, f)
    os.replace(tmp_path, latest_path)

    return best_name, best_fp, all_metrics


def load_latest_predictor(checkpoint_dir: str = "checkpoints") -> FittedPredictor:
    path = os.path.join(checkpoint_dir, "predictor_latest.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def read_leaderboard(checkpoint_dir: str = "checkpoints") -> List[dict]:
    path = os.path.join(checkpoint_dir, "leaderboard.csv")
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
