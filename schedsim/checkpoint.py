"""
Checkpoint management for trained predictors and Q-learning agents.

Every save writes both a versioned (timestamped) file and overwrites the
canonical ``*_latest.*`` path so callers always have a stable reference
to the most recent model, while the full version history is preserved for
the leaderboard and for rollback.

On-disk layout (checkpoints/ inside the repo root)::

    checkpoints/
        predictor_latest.pkl      -- most recently saved predictor
        predictor_<ts>.pkl        -- versioned copies (never overwritten)
        qtable_latest.npz         -- most recently saved Q-table
        qtable_<ts>.npz           -- versioned copies
        training_log.jsonl        -- one JSON line per save/evaluation event

Dependency: joblib (already a transitive sklearn dependency; listed explicitly
in requirements.txt so the import is stable across environments).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

_LOG_FILE = "training_log.jsonl"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Predictor checkpoints  (joblib-serialised FittedPredictor)
# ---------------------------------------------------------------------------

def save_predictor(
    predictor,
    metadata: Dict[str, Any],
    ckpt_dir: str,
) -> str:
    """Save a FittedPredictor and accompanying metadata.

    Creates ``<ckpt_dir>/predictor_<timestamp>.pkl`` (versioned, never
    overwritten) and updates ``predictor_latest.pkl`` atomically.

    Returns
    -------
    Path to the versioned checkpoint file.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ts = _ts()
    versioned = os.path.join(ckpt_dir, f"predictor_{ts}.pkl")
    latest    = os.path.join(ckpt_dir, "predictor_latest.pkl")
    payload   = {"predictor": predictor, "metadata": metadata, "saved_at": ts}
    joblib.dump(payload, versioned)
    shutil.copy2(versioned, latest)
    _append_log(ckpt_dir, "predictor_saved", {**metadata, "path": versioned})
    return versioned


def load_predictor(
    ckpt_dir: str,
    path: Optional[str] = None,
) -> Tuple[Any, Dict]:
    """Load the latest (or a specific) predictor checkpoint.

    Returns ``(predictor, metadata)`` or ``(None, {})`` if nothing exists.
    """
    p = path or os.path.join(ckpt_dir, "predictor_latest.pkl")
    if not os.path.exists(p):
        return None, {}
    payload = joblib.load(p)
    return payload["predictor"], payload.get("metadata", {})


# ---------------------------------------------------------------------------
# Q-table checkpoints  (NumPy NPZ)
# ---------------------------------------------------------------------------

def save_qtable(
    q_table: np.ndarray,
    metadata: Dict[str, Any],
    ckpt_dir: str,
) -> str:
    """Save a Q-table and accompanying metadata.

    Scalar metadata values are stored as 0-d NumPy arrays inside the NPZ so
    the file is self-contained (no separate JSON sidecar needed).

    Returns
    -------
    Path to the versioned checkpoint file.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ts = _ts()
    versioned = os.path.join(ckpt_dir, f"qtable_{ts}.npz")
    latest    = os.path.join(ckpt_dir, "qtable_latest.npz")
    safe_meta = {
        k: np.array(v)
        for k, v in metadata.items()
        if isinstance(v, (int, float, str, bool))
    }
    np.savez(versioned, q_table=q_table, **safe_meta)
    shutil.copy2(versioned, latest)
    _append_log(ckpt_dir, "qtable_saved", {**metadata, "path": versioned})
    return versioned


def load_qtable(
    ckpt_dir: str,
    path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Load the latest (or a specific) Q-table checkpoint.

    Returns ``(q_table, metadata)`` or ``(None, {})`` if nothing exists.
    """
    p = path or os.path.join(ckpt_dir, "qtable_latest.npz")
    if not os.path.exists(p):
        return None, {}
    d = np.load(p, allow_pickle=False)
    q_table = d["q_table"]
    meta = {k: d[k].item() for k in d.files if k != "q_table"}
    return q_table, meta


# ---------------------------------------------------------------------------
# Training log  (append-only JSONL)
# ---------------------------------------------------------------------------

def _append_log(ckpt_dir: str, event: str, data: Dict):
    """Append one structured event to the training log."""
    path  = os.path.join(ckpt_dir, _LOG_FILE)
    entry = {"event": event, "ts": _ts(), **data}
    # Serialise only JSON-safe scalars; skip objects.
    safe  = {k: v for k, v in entry.items()
              if isinstance(v, (str, int, float, bool, type(None)))}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(safe) + "\n")


def read_training_log(ckpt_dir: str) -> list:
    """Return all training-log entries as a list of dicts (oldest first)."""
    path = os.path.join(ckpt_dir, _LOG_FILE)
    if not os.path.exists(path):
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def leaderboard(ckpt_dir: str, metric: str = "val_rmse") -> list:
    """Return predictor-save events sorted by `metric` (ascending = better).

    Use ``metric='val_rmse'`` to rank by validation RMSE.
    """
    log   = read_training_log(ckpt_dir)
    saves = [e for e in log
              if e.get("event") == "predictor_saved" and metric in e]
    return sorted(saves, key=lambda e: e[metric])
