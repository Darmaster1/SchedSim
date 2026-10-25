"""
Accumulating training data pool for the ML burst-length predictors.

The pool is an append-only CSV that stores (feature_vector, true_burst) rows
collected from randomly generated workloads across multiple training runs.
Re-training on the full pool after each run implements a data-flywheel: the
effective training set grows with every call to `train.py run`, progressively
improving prediction RMSE on the frozen evaluation set.

On-disk layout (inside the repo root):
    data/
        training_pool.csv    -- accumulated feature rows
        pool_meta.json       -- provenance: n_rows, n_runs, spec distribution

The pool is append-only; rows are never modified or deleted. This gives full
auditability: the CSV can be opened in any spreadsheet tool to inspect the
training data used by any model checkpoint.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

from .workload import Process

_POOL_CSV  = "training_pool.csv"
_META_JSON = "pool_meta.json"

_FIELDNAMES = [
    "vsize", "text", "endcode", "vmlib", "true_burst",
    "spec_name", "seed", "n_processes", "added_at",
]


@dataclass
class PoolMeta:
    n_rows: int = 0
    n_runs: int = 0
    created_at: str = ""
    last_updated_at: str = ""
    spec_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def n_unique_specs(self) -> int:
        return len(self.spec_counts)


class TrainingPool:
    """Persistent, append-only store of (features, true_burst) rows.

    Thread-safety: not guaranteed. Use a single writer process.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._csv_path  = os.path.join(data_dir, _POOL_CSV)
        self._meta_path = os.path.join(data_dir, _META_JSON)
        os.makedirs(data_dir, exist_ok=True)
        self._meta = self._load_meta()

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def append(self, procs: List[Process], spec_name: str, seed: int) -> int:
        """Append every process from one workload to the pool.

        Parameters
        ----------
        procs:     List[Process] from `generate_workload`.
        spec_name: The WorkloadSpec.name used to generate this workload.
        seed:      The integer seed passed to `generate_workload`.

        Returns
        -------
        Number of rows appended (== len(procs)).
        """
        ts = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "vsize":       round(p.vsize,   6),
                "text":        round(p.text,    6),
                "endcode":     round(p.endcode, 6),
                "vmlib":       round(p.vmlib,   6),
                "true_burst":  p.true_burst,
                "spec_name":   spec_name,
                "seed":        seed,
                "n_processes": len(procs),
                "added_at":    ts,
            }
            for p in procs
        ]
        write_header = not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        self._meta.n_rows += len(rows)
        self._meta.n_runs += 1
        self._meta.spec_counts[spec_name] = (
            self._meta.spec_counts.get(spec_name, 0) + len(rows)
        )
        if not self._meta.created_at:
            self._meta.created_at = ts
        self._meta.last_updated_at = ts
        self._save_meta()
        return len(rows)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def load_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X, y)`` arrays from the entire pool.

        X  shape (n, 4): [vsize, text, endcode, vmlib]
        y  shape (n,):   true_burst values
        """
        if not os.path.exists(self._csv_path):
            raise FileNotFoundError(
                f"Pool CSV not found at {self._csv_path}.\n"
                "Run `python train.py run` to generate training data first."
            )
        xs, ys = [], []
        with open(self._csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                xs.append([
                    float(row["vsize"]),
                    float(row["text"]),
                    float(row["endcode"]),
                    float(row["vmlib"]),
                ])
                ys.append(float(row["true_burst"]))
        if not xs:
            raise ValueError("Training pool CSV exists but is empty.")
        return np.array(xs), np.array(ys)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of (feature, burst) rows in the pool."""
        return self._meta.n_rows

    @property
    def meta(self) -> PoolMeta:
        return self._meta

    def __repr__(self) -> str:
        m = self._meta
        return (
            f"TrainingPool({m.n_rows:,} rows | {m.n_runs} runs | "
            f"{m.n_unique_specs} unique specs | "
            f"last updated {m.last_updated_at or 'never'})"
        )

    # ------------------------------------------------------------------
    # Internal persistence helpers
    # ------------------------------------------------------------------

    def _load_meta(self) -> PoolMeta:
        if os.path.exists(self._meta_path):
            with open(self._meta_path, encoding="utf-8") as f:
                d = json.load(f)
            return PoolMeta(**d)
        return PoolMeta()

    def _save_meta(self):
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self._meta), f, indent=2)
