"""
Append-only training pool for the continuous-training flywheel (`train.py`).

Workloads are sampled from the combinatorial specification space (the same
categories defined in workload.py) using seeds strictly disjoint from the
train/val/test evaluation splits, and the resulting per-process feature/burst
rows are appended to a CSV file that only ever grows -- so repeated training
rounds accumulate more data for the burst predictors to refit on, rather
than re-deriving a fixed sample each time.
"""
from __future__ import annotations

import csv
import os
import random
from typing import List

from .workload import Process, WorkloadSpec, generate_workload, in_distribution_categories

POOL_PATH_DEFAULT = "training_pool.csv"
# Strictly disjoint from SEED_OFFSETS["train"/"val"/"test"] in workload.py
# (which top out at 2_000_000 + 50*10_000 = 2_500_000), so pool data can
# never accidentally coincide with an evaluation workload.
POOL_SEED_RANGE = (3_000_000, 4_000_000)

_FIELDS = ["pid", "arrival", "priority", "true_burst", "vsize", "text", "endcode", "vmlib"]


def _process_to_row(p: Process) -> dict:
    return {"pid": p.pid, "arrival": p.arrival, "priority": p.priority,
            "true_burst": p.true_burst, "vsize": p.vsize, "text": p.text,
            "endcode": p.endcode, "vmlib": p.vmlib}


def append_to_pool(n_workloads: int, pool_path: str = POOL_PATH_DEFAULT,
                    seed: int = None, n_processes: int = 24) -> int:
    """Generate `n_workloads` random (category, seed) workloads from the
    pool's disjoint seed range, append every process's row to `pool_path`
    (creating it with a header if it doesn't exist yet), and return the
    number of process-rows appended."""
    rng = random.Random(seed)
    specs = in_distribution_categories(n_processes)
    file_exists = os.path.isfile(pool_path)
    rows_written = 0

    with open(pool_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            writer.writeheader()
        for _ in range(n_workloads):
            spec = rng.choice(specs)
            workload_seed = rng.randrange(*POOL_SEED_RANGE)
            procs = generate_workload(spec, workload_seed)
            for p in procs:
                writer.writerow(_process_to_row(p))
                rows_written += 1

    return rows_written


def load_pool(pool_path: str = POOL_PATH_DEFAULT) -> List[Process]:
    """Load every process-row ever appended to the pool as a flat list of
    Process objects (pid is re-numbered 0..n-1 since the pool mixes many
    workloads and pid is only meaningful within a single workload)."""
    if not os.path.isfile(pool_path):
        return []
    procs = []
    with open(pool_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            procs.append(Process(
                pid=i, arrival=float(row["arrival"]), priority=int(row["priority"]),
                true_burst=int(float(row["true_burst"])), vsize=float(row["vsize"]),
                text=float(row["text"]), endcode=float(row["endcode"]), vmlib=float(row["vmlib"]),
            ))
    return procs


def pool_size(pool_path: str = POOL_PATH_DEFAULT) -> int:
    return len(load_pool(pool_path))
