"""
Synthetic workload generation for schedsim.

Every workload is a deterministic function of a WorkloadSpec and an integer
seed: the same (spec, seed) pair always produces the same set of processes.
Training / validation / test splits use disjoint seed ranges by construction
(see experiment.py), which is what keeps the three splits from leaking into
one another.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np

BurstMode = Literal["short", "long", "mixed", "heavy_tailed"]
ArrivalMode = Literal["simultaneous", "staggered", "bursty"]
PriorityMode = Literal["uniform", "skewed", "adversarial"]
FeatureStrength = Literal["strong", "moderate", "weak", "noisy", "misleading"]

_STRENGTH_RHO = {
    "strong": 0.95,
    "moderate": 0.70,
    "weak": 0.30,
    "noisy": 0.70,       # same rho as 'moderate', but 15% outlier rows injected
    "misleading": -0.70,
}


@dataclass
class Process:
    pid: int
    arrival: float
    priority: int
    true_burst: int
    vsize: float
    text: float
    endcode: float
    vmlib: float

    def features(self) -> np.ndarray:
        return np.array([self.vsize, self.text, self.endcode, self.vmlib])


@dataclass
class WorkloadSpec:
    name: str
    n_processes: int = 30
    burst_mode: BurstMode = "mixed"
    arrival_mode: ArrivalMode = "staggered"
    priority_mode: PriorityMode = "uniform"
    feature_strength: FeatureStrength = "moderate"
    offered_load: float = 0.7  # used by 'staggered' arrivals


def _sample_burst(rng: np.random.Generator, mode: BurstMode, n: int) -> np.ndarray:
    if mode == "short":
        b = rng.geometric(p=0.12, size=n)
        return np.clip(b, 1, 20)
    if mode == "long":
        b = rng.lognormal(mean=np.log(45), sigma=0.45, size=n)
        return np.clip(b, 20, 150).round()
    if mode == "mixed":
        is_long = rng.random(n) < 0.30
        short = _sample_burst(rng, "short", n)
        long_ = _sample_burst(rng, "long", n)
        return np.where(is_long, long_, short)
    if mode == "heavy_tailed":
        b = (rng.pareto(a=1.5, size=n) + 1) * 15
        return np.clip(b, 1, 400).round()
    raise ValueError(mode)


def _sample_arrivals(rng: np.random.Generator, mode: ArrivalMode, n: int,
                      offered_load: float, mean_burst: float) -> np.ndarray:
    if mode == "simultaneous":
        return np.zeros(n)
    if mode == "staggered":
        mean_interarrival = mean_burst / max(offered_load, 1e-3)
        gaps = rng.exponential(scale=mean_interarrival, size=n)
        return np.cumsum(gaps)
    if mode == "bursty":
        n_clusters = max(1, n // 6)
        cluster_id = rng.integers(0, n_clusters, size=n)
        cluster_start = np.sort(rng.exponential(scale=mean_burst * 4, size=n_clusters).cumsum())
        jitter = rng.exponential(scale=mean_burst * 0.3, size=n)
        return cluster_start[cluster_id] + jitter
    raise ValueError(mode)


def _sample_priorities(rng: np.random.Generator, mode: PriorityMode,
                        burst: np.ndarray) -> np.ndarray:
    n = len(burst)
    if mode == "uniform":
        return rng.integers(1, 11, size=n)
    if mode == "skewed":
        # skew toward low priority (large numbers = low priority)
        p = rng.triangular(left=1, mode=9, right=10, size=n)
        return np.clip(p.round(), 1, 10).astype(int)
    if mode == "adversarial":
        # long bursts tend to get HIGH priority (small numbers)
        rank = burst.argsort().argsort()  # 0 = shortest burst
        base = 1 + (9 * (1 - rank / max(n - 1, 1)))
        noise = rng.normal(0, 1.2, size=n)
        return np.clip((base + noise).round(), 1, 10).astype(int)
    raise ValueError(mode)


def _sample_features(rng: np.random.Generator, burst: np.ndarray,
                      strength: FeatureStrength) -> np.ndarray:
    """Generate 4 synthetic features whose correlation with log(burst) is
    controlled by `strength`. Returns an (n, 4) array: vsize, text, endcode, vmlib.
    """
    n = len(burst)
    log_b = np.log(np.maximum(burst, 1))
    log_b_z = (log_b - log_b.mean()) / (log_b.std() + 1e-9)

    rho = _STRENGTH_RHO[strength]

    def corr_feature(rho_: float, scale: float) -> np.ndarray:
        noise = rng.normal(0, 1, size=n)
        z = rho_ * log_b_z + np.sqrt(max(1 - rho_ ** 2, 0.0)) * noise
        return (z * scale + 50 * scale)  # shift to a positive-ish range

    text = corr_feature(rho, 10.0)
    if strength == "noisy":
        n_outliers = int(0.15 * n)
        idx = rng.choice(n, size=n_outliers, replace=False)
        text[idx] += rng.normal(0, 80, size=n_outliers)

    vsize = corr_feature(rho, 10.0) * 0.8
    vmlib = corr_feature(rho, 10.0) * 0.6
    # endcode: pseudo-random load base + text size -> only partially informative
    load_base = rng.uniform(1000, 5000, size=n)
    endcode = load_base + text

    return np.stack([vsize, text, endcode, vmlib], axis=1)


def generate_workload(spec: WorkloadSpec, seed: int) -> List[Process]:
    """Deterministically build a list of Process objects for (spec, seed)."""
    rng = np.random.default_rng(seed)
    burst = _sample_burst(rng, spec.burst_mode, spec.n_processes)
    mean_burst = float(np.mean(burst))
    arrival = _sample_arrivals(rng, spec.arrival_mode, spec.n_processes,
                                spec.offered_load, mean_burst)
    priority = _sample_priorities(rng, spec.priority_mode, burst)
    feats = _sample_features(rng, burst, spec.feature_strength)

    # simultaneous-arrival ties broken by id order (already pid order below)
    procs = []
    order = np.argsort(arrival, kind="stable")
    for rank, idx in enumerate(order):
        procs.append(Process(
            pid=rank,
            arrival=float(arrival[idx]),
            priority=int(priority[idx]),
            true_burst=int(burst[idx]),
            vsize=float(feats[idx, 0]),
            text=float(feats[idx, 1]),
            endcode=float(feats[idx, 2]),
            vmlib=float(feats[idx, 3]),
        ))
    return procs


# ---------------------------------------------------------------------------
# Workload category catalogue (Section 6.2 of the paper)
# ---------------------------------------------------------------------------

def in_distribution_categories(n_processes: int = 30) -> List[WorkloadSpec]:
    return [
        WorkloadSpec("short_simultaneous", n_processes, "short", "simultaneous", "uniform", "moderate"),
        WorkloadSpec("long_simultaneous", n_processes, "long", "simultaneous", "uniform", "moderate"),
        WorkloadSpec("mixed_small", max(10, n_processes // 2), "mixed", "staggered", "uniform", "moderate"),
        WorkloadSpec("mixed_large", n_processes * 2, "mixed", "staggered", "uniform", "moderate"),
        WorkloadSpec("bursty_idle", n_processes, "mixed", "bursty", "uniform", "moderate"),
        WorkloadSpec("skewed_priority", n_processes, "mixed", "staggered", "skewed", "moderate"),
        WorkloadSpec("adversarial", n_processes, "mixed", "staggered", "adversarial", "moderate"),
        WorkloadSpec("strong_feature", n_processes, "mixed", "staggered", "uniform", "strong"),
        WorkloadSpec("weak_feature", n_processes, "mixed", "staggered", "uniform", "weak"),
        WorkloadSpec("noisy_feature", n_processes, "mixed", "staggered", "uniform", "noisy"),
    ]


def shifted_test_only_categories(n_processes: int = 30) -> List[WorkloadSpec]:
    return [
        WorkloadSpec("misleading_feature", n_processes, "mixed", "staggered", "uniform", "misleading"),
        WorkloadSpec("heavy_tailed", n_processes, "heavy_tailed", "staggered", "uniform", "moderate"),
        WorkloadSpec("large_queue", 120, "mixed", "bursty", "uniform", "moderate"),
    ]


SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}
