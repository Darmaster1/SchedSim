"""
Random WorkloadSpec sampler for continuous training and property-based testing.

Generates WorkloadSpec instances covering the full combinatorial parameter
space: burst distributions, arrival patterns, priority modes, feature strength,
and process counts. This is far beyond the 13 fixed categories in the paper's
pilot study. Each call to `random_workload_spec` is reproducible via the
caller's RNG; pass the same Generator + call sequence to get identical results.

Used by:
  - schedsim/continuous.py  (data-flywheel training)
  - tests/test_properties.py (hypothesis strategy seed)
  - train.py                 (CLI)
"""
from __future__ import annotations

import numpy as np

from .workload import WorkloadSpec

_BURST_MODES = ("short", "long", "mixed", "heavy_tailed")
_ARRIVAL_MODES = ("simultaneous", "staggered", "bursty")
_PRIORITY_MODES = ("uniform", "skewed", "adversarial")
_FEATURE_STRENGTHS = ("strong", "moderate", "weak", "noisy", "misleading")


def random_workload_spec(
    rng: np.random.Generator,
    min_processes: int = 5,
    max_processes: int = 120,
    name_prefix: str = "rand",
) -> WorkloadSpec:
    """Sample one random WorkloadSpec from the full parameter space.

    Parameters
    ----------
    rng:            NumPy Generator supplied by the caller; all randomness
                    flows through this object so results are reproducible.
    min_processes:  Minimum n_processes (inclusive).
    max_processes:  Maximum n_processes (inclusive).
    name_prefix:    Prefix prepended to the spec's `name` field.

    Returns
    -------
    WorkloadSpec whose `name` encodes key parameters for traceability.
    """
    n   = int(rng.integers(min_processes, max_processes + 1))
    bm  = str(rng.choice(_BURST_MODES))
    am  = str(rng.choice(_ARRIVAL_MODES))
    pm  = str(rng.choice(_PRIORITY_MODES))
    fs  = str(rng.choice(_FEATURE_STRENGTHS))
    load = round(float(rng.uniform(0.2, 1.8)), 3)
    name = f"{name_prefix}_{n}_{bm[:3]}_{am[:3]}_{fs[:3]}"
    return WorkloadSpec(
        name=name,
        n_processes=n,
        burst_mode=bm,
        arrival_mode=am,
        priority_mode=pm,
        feature_strength=fs,
        offered_load=load,
    )


def random_spec_batch(
    n_specs: int,
    seed: int = 42,
    min_processes: int = 5,
    max_processes: int = 120,
) -> list:
    """Generate a deterministic batch of n_specs random WorkloadSpec objects.

    Convenience wrapper over `random_workload_spec` with a fresh Generator
    seeded from `seed`; suitable for experiment scripts and CLI helpers.
    """
    rng = np.random.default_rng(seed)
    return [
        random_workload_spec(rng, min_processes=min_processes,
                              max_processes=max_processes)
        for _ in range(n_specs)
    ]
