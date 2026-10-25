"""
Paired statistical comparisons between algorithms (Section 10.2 of the
paper): a two-sided Wilcoxon signed-rank test per metric, computed on
per-workload paired values, plus a percentile bootstrap CI for the mean
ratio between two algorithms.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon


def pair_by_workload(rows: List[dict], alg_a: str, alg_b: str, metric: str
                      ) -> Tuple[np.ndarray, np.ndarray]:
    by_workload_a = {r["workload_id"]: r[metric] for r in rows if r["algorithm"] == alg_a}
    by_workload_b = {r["workload_id"]: r[metric] for r in rows if r["algorithm"] == alg_b}
    common = sorted(set(by_workload_a) & set(by_workload_b))
    a = np.array([by_workload_a[w] for w in common], dtype=float)
    b = np.array([by_workload_b[w] for w in common], dtype=float)
    return a, b


def bootstrap_ci_mean_ratio(a: np.ndarray, b: np.ndarray, n_resamples: int = 2000,
                             seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for mean(a)/mean(b), resampling workload pairs."""
    rng = np.random.default_rng(seed)
    n = len(a)
    ratios = []
    point = float(np.mean(a) / np.mean(b)) if np.mean(b) != 0 else float("nan")
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ra, rb = a[idx], b[idx]
        if np.mean(rb) == 0:
            continue
        ratios.append(np.mean(ra) / np.mean(rb))
    ratios = np.array(ratios)
    lo = float(np.percentile(ratios, 100 * alpha / 2))
    hi = float(np.percentile(ratios, 100 * (1 - alpha / 2)))
    return point, lo, hi


def compare_algorithms(rows: List[dict], alg_a: str, alg_b: str, metric: str = "avg_waiting"
                        ) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank test (alg_a vs alg_b) + bootstrap CI for
    the mean ratio, both computed on per-workload paired values of `metric`.
    """
    a, b = pair_by_workload(rows, alg_a, alg_b, metric)
    if len(a) < 2 or np.allclose(a, b):
        return {"n_pairs": len(a), "statistic": float("nan"), "p_value": float("nan"),
                "mean_ratio": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        stat, p = float("nan"), float("nan")
    point, lo, hi = bootstrap_ci_mean_ratio(a, b)
    return {"n_pairs": len(a), "statistic": float(stat), "p_value": float(p),
            "mean_ratio": point, "ci_low": lo, "ci_high": hi}


def holm_correction(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm step-down correction across a family of named p-values."""
    items = sorted(p_values.items(), key=lambda kv: (np.nan_to_num(kv[1], nan=1.0)))
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        if np.isnan(p):
            adjusted[name] = float("nan")
            continue
        val = min(1.0, (m - i) * p)
        running_max = max(running_max, val)
        adjusted[name] = running_max
    return adjusted
