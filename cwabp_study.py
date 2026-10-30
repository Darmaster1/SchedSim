#!/usr/bin/env python3
"""
Focused CWABP-for-SRTF study (per the professor-revised plan): four
burst predictors -- exponential averaging (EA), regression-only, a
fixed-weight EA/regression blend, and CWABP -- compared fairly on the same
SRTF scheduler across seven synthetic workload types.

Seeds are disjoint by construction:
    development : [0,        1_000_000)   (not used by this script)
    validation  : [1_000_000, 2_000_000)  (hyperparameter selection only)
    test        : [2_000_000, 3_000_000)  (final, frozen evaluation)

Usage:
    python cwabp_study.py                  # run the full frozen study
    python cwabp_study.py --val-only        # just print the validation sweep
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from schedsim.cwabp import (CWABPPredictor, ExponentialAveragingPredictor,
                             FixedWeightPredictor, RegressionOnlyPredictor, simulate_srtf)
from schedsim.stats import holm_correction

N_PROCESSES = 8
BURSTS_PER_PROCESS = 14

WORKLOAD_TYPES = [
    ("stable", dict(rho=0.3, sigma=0.15, phase_change=False, correlation_break=False, heavy_tailed=False)),
    ("feature_driven_strong", dict(rho=0.9, sigma=0.3, phase_change=False, correlation_break=False, heavy_tailed=False)),
    ("feature_driven_moderate", dict(rho=0.6, sigma=0.3, phase_change=False, correlation_break=False, heavy_tailed=False)),
    ("uncorrelated", dict(rho=0.05, sigma=0.6, phase_change=False, correlation_break=False, heavy_tailed=False)),
    ("phase_change", dict(rho=0.7, sigma=0.5, phase_change=True, correlation_break=False, heavy_tailed=False)),
    ("correlation_break", dict(rho=0.8, sigma=0.3, phase_change=False, correlation_break=True, heavy_tailed=False)),
    ("heavy_tailed", dict(rho=0.6, sigma=0.3, phase_change=False, correlation_break=False, heavy_tailed=True)),
]

VAL_SEED_BASE = 1_000_000
TEST_SEED_BASE = 2_000_000
N_VAL_SEEDS = 3
N_TEST_SEEDS = 10

EA_BETA_GRID = [0.2, 0.5, 0.8]
FIXED_WEIGHT_GRID = [0.25, 0.5, 0.75]

# CWABP's own settings are RETAINED UNCHANGED from the main study, per the
# plan (item 19): they were originally hand-chosen (window=8, shrink=0.5,
# min_samples=3, beta=0.5), not selected by a validation sweep on this
# specific workload mixture. We disclose that limitation explicitly rather
# than imply CWABP received the same validation-based selection as EA and
# the fixed-weight baseline -- that asymmetry is itself a limitation
# discussed in the paper (Section V).
CWABP_SETTINGS = dict(window=8, shrink=0.5, min_samples=3, beta=0.5)


def _run_one(predictor_factory, workload_params: dict, seed: int) -> dict:
    r = simulate_srtf(N_PROCESSES, BURSTS_PER_PROCESS, seed=seed, predictor=predictor_factory(),
                       **workload_params)
    return dict(mean_waiting=r.avg_waiting, p95_waiting=r.p95_waiting, rmse=r.rmse,
                context_switches=r.context_switches, mae=r.mae, pearson=r.pearson)


def select_validation_settings(verbose: bool = True):
    """Select EA's beta and the fixed-weight blend's weight by lowest mean
    waiting time, averaged over all 7 workload types and N_VAL_SEEDS
    validation seeds. This is the single, documented selection rule (plan
    item 18); it is applied identically to both hyperparameters."""
    val_seeds = [VAL_SEED_BASE + i for i in range(N_VAL_SEEDS)]

    def mean_wait_over_val(factory) -> float:
        waits = []
        for _name, params in WORKLOAD_TYPES:
            for s in val_seeds:
                waits.append(_run_one(factory, params, s)["mean_waiting"])
        return float(np.mean(waits))

    ea_scores = {beta: mean_wait_over_val(lambda beta=beta: ExponentialAveragingPredictor(beta=beta))
                 for beta in EA_BETA_GRID}
    best_beta = min(ea_scores, key=ea_scores.get)

    fw_scores = {w: mean_wait_over_val(lambda w=w: FixedWeightPredictor(weight=w, **CWABP_SETTINGS_NO_SHRINK()))
                 for w in FIXED_WEIGHT_GRID}
    best_weight = min(fw_scores, key=fw_scores.get)

    if verbose:
        print("[validation] EA beta sweep (mean waiting, lower is better):")
        for b, s in ea_scores.items():
            flag = " <- selected" if b == best_beta else ""
            print(f"    beta={b}: {s:.3f}{flag}")
        print("[validation] Fixed-weight sweep (mean waiting, lower is better):")
        for w, s in fw_scores.items():
            flag = " <- selected" if w == best_weight else ""
            print(f"    weight={w}: {s:.3f}{flag}")

    return best_beta, best_weight


def CWABP_SETTINGS_NO_SHRINK():
    d = dict(CWABP_SETTINGS)
    d.pop("shrink")
    return d


def run_frozen_test(best_beta: float, best_weight: float, outdir: str = "results"):
    os.makedirs(outdir, exist_ok=True)
    test_seeds = [TEST_SEED_BASE + i for i in range(N_TEST_SEEDS)]

    methods = {
        "EA": lambda: ExponentialAveragingPredictor(beta=best_beta),
        "RegressionOnly": lambda: RegressionOnlyPredictor(**CWABP_SETTINGS_NO_SHRINK()),
        "FixedWeight": lambda: FixedWeightPredictor(weight=best_weight, **CWABP_SETTINGS_NO_SHRINK()),
        "CWABP": lambda: CWABPPredictor(**CWABP_SETTINGS),
    }

    rows = []
    for workload_type, params in WORKLOAD_TYPES:
        for seed in test_seeds:
            workload_id = f"{workload_type}#{seed}"
            for method_name, factory in methods.items():
                m = _run_one(factory, params, seed)
                rows.append({"workload_type": workload_type, "workload_id": workload_id,
                             "seed": seed, "method": method_name, **m})

    fieldnames = list(rows[0].keys())
    with open(os.path.join(outdir, "cwabp_focused_per_run.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def paired_stats_by_workload(rows: list, outdir: str = "results"):
    """For each workload type, paired CWABP-vs-{EA, RegressionOnly,
    FixedWeight} differences in mean_waiting across matched seeds, with a
    Wilcoxon signed-rank test and a percentile bootstrap CI for the mean
    difference, Holm-corrected across the 3-comparison family WITHIN each
    workload type (plan items 31-33)."""
    from scipy.stats import wilcoxon

    results = {}
    comparisons = [("CWABP", "EA"), ("CWABP", "RegressionOnly"), ("CWABP", "FixedWeight")]

    for workload_type, _params in WORKLOAD_TYPES:
        wl_rows = [r for r in rows if r["workload_type"] == workload_type]
        by_method = {m: {r["seed"]: r["mean_waiting"] for r in wl_rows if r["method"] == m}
                     for m in ("EA", "RegressionOnly", "FixedWeight", "CWABP")}
        entry = {}
        pvals = {}
        for a, b in comparisons:
            seeds = sorted(set(by_method[a]) & set(by_method[b]))
            xa = np.array([by_method[a][s] for s in seeds])
            xb = np.array([by_method[b][s] for s in seeds])
            diff = xa - xb  # CWABP minus comparator: negative = CWABP faster
            if np.allclose(diff, 0):
                stat, p = float("nan"), float("nan")
            else:
                try:
                    stat, p = wilcoxon(xa, xb)
                except ValueError:
                    stat, p = float("nan"), float("nan")

            rng = np.random.default_rng(0)
            n = len(diff)
            boot_means = []
            for _ in range(2000):
                idx = rng.integers(0, n, size=n)
                boot_means.append(diff[idx].mean())
            ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

            key = f"{a}_vs_{b}"
            entry[key] = {"n_pairs": n, "mean_diff": float(diff.mean()), "ci_low": ci_low,
                          "ci_high": ci_high, "p_value": float(p),
                          "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0)}
            pvals[key] = p

        holm = holm_correction(pvals)
        for key in entry:
            entry[key]["p_holm"] = holm[key]
        results[workload_type] = entry

    with open(os.path.join(outdir, "cwabp_focused_stats.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-only", action="store_true")
    p.add_argument("--outdir", default="results")
    args = p.parse_args()

    best_beta, best_weight = select_validation_settings()
    if args.val_only:
        return

    print(f"\n[frozen config] EA beta={best_beta}, fixed-weight={best_weight}, "
          f"CWABP settings (unchanged)={CWABP_SETTINGS}")
    rows = run_frozen_test(best_beta, best_weight, args.outdir)
    print(f"[test] {len(rows)} runs written to {args.outdir}/cwabp_focused_per_run.csv")

    stats = paired_stats_by_workload(rows, args.outdir)
    print("\n[stats] CWABP vs. each comparator, per workload type (mean_waiting, Holm-corrected):")
    for workload_type, entry in stats.items():
        print(f"  {workload_type}:")
        for comp, r in entry.items():
            sig = "significant" if r["ci_excludes_zero"] else "CI includes zero -- not established"
            print(f"    {comp}: mean_diff={r['mean_diff']:+.3f}  "
                  f"CI=[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]  p_holm={r['p_holm']:.3g}  ({sig})")


if __name__ == "__main__":
    main()
