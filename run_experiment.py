#!/usr/bin/env python3
"""
CLI entry point for schedsim.

Quick smoke run (few seconds):
    python run_experiment.py --quick

Default pilot-scale run (used for the numbers in the accompanying paper,
~10-30 seconds):
    python run_experiment.py

Larger run (closer to the full protocol described in the paper; several
minutes depending on machine):
    python run_experiment.py --full

DEEP training run: 20,000-30,000 Q-learning episodes plus large random-search
budgets for Algorithms B, C, D, and E, with a logged learning/search curve
for every one of them (several minutes; use --episodes to pick an exact
number instead of the 20,000 default):
    python run_experiment.py --deep
    python run_experiment.py --deep --episodes 30000

CWABP (Section 9 extension) comparison only:
    python run_experiment.py --cwabp-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from schedsim.cwabp import CWABPPredictor, ExponentialAveragingPredictor, simulate_srtf
from schedsim.experiment import ExperimentConfig, run_full_experiment
from schedsim.stats import compare_algorithms, holm_correction

CWABP_CONFIGS = [
    ("stable", dict(rho=0.3, sigma=0.15, phase_change=False, correlation_break=False, heavy_tailed=False)),
    ("feature_driven_strong", dict(rho=0.9, sigma=0.3, phase_change=False, correlation_break=False, heavy_tailed=False)),
    ("feature_driven_moderate", dict(rho=0.6, sigma=0.3, phase_change=False, correlation_break=False, heavy_tailed=False)),
    ("uncorrelated", dict(rho=0.05, sigma=0.6, phase_change=False, correlation_break=False, heavy_tailed=False)),
    ("phase_change", dict(rho=0.7, sigma=0.5, phase_change=True, correlation_break=False, heavy_tailed=False)),
    ("correlation_break", dict(rho=0.8, sigma=0.3, phase_change=False, correlation_break=True, heavy_tailed=False)),
    ("heavy_tailed", dict(rho=0.6, sigma=0.3, phase_change=False, correlation_break=False, heavy_tailed=True)),
]


def _write_csv(path: str, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})


def run_main_study(cfg: ExperimentConfig, outdir: str):
    print(f"[main study] n_processes={cfg.n_processes}, "
          f"train/val/test per category={cfg.workloads_per_category_train}/"
          f"{cfg.workloads_per_category_val}/{cfg.workloads_per_category_test}")
    print(f"[main study] search/training budgets: alpha={cfg.alpha_trials} trials, "
          f"C={cfg.c_trials} trials, D={cfg.quantum_trials} trials, "
          f"E={cfg.rule_trials} trials, F(Q-learning)={cfg.qlearning_episodes} episodes")
    t0 = time.time()
    result = run_full_experiment(cfg)
    elapsed = time.time() - t0
    print(f"[main study] done in {elapsed:.1f}s")
    print(f"[main study] best burst predictor: {result['best_predictor']}")
    print(f"[main study] tuned: alpha={result['alpha']:.3f}, w_max={result['w_max']:.1f}, "
          f"gap={result['gap']}, fixed_quantum={result['fixed_quantum']}")

    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "per_workload_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(result["rows"])

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(result["summary"], f, indent=2)

    # --- learning / search curves: one CSV per tunable algorithm -----------
    hist = result["search_histories"]
    for name in ("alpha", "protected_c", "fixed_quantum", "rule_based"):
        rows = [{"trial": h["trial"], "score": h["score"], "best_so_far": h["best_so_far"],
                 "candidate": str(h["candidate"])} for h in hist[name]]
        _write_csv(os.path.join(outdir, f"search_curve_{name}.csv"), rows)
    if hist["qlearning"]:
        _write_csv(os.path.join(outdir, "qlearning_learning_curve.csv"), hist["qlearning"])
        first = hist["qlearning"][0]["mean_avg_waiting"]
        last = hist["qlearning"][-1]["mean_avg_waiting"]
        print(f"[main study] Q-learning curve: episode {hist['qlearning'][0]['episode']} "
              f"avg_waiting={first:.2f}  ->  episode {hist['qlearning'][-1]['episode']} "
              f"avg_waiting={last:.2f}  ({'better' if last < first else 'worse/noisy'})")
    else:
        print("[main study] no Q-learning learning curve logged "
              "(pass --eval-every N, e.g. via --deep, to enable it)")

    print("\n[main study] per-algorithm summary (mean over test workloads):")
    header = ["alg"] + list(next(iter(result["summary"].values())).keys())
    print(" | ".join(f"{h:>14s}" for h in header))
    for alg, s in result["summary"].items():
        row = [alg] + [f"{s[k]:.2f}" for k in header[1:]]
        print(" | ".join(f"{v:>14s}" for v in row))

    print("\n[main study] paired statistical comparisons (avg_waiting, Wilcoxon signed-rank):")
    comparisons = [("B", "A"), ("C", "B"), ("C", "A"), ("E", "D"), ("F", "D"), ("F", "E")]
    pvals, stats_out = {}, {}
    for a, b in comparisons:
        r = compare_algorithms(result["rows"], a, b, "avg_waiting")
        stats_out[f"{a}_vs_{b}"] = r
        pvals[f"{a}_vs_{b}"] = r["p_value"]
    holm = holm_correction(pvals)
    for name, r in stats_out.items():
        print(f"  {name:10s} n={r['n_pairs']:3d}  mean_ratio={r['mean_ratio']:.3f} "
              f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]  p={r['p_value']:.2e}  "
              f"p_holm={holm[name]:.2e}")

    with open(os.path.join(outdir, "statistical_tests.json"), "w") as f:
        for k in stats_out:
            stats_out[k]["p_holm"] = holm[k]
        json.dump(stats_out, f, indent=2)

    return result


def run_cwabp_study(n_seeds: int, outdir: str):
    print(f"[CWABP study] {len(CWABP_CONFIGS)} workload types x {n_seeds} seeds")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for name, params in CWABP_CONFIGS:
        ea_waits, cw_waits, ea_rmse, cw_rmse = [], [], [], []
        for s in range(n_seeds):
            r_ea = simulate_srtf(8, 14, seed=s, predictor=ExponentialAveragingPredictor(beta=0.5), **params)
            r_cw = simulate_srtf(8, 14, seed=s, predictor=CWABPPredictor(window=8, shrink=0.8, min_samples=3, beta=0.5), **params)
            ea_waits.append(r_ea.avg_waiting)
            cw_waits.append(r_cw.avg_waiting)
            ea_rmse.append(r_ea.rmse)
            cw_rmse.append(r_cw.rmse)
        row = dict(category=name, ea_avg_waiting=float(np.mean(ea_waits)),
                   cwabp_avg_waiting=float(np.mean(cw_waits)),
                   ea_rmse=float(np.mean(ea_rmse)), cwabp_rmse=float(np.mean(cw_rmse)))
        row["pct_change_waiting"] = 100 * (row["cwabp_avg_waiting"] - row["ea_avg_waiting"]) / row["ea_avg_waiting"]
        rows.append(row)
        print(f"  {name:24s} EA={row['ea_avg_waiting']:7.2f}  CWABP={row['cwabp_avg_waiting']:7.2f} "
              f"({row['pct_change_waiting']:+5.1f}%)  RMSE {row['ea_rmse']:.2f} -> {row['cwabp_rmse']:.2f}")

    _write_csv(os.path.join(outdir, "cwabp_results.csv"), rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="tiny smoke-test scale (~2s)")
    p.add_argument("--full", action="store_true", help="larger scale, closer to the full protocol (slow)")
    p.add_argument("--deep", action="store_true",
                    help="20,000+ Q-learning episodes and large random-search budgets for B/C/D/E, "
                         "with learning/search curves logged to results/ (several minutes)")
    p.add_argument("--episodes", type=int, default=None,
                    help="override Q-learning episode count directly (e.g. --episodes 30000)")
    p.add_argument("--cwabp-only", action="store_true", help="only run the CWABP (Section 9) comparison")
    p.add_argument("--outdir", default="results", help="output directory for CSV/JSON (default: results)")
    p.add_argument("--seed", type=int, default=0, help="base random seed for tuning and Q-learning training")
    p.add_argument("--quiet", action="store_true", help="suppress per-tuning-stage progress lines")
    args = p.parse_args()

    if args.cwabp_only:
        run_cwabp_study(n_seeds=10, outdir=args.outdir)
        return

    if args.quick:
        cfg = ExperimentConfig(n_processes=12, workloads_per_category_train=1,
                                workloads_per_category_val=1, workloads_per_category_test=1,
                                qlearning_episodes=40, qlearning_seed=args.seed,
                                alpha_trials=5, c_trials=5, quantum_trials=5, rule_trials=5,
                                random_state=args.seed, verbose=not args.quiet)
    elif args.deep:
        cfg = ExperimentConfig(n_processes=25, workloads_per_category_train=4,
                                workloads_per_category_val=4, workloads_per_category_test=6,
                                qlearning_episodes=args.episodes or 20000,
                                qlearning_seed=args.seed,
                                qlearning_eval_every=max(1, (args.episodes or 20000) // 100),
                                qlearning_progress_every=max(1, (args.episodes or 20000) // 20),
                                alpha_trials=200, c_trials=300, quantum_trials=100, rule_trials=600,
                                random_state=args.seed, verbose=not args.quiet)
    elif args.full:
        cfg = ExperimentConfig(n_processes=30, workloads_per_category_train=10,
                                workloads_per_category_val=6, workloads_per_category_test=10,
                                qlearning_episodes=args.episodes or 4000, qlearning_seed=args.seed,
                                qlearning_eval_every=max(1, (args.episodes or 4000) // 50),
                                alpha_trials=50, c_trials=100, quantum_trials=30, rule_trials=250,
                                random_state=args.seed, verbose=not args.quiet)
    else:
        cfg = ExperimentConfig(n_processes=25, workloads_per_category_train=3,
                                workloads_per_category_val=3, workloads_per_category_test=5,
                                qlearning_episodes=args.episodes or 600, qlearning_seed=args.seed,
                                alpha_trials=25, c_trials=60, quantum_trials=20, rule_trials=120,
                                random_state=args.seed, verbose=not args.quiet)

    run_main_study(cfg, args.outdir)
    print()
    run_cwabp_study(n_seeds=10 if not args.quick else 3, outdir=args.outdir)


if __name__ == "__main__":
    main()
