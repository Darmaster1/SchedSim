"""
Experiment orchestration: ties workload generation, burst prediction, the
six scheduling algorithms, and metrics together into one reproducible run.

This is a *scaled-down* but fully functional instance of the protocol
described in the paper (Sections 6-11): smaller sample sizes and a smaller
Q-learning training budget so that the whole pipeline runs in well under a
minute on a laptop, while exercising every code path (leakage control,
train/val/test separation, parameter tuning, statistical comparison).
Scale up via the CLI flags in run_experiment.py for a fuller run.
"""
from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .metrics import compute_metrics, metrics_to_dict
from .predictors import (evaluate_predictor, fit_predictor, out_of_fold_predictions,
                          select_best_predictor)
from .qlearning import FrozenQPolicy, train_qlearning_agent
from .schedulers import (AlgorithmA_Priority, AlgorithmB_WeightedML, AlgorithmC_ProtectedML,
                          FixedQuantumPolicy, QUANTA, RuleBasedQuantumPolicy,
                          run_priority_family, run_round_robin_family)
from .workload import (SEED_OFFSETS, WorkloadSpec, generate_workload,
                        in_distribution_categories, shifted_test_only_categories)


@dataclass
class ExperimentConfig:
    n_processes: int = 24
    workloads_per_category_train: int = 4
    workloads_per_category_val: int = 2
    workloads_per_category_test: int = 4
    context_switch_cost: float = 1.0
    qlearning_episodes: int = 300
    qlearning_seed: int = 0
    random_state: int = 0


def _seeds(spec_idx: int, split: str, count: int) -> List[int]:
    base = SEED_OFFSETS[split] + spec_idx * 10_000
    return [base + i for i in range(count)]


def _predicted_dict(procs, fp) -> Dict[int, float]:
    preds = fp.predict(procs)
    return {p.pid: float(v) for p, v in zip(procs, preds)}


def build_pools(cfg: ExperimentConfig) -> Tuple[dict, dict, dict]:
    """Return {split: {category_name: [(procs, spec)]}} for train/val/test."""
    in_dist = in_distribution_categories(cfg.n_processes)
    shifted = shifted_test_only_categories(cfg.n_processes)

    pools = {"train": {}, "val": {}, "test": {}}
    counts = {"train": cfg.workloads_per_category_train,
              "val": cfg.workloads_per_category_val,
              "test": cfg.workloads_per_category_test}

    for i, spec in enumerate(in_dist):
        for split in ("train", "val", "test"):
            seeds = _seeds(i, split, counts[split])
            pools[split][spec.name] = [(generate_workload(spec, s), spec) for s in seeds]

    for i, spec in enumerate(shifted, start=len(in_dist)):
        seeds = _seeds(i, "test", counts["test"])
        pools["test"][spec.name] = [(generate_workload(spec, s), spec) for s in seeds]

    return pools["train"], pools["val"], pools["test"]


def flatten(pool: dict) -> List:
    out = []
    for lst in pool.values():
        for procs, spec in lst:
            out.extend(procs)
    return out


def fit_burst_predictor(train_pool: dict, val_pool: dict):
    train_procs = flatten(train_pool)
    val_procs = flatten(val_pool)
    best_name, best_fp, all_metrics = select_best_predictor(train_procs, val_procs)
    return best_name, best_fp, all_metrics


def tune_alpha(val_pool: dict, fp) -> float:
    best_alpha, best_score = 0.5, float("inf")
    for alpha in np.arange(0.0, 1.01, 0.2):
        waits = []
        for lst in val_pool.values():
            for procs, spec in lst:
                predicted = _predicted_dict(procs, fp)
                log = run_priority_family(procs, AlgorithmB_WeightedML(alpha), predicted)
                m = compute_metrics(procs, log)
                waits.append(m.avg_waiting)
        score = float(np.mean(waits))
        if score < best_score:
            best_score, best_alpha = score, float(alpha)
    return best_alpha


def tune_protected_c(val_pool: dict, fp, alpha: float) -> Tuple[float, int]:
    """Select (W_max, gap) to minimise mean max-waiting, subject to mean
    average waiting being <= 1.10x Algorithm B's. If NO grid point satisfies
    the constraint (possible when B is already close to SJF-optimal), fall
    back to the feasible-closest point -- the one with the smallest average
    waiting -- rather than silently returning an untested default; the
    constraint is a soft preference, not something to fake compliance with.
    """
    b_waits = []
    for lst in val_pool.values():
        for procs, spec in lst:
            predicted = _predicted_dict(procs, fp)
            log = run_priority_family(procs, AlgorithmB_WeightedML(alpha), predicted)
            b_waits.append(compute_metrics(procs, log).avg_waiting)
    b_avg = float(np.mean(b_waits))
    limit = 1.10 * b_avg

    feasible_best, feasible_score = None, float("inf")
    fallback_best, fallback_score = None, float("inf")  # smallest avg_wait seen, any constraint status

    for w_max in (20.0, 40.0, 70.0, 100.0):
        for gap in (1, 2, 3, 5):
            c_waits, c_max = [], []
            for lst in val_pool.values():
                for procs, spec in lst:
                    predicted = _predicted_dict(procs, fp)
                    log = run_priority_family(procs, AlgorithmC_ProtectedML(alpha, w_max, gap), predicted)
                    m = compute_metrics(procs, log)
                    c_waits.append(m.avg_waiting)
                    c_max.append(m.max_waiting)
            mean_wait = float(np.mean(c_waits))
            mean_max = float(np.mean(c_max))

            if mean_wait <= limit and mean_max < feasible_score:
                feasible_score, feasible_best = mean_max, (w_max, gap)
            if mean_wait < fallback_score:
                fallback_score, fallback_best = mean_wait, (w_max, gap)

    return feasible_best if feasible_best is not None else fallback_best


def tune_fixed_quantum(val_pool: dict) -> float:
    best_q, best_score = QUANTA[1], float("inf")
    for q in QUANTA:
        waits = []
        for lst in val_pool.values():
            for procs, spec in lst:
                predicted = {p.pid: float(p.true_burst) for p in procs}  # D ignores predictions
                log = run_round_robin_family(procs, FixedQuantumPolicy(q), predicted)
                waits.append(compute_metrics(procs, log).avg_waiting)
        score = float(np.mean(waits))
        if score < best_score:
            best_score, best_q = score, q
    return best_q


def tune_rule_based(val_pool: dict, fp) -> RuleBasedQuantumPolicy:
    best_policy, best_score = None, float("inf")
    for qt in (4, 6, 8):
        for mt in (10.0, 20.0, 30.0):
            for wt in (20.0, 30.0, 50.0):
                policy = RuleBasedQuantumPolicy(queue_thresh=qt, median_thresh=mt, wait_thresh=wt)
                waits = []
                for lst in val_pool.values():
                    for procs, spec in lst:
                        predicted = _predicted_dict(procs, fp)
                        log = run_round_robin_family(procs, policy, predicted)
                        waits.append(compute_metrics(procs, log).avg_waiting)
                score = float(np.mean(waits))
                if score < best_score:
                    best_score, best_policy = score, policy
    return best_policy


def run_full_experiment(cfg: ExperimentConfig) -> dict:
    train_pool, val_pool, test_pool = build_pools(cfg)

    best_name, fp, pred_metrics = fit_burst_predictor(train_pool, val_pool)

    alpha = tune_alpha(val_pool, fp)
    w_max, gap = tune_protected_c(val_pool, fp, alpha)
    fixed_q = tune_fixed_quantum(val_pool)
    rule_policy = tune_rule_based(val_pool, fp)

    def predicted_lookup(spec: WorkloadSpec, seed: int):
        procs = generate_workload(spec, seed)
        predicted = _predicted_dict(procs, fp)
        return procs, predicted

    train_specs = [spec for lst in train_pool.values() for (_, spec) in lst]
    train_specs = list({s.name: s for s in train_specs}.values())
    agent = train_qlearning_agent(train_specs, predicted_lookup,
                                   n_episodes=cfg.qlearning_episodes,
                                   seed=cfg.qlearning_seed,
                                   context_switch_cost=cfg.context_switch_cost)
    frozen_policy = FrozenQPolicy(agent.q_table)

    algorithms = {
        "A": lambda procs, predicted: run_priority_family(procs, AlgorithmA_Priority(), predicted,
                                                            cfg.context_switch_cost),
        "B": lambda procs, predicted: run_priority_family(procs, AlgorithmB_WeightedML(alpha), predicted,
                                                            cfg.context_switch_cost),
        "C": lambda procs, predicted: run_priority_family(procs, AlgorithmC_ProtectedML(alpha, w_max, gap),
                                                            predicted, cfg.context_switch_cost),
        "D": lambda procs, predicted: run_round_robin_family(procs, FixedQuantumPolicy(fixed_q), predicted,
                                                               cfg.context_switch_cost),
        "E": lambda procs, predicted: run_round_robin_family(procs, rule_policy, predicted,
                                                               cfg.context_switch_cost),
        "F": lambda procs, predicted: run_round_robin_family(procs, frozen_policy, predicted,
                                                               cfg.context_switch_cost),
    }

    rows = []
    for category, lst in test_pool.items():
        for w_idx, (procs, spec) in enumerate(lst):
            predicted = _predicted_dict(procs, fp)
            workload_id = f"{category}#{w_idx}"
            for alg_name, runner in algorithms.items():
                log = runner(procs, predicted)
                m = compute_metrics(procs, log)
                row = {"category": category, "workload_id": workload_id,
                       "algorithm": alg_name, **metrics_to_dict(m)}
                rows.append(row)

    summary = {}
    for alg_name in algorithms:
        alg_rows = [r for r in rows if r["algorithm"] == alg_name]
        summary[alg_name] = {
            metric: float(np.mean([r[metric] for r in alg_rows]))
            for metric in ("avg_waiting", "avg_turnaround", "avg_response", "max_waiting",
                           "p95_waiting", "context_switches", "utilization",
                           "starvation_violations", "jain_index")
        }

    return {
        "best_predictor": best_name,
        "predictor_metrics": pred_metrics,
        "alpha": alpha,
        "w_max": w_max,
        "gap": gap,
        "fixed_quantum": fixed_q,
        "rule_policy": rule_policy,
        "rows": rows,
        "summary": summary,
    }
