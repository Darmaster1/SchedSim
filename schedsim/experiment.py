"""
Experiment orchestration: ties workload generation, burst prediction, the
six scheduling algorithms, and metrics together into one reproducible run.

Every tunable algorithm (B's alpha, C's safeguard parameters, D's quantum,
E's rule thresholds, F's Q-table) is optimised by an iterative search over
many trials/episodes rather than a handful of fixed grid points, and each
search records a trial-by-trial "best score so far" history -- so, just
like a normal ML training run, you can watch every algorithm actually get
better as the trial/episode budget increases. See `run_experiment.py` for
how to scale these budgets up to tens of thousands of trials/episodes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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

    # Algorithm F (Q-learning): training episode budget. This is the one
    # that scales most dramatically -- set to 20_000-30_000 for a "deep"
    # training run; see train_qlearning_agent's eval_every for the learning
    # curve this produces.
    qlearning_episodes: int = 300
    qlearning_seed: int = 0
    qlearning_eval_every: int = 0       # 0 disables learning-curve logging
    qlearning_progress_every: int = 0   # 0 disables console progress prints

    # Algorithms B, C, D, E: random-search trial budgets. Each of these
    # also produces a "best score so far vs. trial number" history.
    alpha_trials: int = 25
    c_trials: int = 60
    quantum_trials: int = 20
    rule_trials: int = 120

    random_state: int = 0
    verbose: bool = False


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


# ---------------------------------------------------------------------------
# Generic random-search helper: run `n_trials` randomly sampled candidates,
# score each on the validation pool, and record a "best score so far" curve
# -- the same monitoring pattern as a Q-learning learning curve, applied to
# algorithms that are tuned by search rather than by gradient-style updates.
# ---------------------------------------------------------------------------

def _random_search(n_trials: int, seed: int, sample_fn, score_fn, minimize=True):
    """sample_fn(rng) -> candidate; score_fn(candidate) -> float.
    Returns (best_candidate, best_score, history) where history is a list of
    {"trial": int, "score": float, "best_so_far": float, "candidate": ...}.
    """
    rng = np.random.default_rng(seed)
    best_candidate, best_score = None, float("inf") if minimize else -float("inf")
    history = []
    for t in range(n_trials):
        cand = sample_fn(rng)
        score = score_fn(cand)
        improved = (score < best_score) if minimize else (score > best_score)
        if improved:
            best_score, best_candidate = score, cand
        history.append({"trial": t + 1, "score": float(score),
                         "best_so_far": float(best_score), "candidate": cand})
    return best_candidate, best_score, history


def _mean_metric(val_pool: dict, runner, metric: str) -> float:
    vals = []
    for lst in val_pool.values():
        for procs, spec in lst:
            log = runner(procs)
            m = compute_metrics(procs, log)
            vals.append(getattr(m, metric))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Algorithm B: alpha (blend weight)
# ---------------------------------------------------------------------------

def tune_alpha_search(val_pool: dict, fp, n_trials: int = 25, seed: int = 0
                       ) -> Tuple[float, List[dict]]:
    def sample(rng):
        return float(rng.uniform(0.0, 1.0))

    def score(alpha):
        def runner(procs):
            predicted = _predicted_dict(procs, fp)
            return run_priority_family(procs, AlgorithmB_WeightedML(alpha), predicted)
        return _mean_metric(val_pool, runner, "avg_waiting")

    best_alpha, _, history = _random_search(n_trials, seed, sample, score)
    # Always include the two informative endpoints (pure priority / pure SJF)
    # and the midpoint, since a pure random search can otherwise miss them.
    for alpha in (0.0, 0.5, 1.0):
        s = score(alpha)
        if s < history[-1]["best_so_far"]:
            best_alpha = alpha
        history.append({"trial": len(history) + 1, "score": float(s),
                         "best_so_far": min(s, history[-1]["best_so_far"]), "candidate": alpha})
        if history[-1]["best_so_far"] == s:
            best_alpha = alpha
    return best_alpha, history


# ---------------------------------------------------------------------------
# Algorithm C: (W_max, gap)
# ---------------------------------------------------------------------------

def tune_protected_c_search(val_pool: dict, fp, alpha: float, n_trials: int = 60,
                             seed: int = 0) -> Tuple[Tuple[float, int], List[dict]]:
    """As in the fixed-grid version, (W_max, gap) is chosen to minimise mean
    max-waiting subject to mean average waiting <= 1.10x Algorithm B's; if no
    sampled candidate satisfies the constraint, the feasible-closest
    candidate (smallest average waiting) is used as a fallback rather than
    an unvalidated default. With `n_trials` in the hundreds-to-thousands
    range this covers the (W_max, gap) space far more thoroughly than a
    16-point grid.
    """
    def runner_b(procs):
        predicted = _predicted_dict(procs, fp)
        return run_priority_family(procs, AlgorithmB_WeightedML(alpha), predicted)
    b_avg = _mean_metric(val_pool, runner_b, "avg_waiting")
    limit = 1.10 * b_avg

    def sample(rng):
        w_max = float(rng.uniform(10.0, 150.0))
        gap = int(rng.integers(0, 8))
        return (w_max, gap)

    feasible_best, feasible_score = None, float("inf")
    history = []
    fallback_best, fallback_score = None, float("inf")

    rng = np.random.default_rng(seed)
    for t in range(n_trials):
        w_max, gap = sample(rng)

        def runner_c(procs):
            predicted = _predicted_dict(procs, fp)
            return run_priority_family(procs, AlgorithmC_ProtectedML(alpha, w_max, gap), predicted)

        c_waits, c_max = [], []
        for lst in val_pool.values():
            for procs, spec in lst:
                m = compute_metrics(procs, runner_c(procs))
                c_waits.append(m.avg_waiting)
                c_max.append(m.max_waiting)
        mean_wait = float(np.mean(c_waits))
        mean_max = float(np.mean(c_max))

        if mean_wait <= limit and mean_max < feasible_score:
            feasible_score, feasible_best = mean_max, (w_max, gap)
        if mean_wait < fallback_score:
            fallback_score, fallback_best = mean_wait, (w_max, gap)

        current_best_score = feasible_score if feasible_best is not None else fallback_score
        history.append({"trial": t + 1, "score": mean_max, "mean_wait": mean_wait,
                         "feasible": mean_wait <= limit,
                         "best_so_far": float(current_best_score),
                         "candidate": (w_max, gap)})

    best = feasible_best if feasible_best is not None else fallback_best
    return best, history


# ---------------------------------------------------------------------------
# Algorithm D: fixed quantum
# ---------------------------------------------------------------------------

def tune_fixed_quantum_search(val_pool: dict, n_trials: int = 20, seed: int = 0
                               ) -> Tuple[float, List[dict]]:
    """D's quantum is searched over a wider integer range than the four
    candidates E/F choose among (D is just a single dial, not tied to an
    RL action space), so more trials genuinely explore more settings."""
    def sample(rng):
        return int(rng.integers(1, 41))

    def score(q):
        def runner(procs):
            predicted = {p.pid: float(p.true_burst) for p in procs}  # D ignores predictions
            return run_round_robin_family(procs, FixedQuantumPolicy(q), predicted)
        return _mean_metric(val_pool, runner, "avg_waiting")

    best_q, _, history = _random_search(n_trials, seed, sample, score)
    return float(best_q), history


# ---------------------------------------------------------------------------
# Algorithm E: rule-based quantum thresholds
# ---------------------------------------------------------------------------

def tune_rule_based_search(val_pool: dict, fp, n_trials: int = 120, seed: int = 0
                            ) -> Tuple[RuleBasedQuantumPolicy, List[dict]]:
    def sample(rng):
        return RuleBasedQuantumPolicy(
            queue_thresh=int(rng.integers(2, 12)),
            median_thresh=float(rng.uniform(4.0, 45.0)),
            wait_thresh=float(rng.uniform(8.0, 90.0)),
            base_idx=int(rng.integers(0, 3)),
        )

    def score(policy):
        def runner(procs):
            predicted = _predicted_dict(procs, fp)
            return run_round_robin_family(procs, policy, predicted)
        return _mean_metric(val_pool, runner, "avg_waiting")

    best_policy, _, history = _random_search(n_trials, seed, sample, score)
    return best_policy, history


def run_full_experiment(cfg: ExperimentConfig) -> dict:
    train_pool, val_pool, test_pool = build_pools(cfg)

    best_name, fp, pred_metrics = fit_burst_predictor(train_pool, val_pool)

    if cfg.verbose:
        print(f"[tuning] alpha: {cfg.alpha_trials} trials ...")
    alpha, alpha_history = tune_alpha_search(val_pool, fp, cfg.alpha_trials, cfg.random_state)

    if cfg.verbose:
        print(f"[tuning] C (w_max, gap): {cfg.c_trials} trials ...")
    (w_max, gap), c_history = tune_protected_c_search(val_pool, fp, alpha, cfg.c_trials,
                                                        cfg.random_state + 1)

    if cfg.verbose:
        print(f"[tuning] D (fixed quantum): {cfg.quantum_trials} trials ...")
    fixed_q, quantum_history = tune_fixed_quantum_search(val_pool, cfg.quantum_trials,
                                                           cfg.random_state + 2)

    if cfg.verbose:
        print(f"[tuning] E (rule thresholds): {cfg.rule_trials} trials ...")
    rule_policy, rule_history = tune_rule_based_search(val_pool, fp, cfg.rule_trials,
                                                         cfg.random_state + 3)

    def predicted_lookup(spec: WorkloadSpec, seed: int):
        procs = generate_workload(spec, seed)
        predicted = _predicted_dict(procs, fp)
        return procs, predicted

    train_specs = [spec for lst in train_pool.values() for (_, spec) in lst]
    train_specs = list({s.name: s for s in train_specs}.values())
    val_specs = [spec for lst in val_pool.values() for (_, spec) in lst]
    val_specs = list({s.name: s for s in val_specs}.values())

    if cfg.verbose:
        print(f"[tuning] F (Q-learning): {cfg.qlearning_episodes} episodes ...")
    agent, qlearning_history = train_qlearning_agent(
        train_specs, predicted_lookup,
        n_episodes=cfg.qlearning_episodes,
        seed=cfg.qlearning_seed,
        context_switch_cost=cfg.context_switch_cost,
        eval_every=cfg.qlearning_eval_every,
        eval_specs=val_specs if val_specs else None,
        eval_seeds=[555001],
        progress_every=cfg.qlearning_progress_every,
    )
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
        "search_histories": {
            "alpha": alpha_history,
            "protected_c": c_history,
            "fixed_quantum": quantum_history,
            "rule_based": rule_history,
            "qlearning": qlearning_history,
        },
    }
