"""
Continuous training loop for schedsim.

The ``ContinuousTrainer`` class orchestrates the data-flywheel for ML
burst-length predictors and the resumable training loop for the Q-learning
agent (Algorithm F):

1. Generate random workloads and append their (feature, burst) rows to the
   persistent training pool.
2. Retrain all five ML candidate models on the full accumulated pool; save
   the best predictor checkpoint.
3. Resume Q-learning training from the latest Q-table checkpoint, using the
   best saved predictor for burst predictions.
4. Evaluate the checkpointed predictor and Q-agent on the standard frozen
   test workloads (the same categories used in the paper, seed range disjoint
   from the pool data).
5. Append a structured summary to the checkpoint training log.

This module is the backend used by ``train.py``.  It does not touch
``run_experiment.py`` or the paper's fixed experimental protocol.
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

from .checkpoint import (
    _append_log, load_predictor, load_qtable,
    save_predictor, save_qtable,
)
from .metrics import compute_metrics
from .pool import TrainingPool
from .predictors import MODEL_FACTORIES, FittedPredictor
from .qlearning import QLearningAgent, discretize_state, run_round_robin_family
from .random_spec import random_workload_spec
from .schedulers import (
    AlgorithmA_Priority, FixedQuantumPolicy,
    run_priority_family,
)
from .workload import (
    SEED_OFFSETS, generate_workload,
    in_distribution_categories, shifted_test_only_categories,
)

# Pool seeds live in [3_000_000, 4_000_000) -- disjoint from train/val/test ranges.
_POOL_SEED_LOW  = 3_000_000
_POOL_SEED_HIGH = 4_000_000

# Frozen evaluation: a small, fixed test budget that never changes.
_EVAL_SEEDS_PER_CAT = 3


@dataclass
class RoundResult:
    """Summary produced by one call to `ContinuousTrainer.run_round`."""
    round_num:             int
    pool_size:             int
    workloads_generated:   int
    predictor_name:        str
    predictor_val_rmse:    float
    predictor_test_rmse:   float
    q_episodes_this_round: int
    q_episodes_total:      int
    eval_alg_A_avg_wait:   float
    eval_alg_D_avg_wait:   float
    eval_alg_F_avg_wait:   Optional[float]


class ContinuousTrainer:
    """
    Orchestrates the continuous training pipeline.

    Parameters
    ----------
    data_dir:
        Directory for the training pool (default: ``<repo_root>/data/``).
    ckpt_dir:
        Directory for model checkpoints (default: ``<repo_root>/checkpoints/``).
    n_processes:
        Process count for generated workloads.  Workloads generated during
        the pool phase use a random n_processes in
        [max(5, n_processes//2), n_processes]; those used for Q-agent training
        use exactly n_processes.
    """

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR   = os.path.join(_REPO_ROOT, "data")
    CKPT_DIR   = os.path.join(_REPO_ROOT, "checkpoints")

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        ckpt_dir: str = CKPT_DIR,
        n_processes: int = 30,
    ):
        self.pool        = TrainingPool(data_dir)
        self.ckpt_dir    = ckpt_dir
        self.n_processes = n_processes
        os.makedirs(ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Generate workloads and append to pool
    # ------------------------------------------------------------------

    def generate_and_pool(
        self,
        n_workloads: int,
        seed: int,
    ) -> int:
        """Generate ``n_workloads`` random workloads and append to the pool.

        Seeds are drawn from the pool seed range [3M, 4M) so they can never
        collide with the train/val/test seeds used by ``run_experiment.py``.

        Returns
        -------
        Total number of (feature, burst) rows appended.
        """
        spec_rng     = np.random.default_rng(seed)
        seed_rng     = np.random.default_rng(seed + 1)
        min_procs    = max(5, self.n_processes // 2)
        max_procs    = self.n_processes

        total = 0
        for _ in range(n_workloads):
            spec         = random_workload_spec(
                spec_rng, min_processes=min_procs, max_processes=max_procs)
            wl_seed      = int(seed_rng.integers(_POOL_SEED_LOW, _POOL_SEED_HIGH))
            procs        = generate_workload(spec, wl_seed)
            total       += self.pool.append(procs, spec.name, wl_seed)
        return total

    # ------------------------------------------------------------------
    # Step 2: Train ML predictors on the full accumulated pool
    # ------------------------------------------------------------------

    def train_predictors(self) -> Dict:
        """Fit all five candidate models on the full pool.

        Splits 80 / 20 (shuffled, seed-42 for reproducibility).
        Selects the winner by lowest 20 %-split RMSE, then refits the winner
        on the full pool for deployment.  Saves a predictor checkpoint.

        Returns
        -------
        Dict with keys: best_model, val_rmse, pool_size, all_metrics.
        """
        X, y = self.pool.load_xy()
        n    = len(X)

        rng     = np.random.default_rng(42)
        idx     = rng.permutation(n)
        n_train = int(0.8 * n)
        tr_idx, va_idx = idx[:n_train], idx[n_train:]

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        scaler   = StandardScaler().fit(X_tr)
        X_tr_s   = scaler.transform(X_tr)
        X_va_s   = scaler.transform(X_va)

        best_name, best_rmse, best_model = None, math.inf, None
        all_metrics: Dict = {}

        for name, factory in MODEL_FACTORIES.items():
            m = factory()
            m.fit(X_tr_s, y_tr)
            preds = np.maximum(m.predict(X_va_s), 1.0)
            rmse  = float(np.sqrt(np.mean((y_va - preds) ** 2)))
            mae   = float(np.mean(np.abs(y_va - preds)))
            all_metrics[name] = {
                "val_rmse": rmse, "val_mae": mae,
                "n_train": n_train, "n_val": len(va_idx),
            }
            if rmse < best_rmse:
                best_rmse, best_name, best_model = rmse, name, m

        # Refit winner on full pool for maximum deployment accuracy.
        X_full_s    = scaler.transform(X)
        best_full   = MODEL_FACTORIES[best_name]()
        best_full.fit(X_full_s, y)

        fp = FittedPredictor(
            name=best_name, scaler=scaler,
            model=best_full, use_endcode=True,
        )
        metadata = {
            "best_model": best_name,
            "val_rmse":   best_rmse,
            "pool_size":  self.pool.size,
            "all_metrics_json": str(all_metrics),
        }
        save_predictor(fp, metadata, self.ckpt_dir)
        return {"best_model": best_name, "val_rmse": best_rmse,
                "pool_size": self.pool.size, "all_metrics": all_metrics}

    # ------------------------------------------------------------------
    # Step 3: Train Q-learning agent (resumable across runs)
    # ------------------------------------------------------------------

    def train_qagent(
        self,
        n_episodes: int,
        seed: int = 0,
        n_train_specs: int = 20,
    ) -> Dict:
        """Train the Q-learning agent for ``n_episodes``, resuming from the
        latest checkpoint if one exists.

        The epsilon schedule is adjusted based on total episodes already
        completed so exploration stays appropriate as the agent matures.

        Returns
        -------
        Dict with keys: total_episodes, episodes_this_run, eps_start.
        """
        q_table, q_meta      = load_qtable(self.ckpt_dir)
        episodes_so_far      = int(q_meta.get("total_episodes", 0))

        fp, _                = load_predictor(self.ckpt_dir)

        # Build a fresh set of random training specs for this run.
        spec_rng   = np.random.default_rng(seed + episodes_so_far)
        min_procs  = max(5, self.n_processes // 2)
        train_specs = [
            random_workload_spec(spec_rng, min_procs, self.n_processes)
            for _ in range(n_train_specs)
        ]

        def predicted_lookup(spec, ep_seed):
            procs = generate_workload(spec, ep_seed)
            if fp is not None:
                pred_arr  = fp.predict(procs)
                predicted = {p.pid: float(pred_arr[i])
                              for i, p in enumerate(procs)}
            else:
                predicted = {p.pid: float(p.true_burst) for p in procs}
            return procs, predicted

        # Decay exploration: start high when fresh, low after many episodes.
        eps_start = max(0.05, 1.0 * (0.998 ** episodes_so_far))

        agent = QLearningAgent(
            seed      = seed,
            q_table   = q_table,          # None -> zeros (first run)
            eps_start = eps_start,
            eps_end   = 0.02,
            eps_decay_episodes = max(1, n_episodes // 2),
        )

        rng_ep = random.Random(seed + episodes_so_far * 7)
        for ep in range(n_episodes):
            agent.episode = ep
            spec       = rng_ep.choice(train_specs)
            ep_seed    = rng_ep.randrange(1_000_000)
            procs, predicted = predicted_lookup(spec, ep_seed)

            def cb(state, a_idx, reward, next_state, done):
                s  = discretize_state(**state)
                ns = (None if (done or next_state is None)
                      else discretize_state(**next_state))
                agent.update(s, a_idx, reward, ns)

            run_round_robin_family(procs, agent, predicted, step_callback=cb)

        total = episodes_so_far + n_episodes
        meta  = {
            "total_episodes":    total,
            "episodes_this_run": n_episodes,
            "eps_start":         eps_start,
            "seed":              seed,
        }
        save_qtable(agent.q_table, meta, self.ckpt_dir)
        return meta

    # ------------------------------------------------------------------
    # Step 4: Frozen-test-set evaluation
    # ------------------------------------------------------------------

    def evaluate(self, n_seeds: int = _EVAL_SEEDS_PER_CAT) -> Dict:
        """Evaluate the checkpointed predictor and Q-agent on standard test workloads.

        Uses the same workload categories as the paper but a small, fixed
        seed count to keep evaluation fast (full evaluation is ``run_experiment.py``).

        Returns
        -------
        Dict with per-algorithm mean waiting times and predictor test RMSE.
        """
        from .qlearning import FrozenQPolicy

        fp, pred_meta   = load_predictor(self.ckpt_dir)
        q_table, q_meta = load_qtable(self.ckpt_dir)

        test_cats = (
            in_distribution_categories(self.n_processes)
            + shifted_test_only_categories(self.n_processes)
        )

        waits: Dict[str, list] = {"A": [], "D": [], "F": []}
        rmse_vals: list        = []

        for spec in test_cats:
            for s in range(n_seeds):
                seed  = SEED_OFFSETS["test"] + s
                procs = generate_workload(spec, seed)

                if fp is not None:
                    pred_arr  = fp.predict(procs)
                    predicted = {p.pid: float(pred_arr[i])
                                  for i, p in enumerate(procs)}
                    y_true    = np.array([p.true_burst for p in procs], float)
                    rmse_vals.append(
                        float(np.sqrt(np.mean((y_true - pred_arr) ** 2))))
                else:
                    predicted = {p.pid: float(p.true_burst) for p in procs}

                log_a = run_priority_family(
                    procs, AlgorithmA_Priority(), predicted)
                waits["A"].append(compute_metrics(procs, log_a).avg_waiting)

                log_d = run_round_robin_family(
                    procs, FixedQuantumPolicy(q=16), predicted)
                waits["D"].append(compute_metrics(procs, log_d).avg_waiting)

                if q_table is not None:
                    pol_f = FrozenQPolicy(q_table=q_table)
                    log_f = run_round_robin_family(procs, pol_f, predicted)
                    waits["F"].append(compute_metrics(procs, log_f).avg_waiting)

        result = {
            alg: float(np.mean(v)) for alg, v in waits.items() if v
        }
        result["predictor_test_rmse"] = (
            float(np.mean(rmse_vals)) if rmse_vals else None)
        result["predictor_name"]      = pred_meta.get("best_model", "none")
        result["q_total_episodes"]    = int(q_meta.get("total_episodes", 0))
        return result

    # ------------------------------------------------------------------
    # High-level: one complete training round
    # ------------------------------------------------------------------

    def run_round(
        self,
        round_num:    int,
        n_workloads:  int = 50,
        q_episodes:   int = 500,
        seed: Optional[int] = None,
    ) -> RoundResult:
        """Execute one complete training round.

        Steps: generate → pool → train predictors → train Q-agent → evaluate.
        Prints progress to stdout and appends a summary to the training log.

        Parameters
        ----------
        round_num:   Integer label for this round (used only for display/logging).
        n_workloads: Number of random workloads to generate in this round.
        q_episodes:  Number of additional Q-learning episodes to run.
        seed:        Master seed; defaults to a deterministic function of round_num.
        """
        if seed is None:
            seed = round_num * 9_973  # prime multiplier -> good spread

        print(f"\n[round {round_num}] "
              f"Generating {n_workloads} workloads (seed={seed})...")
        added = self.generate_and_pool(n_workloads, seed=seed)
        print(f"[round {round_num}] +{added:,} rows.  {self.pool}")

        print(f"[round {round_num}] Training predictors on {self.pool.size:,} rows...")
        pred = self.train_predictors()
        print(f"[round {round_num}] Best predictor: {pred['best_model']}  "
              f"val_RMSE={pred['val_rmse']:.3f}")

        print(f"[round {round_num}] Training Q-agent ({q_episodes} episodes)...")
        qm = self.train_qagent(n_episodes=q_episodes, seed=seed)
        print(f"[round {round_num}] Q-agent total episodes: {qm['total_episodes']:,}")

        print(f"[round {round_num}] Evaluating on frozen test set...")
        ev = self.evaluate()
        print(
            f"[round {round_num}] A={ev.get('A', '?'):.1f}  "
            f"D={ev.get('D', '?'):.1f}  "
            f"F={ev.get('F', 'n/a')}  "
            f"pred_RMSE={ev.get('predictor_test_rmse', '?')}"
        )

        _append_log(self.ckpt_dir, "round_complete", {
            "round":       round_num,
            "pool_size":   self.pool.size,
            "best_model":  pred["best_model"],
            "val_rmse":    pred["val_rmse"],
            "total_episodes": qm["total_episodes"],
            "eval_A":      ev.get("A"),
            "eval_D":      ev.get("D"),
            "eval_F":      ev.get("F"),
            "test_rmse":   ev.get("predictor_test_rmse"),
        })

        return RoundResult(
            round_num             = round_num,
            pool_size             = self.pool.size,
            workloads_generated   = n_workloads,
            predictor_name        = pred["best_model"],
            predictor_val_rmse    = pred["val_rmse"],
            predictor_test_rmse   = ev.get("predictor_test_rmse", float("nan")),
            q_episodes_this_round = q_episodes,
            q_episodes_total      = qm["total_episodes"],
            eval_alg_A_avg_wait   = ev.get("A", float("nan")),
            eval_alg_D_avg_wait   = ev.get("D", float("nan")),
            eval_alg_F_avg_wait   = ev.get("F"),
        )

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Return a human-readable status summary."""
        lines = ["=" * 50, "schedsim Continuous Training Suite", "=" * 50]
        lines.append(str(self.pool))

        _, pm = load_predictor(self.ckpt_dir)
        if pm:
            lines.append(
                f"Predictor : {pm.get('best_model', '?')}  "
                f"val_RMSE={pm.get('val_rmse', '?'):.3f}  "
                f"trained on {pm.get('pool_size', '?'):,} rows"
            )
        else:
            lines.append("Predictor : not trained yet")

        _, qm = load_qtable(self.ckpt_dir)
        if qm:
            lines.append(f"Q-agent   : {qm.get('total_episodes', 0):,} episodes total")
        else:
            lines.append("Q-agent   : not trained yet")

        lines.append("=" * 50)
        return "\n".join(lines)
