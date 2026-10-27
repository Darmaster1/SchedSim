#!/usr/bin/env python3
"""
Continuous training CLI: a data flywheel that repeatedly (1) grows the
training pool with freshly sampled synthetic workloads, (2) refits every
burst-prediction model on the accumulated pool and checkpoints the best one,
and (3) resumes Q-learning training for Algorithm F from its last saved
Q-table, so that more wall-clock time spent running this script keeps
producing a better predictor and a better-trained agent rather than redoing
the same fixed-size batch every time.

Usage:
    python train.py run --rounds 5 --workloads 50
    python train.py run --rounds 20 --workloads 100 --episodes-per-round 2000
    python train.py status
"""
from __future__ import annotations

import argparse
import json
import os
import time

from schedsim.checkpoint import read_leaderboard, refit_all_models
from schedsim.pool import append_to_pool, load_pool, pool_size
from schedsim.qlearning import QLearningAgent, train_qlearning_agent
from schedsim.workload import WorkloadSpec, generate_workload, in_distribution_categories

CHECKPOINT_DIR_DEFAULT = "checkpoints"
QTABLE_PATH_DEFAULT = os.path.join(CHECKPOINT_DIR_DEFAULT, "qtable_latest.npz")
POOL_PATH_DEFAULT = "training_pool.csv"


def _fixed_validation_set(n_processes: int = 24, seed_base: int = 2_200_000):
    """A small, fixed validation set (disjoint from both the pool's seed
    range and the main evaluation splits) used consistently across rounds so
    leaderboard RMSE is comparable round over round."""
    specs = in_distribution_categories(n_processes)
    procs = []
    for i, spec in enumerate(specs):
        procs.extend(generate_workload(spec, seed_base + i))
    return procs


def cmd_run(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    val_procs = _fixed_validation_set(n_processes=args.n_processes)
    qtable_path = os.path.join(args.checkpoint_dir, "qtable_latest.npz")

    cumulative_episodes = 0
    initial_q_table = None
    if os.path.isfile(qtable_path):
        agent, cumulative_episodes = QLearningAgent.load(qtable_path)
        initial_q_table = agent.q_table
        print(f"[train] resumed Q-table from {qtable_path} "
              f"(cumulative_episodes={cumulative_episodes})")
    else:
        print("[train] no existing Q-table found, starting from scratch")

    train_specs = in_distribution_categories(args.n_processes)

    for round_idx in range(args.rounds):
        t0 = time.time()
        print(f"\n=== Round {round_idx + 1}/{args.rounds} ===")

        n_rows = append_to_pool(args.workloads, pool_path=args.pool_path,
                                 seed=args.seed * 100_000 + round_idx, n_processes=args.n_processes)
        pool_procs = load_pool(args.pool_path)
        print(f"[train] pool grew by {n_rows} rows -> {len(pool_procs)} total processes")

        best_name, best_fp, all_metrics = refit_all_models(
            pool_procs, val_procs, checkpoint_dir=args.checkpoint_dir, round_idx=round_idx)
        print(f"[train] best predictor this round: {best_name} "
              f"(val RMSE={all_metrics[best_name]['rmse']:.3f})")

        def predicted_lookup(spec: WorkloadSpec, seed: int):
            procs = generate_workload(spec, seed)
            preds = best_fp.predict(procs)
            return procs, {p.pid: float(v) for p, v in zip(procs, preds)}

        agent, history = train_qlearning_agent(
            train_specs, predicted_lookup,
            n_episodes=args.episodes_per_round,
            seed=args.seed,
            initial_q_table=initial_q_table,
            cumulative_episode_offset=cumulative_episodes,
            eps_decay_episodes=args.eps_decay_episodes,
            eval_every=max(1, args.episodes_per_round // 5),
            progress_every=0,
        )
        initial_q_table = agent.q_table
        cumulative_episodes += args.episodes_per_round
        agent.save(qtable_path, cumulative_episodes=cumulative_episodes)

        last_eval = history[-1]["mean_avg_waiting"] if history else float("nan")
        print(f"[train] Q-learning: +{args.episodes_per_round} episodes "
              f"(cumulative={cumulative_episodes}), last eval avg_waiting={last_eval:.2f}, "
              f"saved -> {qtable_path}")
        print(f"[train] round {round_idx + 1} took {time.time() - t0:.1f}s")

    print("\n[train] done. Leaderboard tail:")
    board = read_leaderboard(args.checkpoint_dir)
    for row in board[-len(all_metrics):]:
        print(f"  round={row['round']:>3} model={row['model']:<14} "
              f"val_rmse={float(row['val_rmse']):.3f} best={row['is_best_this_round']}")


def cmd_status(args):
    n = pool_size(args.pool_path)
    print(f"Training pool: {n} process-rows at {args.pool_path}")
    board = read_leaderboard(args.checkpoint_dir)
    if board:
        last_round = board[-1]["round"]
        print(f"Leaderboard: {len(board)} rows, last round = {last_round}")
    else:
        print("Leaderboard: empty (no training rounds run yet)")
    qtable_path = os.path.join(args.checkpoint_dir, "qtable_latest.npz")
    if os.path.isfile(qtable_path):
        _, cumulative = QLearningAgent.load(qtable_path)
        print(f"Q-table checkpoint: {qtable_path} (cumulative_episodes={cumulative})")
    else:
        print("Q-table checkpoint: none yet")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run one or more continuous-training rounds")
    run_p.add_argument("--rounds", type=int, default=5)
    run_p.add_argument("--workloads", type=int, default=50, help="new workloads sampled into the pool per round")
    run_p.add_argument("--episodes-per-round", type=int, default=1000)
    run_p.add_argument("--eps-decay-episodes", type=int, default=5000)
    run_p.add_argument("--n-processes", type=int, default=24)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--pool-path", default=POOL_PATH_DEFAULT)
    run_p.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR_DEFAULT)
    run_p.set_defaults(func=cmd_run)

    status_p = sub.add_parser("status", help="show current pool size, leaderboard, and Q-table checkpoint status")
    status_p.add_argument("--pool-path", default=POOL_PATH_DEFAULT)
    status_p.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR_DEFAULT)
    status_p.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
