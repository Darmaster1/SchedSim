#!/usr/bin/env python3
"""
train.py -- Continuous training CLI for schedsim.

This script manages the data-flywheel for ML burst-length predictors and the
resumable Q-learning training loop for Algorithm F.  It is entirely separate
from ``run_experiment.py``, which reproduces the paper's pilot-scale numbers
with a fixed protocol.  ``train.py`` is for *improving* the models beyond
the pilot -- it generates new random workloads, accumulates them in a
persistent pool, and retrains on the growing dataset.

Usage
-----
One training round (generates data + retrains + evaluates)::

    python train.py run

Multiple rounds::

    python train.py run --rounds 10

Scale up each round::

    python train.py run --rounds 5 --workloads 100 --q-episodes 1000

Check current state::

    python train.py status

Show leaderboard (best predictor checkpoints by val RMSE)::

    python train.py leaderboard

Evaluate current checkpoint on the frozen test set::

    python train.py evaluate

Generate data only (no model retraining)::

    python train.py generate --n 200

Retrain predictors only (on existing pool)::

    python train.py train-predictor

Train Q-agent only::

    python train.py train-qagent --episodes 2000

Wipe everything and start fresh (asks for confirmation)::

    python train.py reset

Design notes
------------
* Pool seeds live in [3_000_000, 4_000_000), disjoint from the train/val/test
  seed ranges used by run_experiment.py so there is zero data leakage.
* Q-table training resumes from the latest checkpoint; epsilon is decayed
  according to total episodes seen so far.
* The frozen evaluation set uses the paper's standard 13 test categories
  (10 in-distribution + 3 shifted), each with 3 seeds, giving 39 test
  workloads -- a quick but meaningful quality signal.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from schedsim.checkpoint import leaderboard, read_training_log
from schedsim.continuous import ContinuousTrainer

# ---------------------------------------------------------------------------
# Paths (relative to repo root, which is this file's directory)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR  = os.path.join(_REPO_ROOT, "data")
_CKPT_DIR  = os.path.join(_REPO_ROOT, "checkpoints")


def _make_trainer(args) -> ContinuousTrainer:
    return ContinuousTrainer(
        data_dir     = _DATA_DIR,
        ckpt_dir     = _CKPT_DIR,
        n_processes  = args.n_processes,
    )


# ---------------------------------------------------------------------------
# Sub-command: run
# ---------------------------------------------------------------------------

def cmd_run(args):
    trainer = _make_trainer(args)
    print(trainer.status())

    for i in range(1, args.rounds + 1):
        r = trainer.run_round(
            round_num   = i,
            n_workloads = args.workloads,
            q_episodes  = args.q_episodes,
        )
        print(
            f"\n--- Round {r.round_num} summary ---\n"
            f"  Pool size     : {r.pool_size:,} rows "
            f"(+{r.workloads_generated * args.n_processes} this round)\n"
            f"  Best model    : {r.predictor_name}\n"
            f"  val RMSE      : {r.predictor_val_rmse:.4f}\n"
            f"  test RMSE     : {r.predictor_test_rmse:.4f}\n"
            f"  Q-episodes    : {r.q_episodes_total:,} total\n"
            f"  Eval A        : {r.eval_alg_A_avg_wait:.2f}\n"
            f"  Eval D        : {r.eval_alg_D_avg_wait:.2f}\n"
            f"  Eval F        : {r.eval_alg_F_avg_wait}\n"
        )

    print("Done.  Run `python train.py leaderboard` to see version history.")


# ---------------------------------------------------------------------------
# Sub-command: status
# ---------------------------------------------------------------------------

def cmd_status(args):
    trainer = _make_trainer(args)
    print(trainer.status())


# ---------------------------------------------------------------------------
# Sub-command: leaderboard
# ---------------------------------------------------------------------------

def cmd_leaderboard(args):
    board = leaderboard(_CKPT_DIR, metric="val_rmse")
    if not board:
        print("No predictor checkpoints found. Run `python train.py run` first.")
        return

    print(f"{'Rank':>4}  {'val_RMSE':>10}  {'model':>15}  {'pool_size':>12}  {'saved_at'}")
    print("-" * 70)
    for rank, entry in enumerate(board, start=1):
        print(
            f"{rank:>4}  "
            f"{entry.get('val_rmse', float('nan')):>10.4f}  "
            f"{entry.get('best_model', '?'):>15}  "
            f"{entry.get('pool_size', '?'):>12}  "
            f"{entry.get('ts', '?')}"
        )


# ---------------------------------------------------------------------------
# Sub-command: evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args):
    trainer = _make_trainer(args)
    print("Evaluating current checkpoint on the frozen test set...")
    ev = trainer.evaluate(n_seeds=5)
    print("\nResults (mean over 5 seeds x 13 test categories = 65 workloads):")
    for k, v in ev.items():
        if isinstance(v, float):
            print(f"  {k:<30} {v:.4f}")
        else:
            print(f"  {k:<30} {v}")


# ---------------------------------------------------------------------------
# Sub-command: generate
# ---------------------------------------------------------------------------

def cmd_generate(args):
    trainer = _make_trainer(args)
    print(f"Generating {args.n} random workloads (seed={args.seed})...")
    added = trainer.generate_and_pool(n_workloads=args.n, seed=args.seed)
    print(f"Added {added:,} rows. {trainer.pool}")


# ---------------------------------------------------------------------------
# Sub-command: train-predictor
# ---------------------------------------------------------------------------

def cmd_train_predictor(args):
    trainer = _make_trainer(args)
    if trainer.pool.size == 0:
        print("Pool is empty. Run `python train.py generate` first.")
        sys.exit(1)
    print(f"Training predictors on {trainer.pool.size:,} rows...")
    meta = trainer.train_predictors()
    print(f"Best: {meta['best_model']}  val_RMSE={meta['val_rmse']:.4f}")
    print("Checkpoint saved.")


# ---------------------------------------------------------------------------
# Sub-command: train-qagent
# ---------------------------------------------------------------------------

def cmd_train_qagent(args):
    trainer = _make_trainer(args)
    print(f"Training Q-agent for {args.episodes} episodes (resumable)...")
    meta = trainer.train_qagent(
        n_episodes = args.episodes,
        seed       = args.seed,
    )
    print(
        f"Done. Total episodes: {meta['total_episodes']:,}  "
        f"eps_start_this_run={meta['eps_start']:.4f}"
    )


# ---------------------------------------------------------------------------
# Sub-command: reset
# ---------------------------------------------------------------------------

def cmd_reset(args):
    target_dirs = []
    if args.pool_only:
        target_dirs = [_DATA_DIR]
    elif args.models_only:
        target_dirs = [_CKPT_DIR]
    else:
        target_dirs = [_DATA_DIR, _CKPT_DIR]

    if not args.force:
        targets_str = " and ".join([os.path.basename(d) for d in target_dirs])
        print(f"WARNING: This will permanently delete: {targets_str}")
        ans = input("Type 'yes' to confirm: ").strip().lower()
        if ans != "yes":
            print("Aborted.")
            return

    for d in target_dirs:
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"Deleted {d}")
        else:
            print(f"Directory {d} does not exist (already clean).")
    print("Reset complete. You can start fresh.")


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n-processes", type=int, default=30,
        help="Max process count for generated workloads (default: 30).",
    )

    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # run
    p_run = sub.add_parser("run", help="Execute N training rounds.")
    p_run.add_argument("--rounds",     type=int, default=1,
                        help="Number of rounds to run (default: 1).")
    p_run.add_argument("--workloads",  type=int, default=50,
                        help="Random workloads per round (default: 50).")
    p_run.add_argument("--q-episodes", type=int, default=500,
                        help="Q-learning episodes per round (default: 500).")
    p_run.set_defaults(func=cmd_run)

    # status
    p_status = sub.add_parser("status", help="Show pool and checkpoint status.")
    p_status.set_defaults(func=cmd_status)

    # leaderboard
    p_lb = sub.add_parser("leaderboard",
                           help="Show predictor version history ranked by val RMSE.")
    p_lb.set_defaults(func=cmd_leaderboard)

    # evaluate
    p_ev = sub.add_parser("evaluate",
                           help="Evaluate current checkpoint on frozen test set.")
    p_ev.set_defaults(func=cmd_evaluate)

    # generate
    p_gen = sub.add_parser("generate",
                            help="Generate and pool random workloads only.")
    p_gen.add_argument("--n",    type=int, default=50,
                        help="Number of workloads to generate (default: 50).")
    p_gen.add_argument("--seed", type=int, default=42,
                        help="Master seed for generation (default: 42).")
    p_gen.set_defaults(func=cmd_generate)

    # train-predictor
    p_tp = sub.add_parser("train-predictor",
                           help="Retrain ML predictors on the existing pool.")
    p_tp.set_defaults(func=cmd_train_predictor)

    # train-qagent
    p_tq = sub.add_parser("train-qagent",
                           help="Train Q-learning agent (resumable).")
    p_tq.add_argument("--episodes", type=int, default=500,
                       help="Number of Q-learning episodes (default: 500).")
    p_tq.add_argument("--seed",     type=int, default=0,
                       help="Training seed (default: 0).")
    p_tq.set_defaults(func=cmd_train_qagent)

    # reset
    p_reset = sub.add_parser("reset",
                              help="Delete pool and checkpoints (irreversible).")
    p_reset.add_argument("-f", "--force", action="store_true",
                         help="Bypass interactive confirmation prompt.")
    p_reset.add_argument("--pool-only", action="store_true",
                         help="Only reset the training data pool (keeps model checkpoints).")
    p_reset.add_argument("--models-only", action="store_true",
                         help="Only reset model checkpoints and log (keeps training data pool).")
    p_reset.set_defaults(func=cmd_reset)

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
