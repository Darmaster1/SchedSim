"""
Tabular Q-learning for adaptive Round Robin quantum selection (Algorithm F,
Section 8 of the paper).

State: 4 (queue length) x 4 (mean predicted remaining burst) x 3 (max wait)
       x 3 (imbalance) = 144 discrete states.
Action: index into QUANTA = (2, 4, 8, 16).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .schedulers import QUANTA, run_round_robin_family
from .workload import Process, WorkloadSpec, generate_workload

N_ACTIONS = len(QUANTA)

_QUEUE_BOUNDS = (1, 4, 9)
_BURST_BOUNDS = (4, 12, 30)
_WAIT_BOUNDS = (25, 100)
_IMBALANCE_BOUNDS = (1.5, 3.0)


def _bin_value(value: float, bounds: Tuple[float, ...]) -> int:
    cls = 0
    for b in bounds:
        if value >= b:
            cls += 1
    return cls


def discretize_state(queue_len: int, mean_pred_remaining: float, max_wait: float,
                      imbalance: float, own_wait: float = 0.0) -> int:
    q_bin = _bin_value(queue_len, _QUEUE_BOUNDS)        # 0..3  (4 classes)
    b_bin = _bin_value(mean_pred_remaining, _BURST_BOUNDS)  # 0..3 (4 classes)
    w_bin = _bin_value(max_wait, _WAIT_BOUNDS)          # 0..2  (3 classes)
    i_bin = _bin_value(imbalance, _IMBALANCE_BOUNDS)    # 0..2  (3 classes)
    return ((q_bin * 4 + b_bin) * 3 + w_bin) * 3 + i_bin  # 0..143


N_STATES = 4 * 4 * 3 * 3  # 144

WEIGHT_PRESETS = {
    "balanced": dict(w_wait=1.0, w_response=0.1, w_slowdown=0.1, w_tailwait=0.2),
    "wait_only": dict(w_wait=1.0, w_response=0.0, w_slowdown=0.0, w_tailwait=0.0),
    "context_switch_heavy": dict(w_wait=0.7, w_response=0.1, w_slowdown=0.1, w_tailwait=0.2),
    "fairness_heavy": dict(w_wait=0.6, w_response=0.1, w_slowdown=0.1, w_tailwait=0.6),
}


@dataclass
class QLearningAgent:
    gamma: float = 0.9
    lr: float = 0.15
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 200
    seed: int = 0
    q_table: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.q_table is None:
            self.q_table = np.zeros((N_STATES, N_ACTIONS))
        self._rng = random.Random(self.seed)
        self.episode = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.episode / max(self.eps_decay_episodes, 1))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state_idx: int, explore: bool) -> int:
        if explore and self._rng.random() < self.epsilon():
            return self._rng.randrange(N_ACTIONS)
        return int(np.argmax(self.q_table[state_idx]))

    def update(self, s: int, a: int, r: float, s_next: int | None):
        target = r if s_next is None else r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.lr * (target - self.q_table[s, a])

    # --- QuantumPolicy protocol, used directly by run_round_robin_family ---
    def choose_quantum_idx(self, queue_len, mean_pred_remaining, max_wait, imbalance, own_wait):
        s = discretize_state(queue_len, mean_pred_remaining, max_wait, imbalance, own_wait)
        return self.select_action(s, explore=True)


@dataclass
class FrozenQPolicy:
    """Wraps a trained Q-table for evaluation: no exploration, no updates."""
    q_table: np.ndarray

    def choose_quantum_idx(self, queue_len, mean_pred_remaining, max_wait, imbalance, own_wait):
        s = discretize_state(queue_len, mean_pred_remaining, max_wait, imbalance, own_wait)
        # Ties -> smaller quantum (lower index), as specified.
        row = self.q_table[s]
        best = np.flatnonzero(row == row.max())
        return int(best[0])


def train_qlearning_agent(train_specs: List[WorkloadSpec], predicted_lookup,
                           n_episodes: int = 400, seed: int = 0,
                           weight_preset: str = "balanced",
                           context_switch_cost: float = 1.0) -> QLearningAgent:
    """Train one Q-learning agent over `n_episodes` sampled training workloads.

    `predicted_lookup(spec, seed) -> (procs, predicted_dict)` supplies the
    burst predictions each episode should use (out-of-fold predictions for
    training workloads, per Section 7.1).
    """
    agent = QLearningAgent(seed=seed, eps_decay_episodes=max(1, n_episodes // 2))
    rng = random.Random(seed)

    for ep in range(n_episodes):
        agent.episode = ep
        spec = rng.choice(train_specs)
        episode_seed = rng.randrange(1_000_000)
        procs, predicted = predicted_lookup(spec, episode_seed)

        def cb(state, a_idx, reward, next_state, done):
            s_idx = discretize_state(**state)
            ns_idx = None if (done or next_state is None) else discretize_state(**next_state)
            agent.update(s_idx, a_idx, reward, ns_idx)

        run_round_robin_family(procs, agent, predicted, context_switch_cost, step_callback=cb)

    return agent
