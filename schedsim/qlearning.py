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
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schedulers import DEFAULT_REWARD_WEIGHTS, QUANTA, run_round_robin_family
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

#   r_t = -w1*dW/Wref - w2*Rfirst/Rref - w3*min(S-1,Smax)/Sref
#        - w4*1[switch] - w5*clip((Wmax-Wfair)/Wfair, 0, 1)          (Eq. 7)
WEIGHT_PRESETS = {
    "balanced": dict(w1=1.0, w2=0.5, w3=0.5, w4=0.2, w5=0.2),
    "wait_only": dict(w1=1.0, w2=0.0, w3=0.0, w4=0.0, w5=0.0),
    "context_switch_heavy": dict(w1=0.7, w2=0.3, w3=0.3, w4=0.8, w5=0.2),
    "fairness_heavy": dict(w1=0.6, w2=0.3, w3=0.3, w4=0.2, w5=0.8),
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

    # --- Resumable training: persist/restore the Q-table and cumulative
    # episode count (so epsilon decay continues across process runs) ------
    def save(self, path: str, cumulative_episodes: Optional[int] = None):
        np.savez(path, q_table=self.q_table,
                 cumulative_episodes=np.array(
                     cumulative_episodes if cumulative_episodes is not None else self.episode))

    @classmethod
    def load(cls, path: str, seed: int = 0, eps_decay_episodes: int = 200) -> Tuple["QLearningAgent", int]:
        data = np.load(path)
        agent = cls(seed=seed, eps_decay_episodes=eps_decay_episodes, q_table=data["q_table"].copy())
        cumulative_episodes = int(data["cumulative_episodes"])
        return agent, cumulative_episodes


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


def _eval_agent_mean_waiting(agent: "QLearningAgent", eval_specs: List[WorkloadSpec],
                              eval_seeds: List[int], predicted_lookup,
                              context_switch_cost: float) -> float:
    """Mean average-waiting-time of the agent's CURRENT greedy (frozen)
    policy over a small fixed set of (spec, seed) evaluation workloads --
    used only to trace a learning curve during training, never to update
    the agent itself."""
    from .metrics import compute_metrics  # local import avoids a cycle at module load
    frozen = FrozenQPolicy(agent.q_table)
    waits = []
    for spec in eval_specs:
        for seed in eval_seeds:
            procs, predicted = predicted_lookup(spec, seed)
            log = run_round_robin_family(procs, frozen, predicted, context_switch_cost)
            waits.append(compute_metrics(procs, log).avg_waiting)
    return float(np.mean(waits)) if waits else float("nan")


def train_qlearning_agent(train_specs: List[WorkloadSpec], predicted_lookup,
                           n_episodes: int = 400, seed: int = 0,
                           weight_preset: str = "balanced",
                           context_switch_cost: float = 1.0,
                           eval_every: int = 0,
                           eval_specs: Optional[List[WorkloadSpec]] = None,
                           eval_seeds: Optional[List[int]] = None,
                           progress_every: int = 0,
                           initial_q_table: Optional[np.ndarray] = None,
                           cumulative_episode_offset: int = 0,
                           eps_decay_episodes: Optional[int] = None) -> Tuple[QLearningAgent, List[dict]]:
    """Train one Q-learning agent over `n_episodes` sampled training workloads.

    `predicted_lookup(spec, seed) -> (procs, predicted_dict)` supplies the
    burst predictions each episode should use (out-of-fold predictions for
    training workloads, per Section 7.1).

    If `eval_every > 0`, the agent's current greedy policy is evaluated on
    `eval_specs` (defaults to `train_specs`) every `eval_every` episodes,
    producing a learning curve: a list of
    {"episode": int, "mean_avg_waiting": float, "epsilon": float} dicts,
    returned as the second element of the result tuple. This is the
    mechanism that lets training actually be watched "getting better" over
    many iterations, the way a normal ML training loop is monitored, rather
    than being a single opaque batch call.
    """
    decay = eps_decay_episodes if eps_decay_episodes is not None else max(1, int(n_episodes * 0.6))
    agent = QLearningAgent(seed=seed, eps_decay_episodes=decay,
                            q_table=initial_q_table.copy() if initial_q_table is not None else None)
    rng = random.Random(seed)
    eval_specs = eval_specs if eval_specs is not None else train_specs
    eval_seeds = eval_seeds if eval_seeds is not None else [987654]
    weights = WEIGHT_PRESETS.get(weight_preset, DEFAULT_REWARD_WEIGHTS)
    history: List[dict] = []

    for ep in range(n_episodes):
        agent.episode = cumulative_episode_offset + ep
        spec = rng.choice(train_specs)
        episode_seed = rng.randrange(1_000_000)
        procs, predicted = predicted_lookup(spec, episode_seed)

        def cb(state, a_idx, reward, next_state, done):
            s_idx = discretize_state(**state)
            ns_idx = None if (done or next_state is None) else discretize_state(**next_state)
            agent.update(s_idx, a_idx, reward, ns_idx)

        run_round_robin_family(procs, agent, predicted, context_switch_cost, step_callback=cb,
                                reward_weights=weights)

        if eval_every and (ep + 1) % eval_every == 0:
            score = _eval_agent_mean_waiting(agent, eval_specs, eval_seeds, predicted_lookup,
                                              context_switch_cost)
            history.append({"episode": ep + 1, "mean_avg_waiting": score,
                             "epsilon": agent.epsilon()})
        if progress_every and (ep + 1) % progress_every == 0:
            last = history[-1]["mean_avg_waiting"] if history else float("nan")
            print(f"    [qlearning] episode {ep+1}/{n_episodes}  "
                  f"eps={agent.epsilon():.3f}  last_eval_avg_waiting={last:.2f}")

    return agent, history
