import numpy as np
import pytest

from schedsim.metrics import compute_metrics
from schedsim.qlearning import (FrozenQPolicy, N_STATES, discretize_state,
                                 train_qlearning_agent)
from schedsim.schedulers import run_round_robin_family
from schedsim.workload import WorkloadSpec, generate_workload


def test_state_discretization_always_in_range():
    rng = np.random.default_rng(0)
    for _ in range(500):
        q = rng.integers(0, 30)
        mp = rng.uniform(0, 50)
        mw = rng.uniform(0, 200)
        imb = rng.uniform(0, 6)
        s = discretize_state(q, mp, mw, imb)
        assert 0 <= s < N_STATES


def test_state_discretization_boundary_goes_to_lower_class():
    # A value exactly AT a bin boundary is assigned to the class that
    # boundary starts (i.e. treated as ">=", per the paper's convention),
    # so it groups with values just above the boundary, not just below it.
    s_at_bound = discretize_state(4, 0, 0, 0)
    s_below = discretize_state(3.999, 0, 0, 0)
    s_above = discretize_state(4.001, 0, 0, 0)
    assert s_at_bound != s_below
    assert s_at_bound == s_above


def test_frozen_policy_is_deterministic_and_ties_favor_smaller_quantum():
    q_table = np.zeros((N_STATES, 4))  # all-zero -> every action tied
    policy = FrozenQPolicy(q_table)
    idx = policy.choose_quantum_idx(queue_len=3, mean_pred_remaining=5, max_wait=10, imbalance=1.0,
                                     own_wait=0)
    assert idx == 0  # smallest quantum wins ties


def test_training_runs_and_produces_nontrivial_q_table():
    spec = WorkloadSpec("qtrain", n_processes=15)

    def lookup(spec, seed):
        procs = generate_workload(spec, seed)
        predicted = {p.pid: float(p.true_burst) for p in procs}
        return procs, predicted

    agent = train_qlearning_agent([spec], lookup, n_episodes=40, seed=0)
    assert agent.q_table.shape == (N_STATES, 4)
    assert np.any(agent.q_table != 0.0)


def test_frozen_eval_has_no_exploration_randomness():
    spec = WorkloadSpec("qeval", n_processes=15)

    def lookup(spec, seed):
        procs = generate_workload(spec, seed)
        predicted = {p.pid: float(p.true_burst) for p in procs}
        return procs, predicted

    agent = train_qlearning_agent([spec], lookup, n_episodes=30, seed=0)
    frozen = FrozenQPolicy(agent.q_table)

    procs, predicted = lookup(spec, 999)
    log1 = run_round_robin_family(procs, frozen, predicted)
    log2 = run_round_robin_family(procs, frozen, predicted)
    order1 = [e.pid for e in log1.events]
    order2 = [e.pid for e in log2.events]
    assert order1 == order2  # frozen policy must be fully deterministic
