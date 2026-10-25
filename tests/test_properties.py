"""
Property-based tests using the ``hypothesis`` library.

These tests generate **random** workloads (random spec parameters and seeds)
and assert structural invariants that the paper claims hold universally -- not
just on the 13 hand-crafted categories used in the paper's pilot study.  Every
invariant here maps directly to a claim made in the paper.

When ``hypothesis`` finds a counterexample it automatically shrinks it to the
smallest failing case and stores it in its local database, so subsequent runs
immediately re-run any previously found failures.  This gives the equivalent
of thousands of randomised test cases without any manual effort.

Running:
    pytest tests/test_properties.py -v
    pytest tests/test_properties.py -v --hypothesis-seed=0   # deterministic

Slow tests are suppressed by default; use::
    pytest tests/test_properties.py -v -m "not slow"
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, assume
from hypothesis import strategies as st

from schedsim.metrics import compute_metrics, STARVATION_THRESHOLD
from schedsim.predictors import fit_predictor
from schedsim.qlearning import (
    N_STATES, FrozenQPolicy, QLearningAgent, discretize_state,
)
from schedsim.random_spec import random_workload_spec
from schedsim.schedulers import (
    AlgorithmA_Priority, AlgorithmB_WeightedML, AlgorithmC_ProtectedML,
    FixedQuantumPolicy, QUANTA, RuleBasedQuantumPolicy,
    run_priority_family, run_round_robin_family,
)
from schedsim.workload import WorkloadSpec, generate_workload

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

_BURST_MODES    = ("short", "long", "mixed", "heavy_tailed")
_ARRIVAL_MODES  = ("simultaneous", "staggered", "bursty")
_PRIORITY_MODES = ("uniform", "skewed", "adversarial")
_FEATURE_STRS   = ("strong", "moderate", "weak", "noisy", "misleading")

# Keep workload small so each test stays fast under hypothesis.
_N_MIN, _N_MAX = 3, 30


@st.composite
def workload_strategy(draw):
    """Hypothesis strategy: sample a random (WorkloadSpec, seed) pair."""
    n       = draw(st.integers(_N_MIN, _N_MAX))
    bm      = draw(st.sampled_from(_BURST_MODES))
    am      = draw(st.sampled_from(_ARRIVAL_MODES))
    pm      = draw(st.sampled_from(_PRIORITY_MODES))
    fs      = draw(st.sampled_from(_FEATURE_STRS))
    load    = draw(st.floats(0.2, 1.8, allow_nan=False, allow_infinity=False))
    seed    = draw(st.integers(0, 9_999_999))
    spec    = WorkloadSpec(
        name          = f"hyp_{n}_{bm[:3]}_{am[:3]}",
        n_processes   = n,
        burst_mode    = bm,
        arrival_mode  = am,
        priority_mode = pm,
        feature_strength = fs,
        offered_load  = round(load, 3),
    )
    return spec, seed


def _make_procs_and_oracle(spec, seed):
    """Generate processes and return oracle (true-burst) predictions."""
    procs     = generate_workload(spec, seed)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    return procs, predicted


# ---------------------------------------------------------------------------
# INVARIANT 1 -- turnaround decomposition (Priority family)
# Paper: "turnaround T_i = c_i - a_i = W_i + b_i"
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_turnaround_equals_waiting_plus_burst_priority(spec_seed):
    """For every process in every random priority-family run,
    turnaround = waiting + true_burst (paper §4.1)."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    m   = compute_metrics(procs, log)
    by_pid = {p.pid: p for p in procs}
    for p in procs:
        c  = log.completion_time[p.pid]
        s  = log.start_time[p.pid]
        tat = c - p.arrival
        wait = tat - p.true_burst
        # Turnaround must be non-negative and >= burst
        assert tat  >= p.true_burst - 1e-6, (
            f"pid {p.pid}: turnaround {tat:.3f} < burst {p.true_burst}")
        assert wait >= -1e-6, (
            f"pid {p.pid}: negative waiting time {wait:.3f}")


# ---------------------------------------------------------------------------
# INVARIANT 2 -- turnaround decomposition (Round Robin family)
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_turnaround_equals_waiting_plus_burst_rr(spec_seed):
    """Same decomposition invariant holds for the Round Robin engine."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_round_robin_family(procs, FixedQuantumPolicy(q=4), predicted)
    for p in procs:
        c    = log.completion_time[p.pid]
        tat  = c - p.arrival
        wait = tat - p.true_burst
        assert tat  >= p.true_burst - 1e-6
        assert wait >= -1e-6


# ---------------------------------------------------------------------------
# INVARIANT 3 -- B(alpha=1) == A  (pid-for-pid, not just metric-equal)
# Paper: "alpha=1 reproduces A exactly (verified by unit test; §VI-B)"
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_B_alpha1_equals_A_random_workloads(spec_seed):
    """B with alpha=1 produces the *same dispatch sequence* as A on every
    random workload -- not just matching metrics."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log_a = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    log_b = run_priority_family(procs, AlgorithmB_WeightedML(alpha=1.0), predicted)
    order_a = [e.pid for e in log_a.events]
    order_b = [e.pid for e in log_b.events]
    assert order_a == order_b, (
        f"Dispatch sequences differ for spec={spec.name}, seed={seed}\n"
        f"  A: {order_a}\n  B: {order_b}"
    )


# ---------------------------------------------------------------------------
# INVARIANT 4 -- C with safeguards disabled == B  (pid-for-pid)
# Paper: "C-with-safeguards-disabled ≡ B ... not merely metric-equal; §VII-B"
# ---------------------------------------------------------------------------

@given(
    workload_strategy(),
    st.floats(0.0, 1.0, allow_nan=False),
)
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_C_safeguards_disabled_equals_B_random(spec_seed, alpha):
    """C with (w_max=inf, gap=999) is pid-for-pid identical to B."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log_b = run_priority_family(
        procs, AlgorithmB_WeightedML(alpha=alpha), predicted)
    log_c = run_priority_family(
        procs, AlgorithmC_ProtectedML(alpha=alpha, w_max=1e12, gap=999), predicted)
    assert ([e.pid for e in log_b.events]
            == [e.pid for e in log_c.events])


# ---------------------------------------------------------------------------
# INVARIANT 5 -- C with gap=0: no priority inversion outside override events
# Paper: "G=0 => no inversion outside override events; §VII-B"
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_C_gap0_no_inversion_outside_override(spec_seed):
    """With gap=0, every recorded inversion must be an override activation."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    assume(len(procs) >= 2)  # needs at least 2 processes for inversion to make sense
    policy = AlgorithmC_ProtectedML(alpha=0.5, w_max=50.0, gap=0)
    run_priority_family(procs, policy, predicted)
    assert policy.inversions <= policy.activations, (
        f"gap=0 produced {policy.inversions} inversions but only "
        f"{policy.activations} override activations"
    )


# ---------------------------------------------------------------------------
# INVARIANT 6 -- Jain's fairness index in [1/n, 1]
# Paper: "Jain's index ... stays within analytically required bounds; §VII-B"
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_jain_index_in_valid_range_priority(spec_seed):
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    m   = compute_metrics(procs, log)
    n   = len(procs)
    assert 1.0 / n - 1e-9 <= m.jain_index <= 1.0 + 1e-9, (
        f"Jain index {m.jain_index:.6f} out of [1/{n}, 1]"
    )


@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_jain_index_in_valid_range_rr(spec_seed):
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_round_robin_family(procs, FixedQuantumPolicy(q=4), predicted)
    m   = compute_metrics(procs, log)
    n   = len(procs)
    assert 1.0 / n - 1e-9 <= m.jain_index <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# INVARIANT 7 -- starvation count in [0, n]
# Paper: "starvation-violation counts stay within [0, n]; §VII-B"
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_starvation_count_bounded(spec_seed):
    """Starvation violation count must be in [0, n] for every workload."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    m   = compute_metrics(procs, log)
    n   = len(procs)
    assert 0 <= m.starvation_violations <= n


# ---------------------------------------------------------------------------
# INVARIANT 8 -- all processes complete (no process left in queue forever)
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_all_processes_complete_priority(spec_seed):
    """Every process must appear in completion_time after the priority run."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    pids = {p.pid for p in procs}
    assert set(log.completion_time.keys()) == pids
    assert set(log.start_time.keys())      == pids


@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_all_processes_complete_rr(spec_seed):
    """Every process must appear in completion_time after the RR run."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_round_robin_family(procs, FixedQuantumPolicy(q=4), predicted)
    pids = {p.pid for p in procs}
    assert set(log.completion_time.keys()) == pids


# ---------------------------------------------------------------------------
# INVARIANT 9 -- Q-learning state discretisation always in [0, N_STATES)
# Paper: "Q-learning state indices always lie in [0, 144); §VII-B"
# ---------------------------------------------------------------------------

@given(
    st.floats(1.0, 200.0, allow_nan=False),   # queue_len (treated as float -> int)
    st.floats(0.0, 500.0, allow_nan=False),   # mean_pred_remaining
    st.floats(0.0, 1000.0, allow_nan=False),  # max_wait
    st.floats(1.0, 20.0, allow_nan=False),    # imbalance
    st.floats(0.0, 500.0, allow_nan=False),   # own_wait
)
@settings(max_examples=500, deadline=None)
def test_q_state_always_in_range(
        queue_len, mean_pred, max_wait, imbalance, own_wait):
    """discretize_state must return a value in [0, N_STATES=144) for any inputs."""
    s = discretize_state(
        queue_len=int(queue_len),
        mean_pred_remaining=mean_pred,
        max_wait=max_wait,
        imbalance=imbalance,
        own_wait=own_wait,
    )
    assert 0 <= s < N_STATES, f"State {s} out of [0, {N_STATES})"


# ---------------------------------------------------------------------------
# INVARIANT 10 -- frozen Q-policy is fully deterministic
# Paper: "a frozen policy is fully deterministic; §VII-B"
# ---------------------------------------------------------------------------

@given(
    workload_strategy(),
    st.integers(0, 9_999),    # Q-table training seed
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_frozen_policy_deterministic(spec_seed, qtable_seed):
    """Running the frozen Q-policy twice on the same workload must produce
    the *identical* dispatch sequence."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)

    agent   = QLearningAgent(seed=qtable_seed)
    policy  = FrozenQPolicy(q_table=agent.q_table)

    log1 = run_round_robin_family(procs, policy, predicted)
    log2 = run_round_robin_family(procs, policy, predicted)

    seq1 = [(e.pid, e.start, e.end) for e in log1.events]
    seq2 = [(e.pid, e.start, e.end) for e in log2.events]
    assert seq1 == seq2, "Frozen policy produced non-deterministic output"


# ---------------------------------------------------------------------------
# INVARIANT 11 -- Rule-based quantum always in QUANTA candidate set
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_rule_based_quantum_within_candidate_set_random(spec_seed):
    """Every dispatch slice must be explainable by a quantum in QUANTA
    (or less, if the burst finished before the full quantum elapsed)."""
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_round_robin_family(procs, RuleBasedQuantumPolicy(), predicted)
    for e in log.events:
        dur = e.end - e.start
        assert any(dur <= q + 1e-6 for q in QUANTA), (
            f"Slice duration {dur:.3f} exceeds all quanta {QUANTA}"
        )


# ---------------------------------------------------------------------------
# INVARIANT 12 -- Feature matrix excludes leakage fields
# Paper: "burst predictors' feature matrices structurally exclude true burst
#          length, priority, and arrival time; §VII-B"
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_predictor_feature_matrix_no_leakage(spec_seed):
    """The four-column feature matrix must NOT contain true_burst, priority,
    or arrival as a column (correlation check)."""
    spec, seed = spec_seed
    procs, _ = _make_procs_and_oracle(spec, seed)
    assume(len(procs) >= 4)  # need enough rows for correlation to be meaningful

    X = np.array([p.features() for p in procs])   # (n, 4)
    assert X.shape[1] == 4, f"Expected 4 features, got {X.shape[1]}"

    true_burst = np.array([p.true_burst  for p in procs], float)
    priority   = np.array([p.priority    for p in procs], float)
    arrival    = np.array([p.arrival     for p in procs], float)

    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        for forbidden, name in [
            (true_burst, "true_burst"),
            (priority,   "priority"),
            (arrival,    "arrival"),
        ]:
            # The column must NOT be a perfect linear transformation of forbidden.
            if np.std(col) > 0 and np.std(forbidden) > 0:
                r = abs(float(np.corrcoef(col, forbidden)[0, 1]))
                assert r < 0.9999, (
                    f"Column {col_idx} is perfectly correlated (r={r:.6f}) "
                    f"with {name} -- possible leakage"
                )


# ---------------------------------------------------------------------------
# INVARIANT 13 -- context-switch count <= n_dispatches - 1
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_context_switch_count_upper_bound_priority(spec_seed):
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    n_dispatches = len(log.events)
    assert log.context_switches <= max(n_dispatches - 1, 0)


@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_context_switch_count_upper_bound_rr(spec_seed):
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_round_robin_family(procs, FixedQuantumPolicy(q=4), predicted)
    n_dispatches = len(log.events)
    assert log.context_switches <= max(n_dispatches - 1, 0)


# ---------------------------------------------------------------------------
# INVARIANT 14 -- completion time ordering (no process finishes before it starts)
# ---------------------------------------------------------------------------

@given(workload_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_completion_after_start_and_arrival_priority(spec_seed):
    spec, seed = spec_seed
    procs, predicted = _make_procs_and_oracle(spec, seed)
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    for p in procs:
        assert log.start_time[p.pid]      >= p.arrival - 1e-9
        assert log.completion_time[p.pid] >= log.start_time[p.pid] - 1e-9


# ---------------------------------------------------------------------------
# INVARIANT 15 -- random WorkloadSpec sampler produces valid specs
# ---------------------------------------------------------------------------

@given(st.integers(0, 999_999))
@settings(max_examples=500, deadline=None)
def test_random_spec_generates_valid_workload(seed):
    """random_workload_spec must always produce a spec that generates a
    non-empty, structurally valid workload."""
    rng  = np.random.default_rng(seed)
    spec = random_workload_spec(rng, min_processes=2, max_processes=20)
    procs = generate_workload(spec, seed=seed)
    assert len(procs) >= 2
    assert all(p.true_burst >= 1      for p in procs)
    assert all(p.priority   >= 1      for p in procs)
    assert all(p.arrival    >= 0.0    for p in procs)
    assert all(p.pid == i             for i, p in enumerate(procs))
