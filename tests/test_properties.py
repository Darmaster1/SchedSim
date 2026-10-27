"""
Property-based randomized testing (Hypothesis).

Unlike tests/test_*.py, which check specific hand-picked cases, every test
here states an invariant that must hold for *any* input in a described
range, and Hypothesis searches that space for a counterexample across many
randomly generated (and adversarially shrunk) inputs per run. If a property
is violated, Hypothesis automatically shrinks the failing input down to a
minimal reproducing example before reporting it.

10 property suites, covering: workload generation invariants, the A/B/C
and D/E/F structural identities also spot-checked in test_schedulers.py
(here re-verified across a much wider input space), Q-learning state
discretization bounds, CWABP's no-future-information guarantee, Jain's
index bounds, and burst-predictor leakage control.
"""
from __future__ import annotations

import math

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from schedsim.cwabp import Burst, CWABPPredictor
from schedsim.metrics import compute_metrics
from schedsim.predictors import _feature_matrix
from schedsim.qlearning import N_STATES, discretize_state
from schedsim.schedulers import (AlgorithmA_Priority, AlgorithmB_WeightedML,
                                  AlgorithmC_ProtectedML, FixedQuantumPolicy,
                                  run_priority_family, run_round_robin_family)
from schedsim.workload import WorkloadSpec, generate_workload

_COMMON_SETTINGS = settings(max_examples=40, deadline=None,
                             suppress_health_check=[HealthCheck.too_slow])

_n_processes_st = st.integers(min_value=2, max_value=40)
_seed_st = st.integers(min_value=0, max_value=10_000_000)
_priority_mode_st = st.sampled_from(["uniform", "skewed", "adversarial"])
_burst_mode_st = st.sampled_from(["short", "long", "mixed", "heavy_tailed"])
_arrival_mode_st = st.sampled_from(["simultaneous", "staggered", "bursty"])


def _random_workload(n, seed, priority_mode="uniform", burst_mode="mixed", arrival_mode="staggered"):
    spec = WorkloadSpec("prop", n_processes=n, priority_mode=priority_mode,
                         burst_mode=burst_mode, arrival_mode=arrival_mode)
    return generate_workload(spec, seed)


# ---------------------------------------------------------------------------
# 1. Workload generation invariants
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(n=_n_processes_st, seed=_seed_st, priority_mode=_priority_mode_st,
       burst_mode=_burst_mode_st)
def test_property_workload_fields_always_in_range(n, seed, priority_mode, burst_mode):
    procs = _random_workload(n, seed, priority_mode, burst_mode)
    assert len(procs) == n
    for p in procs:
        assert 1 <= p.priority <= 10
        assert 1 <= p.true_burst <= 400
        assert p.arrival >= 0.0


# ---------------------------------------------------------------------------
# 2. Turnaround = Waiting + Burst, for any workload, under Algorithm A
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(n=_n_processes_st, seed=_seed_st)
def test_property_turnaround_decomposition_holds_for_any_workload(n, seed):
    procs = _random_workload(n, seed)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    for p in procs:
        t = log.completion_time[p.pid] - p.arrival
        w = t - p.true_burst
        r = log.start_time[p.pid] - p.arrival
        assert t >= 0
        assert r >= -1e-9
        assert math.isclose(t, w + p.true_burst, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 3. B(alpha=1) == A, for any workload (wider re-check of test_schedulers.py)
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(n=_n_processes_st, seed=_seed_st, priority_mode=_priority_mode_st)
def test_property_B_alpha_1_equals_A_for_any_workload(n, seed, priority_mode):
    procs = _random_workload(n, seed, priority_mode)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log_a = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    log_b = run_priority_family(procs, AlgorithmB_WeightedML(alpha=1.0), predicted)
    assert [e.pid for e in log_a.events] == [e.pid for e in log_b.events]


# ---------------------------------------------------------------------------
# 4. C with safeguards disabled == B, for any workload and any alpha
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(n=_n_processes_st, seed=_seed_st, alpha=st.floats(min_value=0.0, max_value=1.0,
                                                          allow_nan=False, allow_infinity=False))
def test_property_C_disabled_equals_B_for_any_workload_and_alpha(n, seed, alpha):
    procs = _random_workload(n, seed)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log_b = run_priority_family(procs, AlgorithmB_WeightedML(alpha), predicted)
    log_c = run_priority_family(procs, AlgorithmC_ProtectedML(alpha, w_max=1e9, gap=999), predicted)
    assert [e.pid for e in log_b.events] == [e.pid for e in log_c.events]


# ---------------------------------------------------------------------------
# 5. C with gap=0: inversions never exceed override activations
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(n=_n_processes_st, seed=_seed_st,
       w_max=st.floats(min_value=5.0, max_value=200.0, allow_nan=False, allow_infinity=False))
def test_property_C_gap_zero_bounds_inversions_by_activations(n, seed, w_max):
    procs = _random_workload(n, seed, priority_mode="adversarial")
    predicted = {p.pid: float(p.true_burst) for p in procs}
    policy = AlgorithmC_ProtectedML(alpha=0.5, w_max=w_max, gap=0)
    run_priority_family(procs, policy, predicted)
    assert policy.inversions <= policy.activations


# ---------------------------------------------------------------------------
# 6. Round Robin: every process in the workload eventually completes
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(n=_n_processes_st, seed=_seed_st,
       quantum=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False))
def test_property_rr_every_process_completes(n, seed, quantum):
    procs = _random_workload(n, seed)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_round_robin_family(procs, FixedQuantumPolicy(quantum), predicted)
    assert set(log.completion_time.keys()) == {p.pid for p in procs}
    for p in procs:
        assert log.completion_time[p.pid] >= p.arrival


# ---------------------------------------------------------------------------
# 7. Q-learning state discretization always lands in [0, N_STATES)
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(q=st.floats(min_value=0, max_value=1000, allow_nan=False),
       b=st.floats(min_value=0, max_value=1000, allow_nan=False),
       w=st.floats(min_value=0, max_value=5000, allow_nan=False),
       i=st.floats(min_value=0, max_value=1000, allow_nan=False))
def test_property_state_discretization_always_bounded(q, b, w, i):
    s = discretize_state(q, b, w, i)
    assert 0 <= s < N_STATES


# ---------------------------------------------------------------------------
# 8. CWABP: prediction for burst n never depends on burst n's own true length
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(feature=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
       true_len_a=st.floats(min_value=0.1, max_value=10000, allow_nan=False, allow_infinity=False),
       true_len_b=st.floats(min_value=0.1, max_value=10000, allow_nan=False, allow_infinity=False))
def test_property_cwabp_no_future_leakage(feature, true_len_a, true_len_b):
    pred = CWABPPredictor(window=5, shrink=0.8, min_samples=2, beta=0.5)
    for i, (x, y) in enumerate([(1.0, 10.0), (1.2, 11.0)]):
        b = Burst(0, i, float(i), true_length=y, feature=x)
        pred.observe(b, y)

    b_a = Burst(0, 2, 2.0, true_length=true_len_a, feature=feature)
    b_b = Burst(0, 2, 2.0, true_length=true_len_b, feature=feature)
    # Same history, same feature, different (not-yet-observed) true length:
    # predictions must be identical, and finite/positive.
    pa, pb = pred.predict(b_a), pred.predict(b_b)
    assert pa == pb
    assert pa >= 1.0
    assert math.isfinite(pa)


# ---------------------------------------------------------------------------
# 9. Jain's fairness index stays within [1/n, 1] for any workload
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(n=_n_processes_st, seed=_seed_st, arrival_mode=_arrival_mode_st)
def test_property_jain_index_bounded_for_any_workload(n, seed, arrival_mode):
    procs = _random_workload(n, seed, arrival_mode=arrival_mode)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    m = compute_metrics(procs, log)
    assert 1.0 / n - 1e-9 <= m.jain_index <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 10. Burst predictor feature matrix never leaks true burst length
# ---------------------------------------------------------------------------

@_COMMON_SETTINGS
@given(n=_n_processes_st, seed=_seed_st, burst_mode=_burst_mode_st)
def test_property_feature_matrix_never_equals_true_burst(n, seed, burst_mode):
    procs = _random_workload(n, seed, burst_mode=burst_mode)
    X = _feature_matrix(procs, use_endcode=True)
    assert X.shape == (n, 4)
    true_bursts = [p.true_burst for p in procs]
    for col in range(X.shape[1]):
        assert list(X[:, col]) != true_bursts
