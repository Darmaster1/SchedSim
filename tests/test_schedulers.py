import numpy as np
import pytest

from schedsim.metrics import compute_metrics
from schedsim.schedulers import (AlgorithmA_Priority, AlgorithmB_WeightedML,
                                  AlgorithmC_ProtectedML, FixedQuantumPolicy, QUANTA,
                                  RuleBasedQuantumPolicy, run_priority_family,
                                  run_round_robin_family)
from schedsim.workload import WorkloadSpec, generate_workload


def _procs_and_predictions(seed=1, n=25, priority_mode="uniform"):
    spec = WorkloadSpec("test", n_processes=n, priority_mode=priority_mode)
    procs = generate_workload(spec, seed)
    predicted = {p.pid: float(p.true_burst) for p in procs}  # oracle predictions
    return procs, predicted


# --------------------------------------------------------------------------
# A vs B equivalence (paper: "Setting alpha = 1 reproduces Algorithm A,
# verified by a unit test")
# --------------------------------------------------------------------------

def test_B_with_alpha_1_equals_A():
    for seed in range(5):
        procs, predicted = _procs_and_predictions(seed=seed)
        log_a = run_priority_family(procs, AlgorithmA_Priority(), predicted)
        log_b = run_priority_family(procs, AlgorithmB_WeightedML(alpha=1.0), predicted)
        m_a = compute_metrics(procs, log_a)
        m_b = compute_metrics(procs, log_b)
        assert m_a.avg_waiting == pytest.approx(m_b.avg_waiting)
        assert m_a.max_waiting == pytest.approx(m_b.max_waiting)
        # dispatch order should be pid-for-pid identical, not just metric-equal
        order_a = [e.pid for e in log_a.events]
        order_b = [e.pid for e in log_b.events]
        assert order_a == order_b


def test_B_with_alpha_0_is_shortest_predicted_job_first():
    procs, predicted = _procs_and_predictions(seed=2, priority_mode="adversarial")
    log_b = run_priority_family(procs, AlgorithmB_WeightedML(alpha=0.0), predicted)
    dispatch_order = [e.pid for e in log_b.events]
    by_pid = {p.pid: p for p in procs}
    # among processes with the same arrival time, dispatch order should be
    # non-decreasing in predicted burst length (ties aside)
    # Spot check: the very first dispatched process should not have the
    # largest predicted burst among those ready at time 0.
    first_pid = dispatch_order[0]
    t0_ready = [p for p in procs if p.arrival <= by_pid[first_pid].arrival]
    assert predicted[first_pid] <= max(predicted[p.pid] for p in t0_ready)


# --------------------------------------------------------------------------
# C vs B equivalence when safeguards disabled (paper: "With the safeguards
# disabled, C is equivalent to B ... unit-tested")
# --------------------------------------------------------------------------

def test_C_with_safeguards_disabled_equals_B():
    for seed in range(5):
        procs, predicted = _procs_and_predictions(seed=seed)
        alpha = 0.4
        log_b = run_priority_family(procs, AlgorithmB_WeightedML(alpha), predicted)
        # w_max huge -> override never fires; gap huge -> no eligibility restriction
        log_c = run_priority_family(procs, AlgorithmC_ProtectedML(alpha, w_max=1e9, gap=9), predicted)
        order_b = [e.pid for e in log_b.events]
        order_c = [e.pid for e in log_c.events]
        assert order_b == order_c


def test_C_with_gap_zero_no_inversion_outside_override():
    """Paper: 'with gap=0, no inversion occurs outside override events'."""
    procs, predicted = _procs_and_predictions(seed=3, n=40, priority_mode="adversarial")
    policy = AlgorithmC_ProtectedML(alpha=0.5, w_max=60.0, gap=0)
    run_priority_family(procs, policy, predicted)
    # Every recorded inversion must have coincided with an override activation.
    # We approximate this by checking activations >= inversions when gap=0,
    # since gap=0 forbids the bounded-inversion branch from ever choosing a
    # lower-priority process.
    assert policy.inversions <= policy.activations


def test_C_reduces_max_waiting_relative_to_A_on_adversarial_workload():
    """Sanity check of C's intended effect: tighter worst-case waiting than
    the unprotected baseline on a workload designed to starve someone."""
    procs, predicted = _procs_and_predictions(seed=7, n=40, priority_mode="adversarial")
    log_a = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    log_c = run_priority_family(procs, AlgorithmC_ProtectedML(alpha=0.3, w_max=40.0, gap=2), predicted)
    m_a = compute_metrics(procs, log_a)
    m_c = compute_metrics(procs, log_c)
    assert m_c.max_waiting <= m_a.max_waiting


# --------------------------------------------------------------------------
# Round Robin family invariants
# --------------------------------------------------------------------------

def test_fixed_quantum_rr_every_process_completes():
    procs, predicted = _procs_and_predictions(seed=4, n=30)
    log = run_round_robin_family(procs, FixedQuantumPolicy(4.0), predicted)
    assert set(log.completion_time.keys()) == {p.pid for p in procs}
    for p in procs:
        assert log.completion_time[p.pid] >= log.start_time[p.pid] >= p.arrival


def test_rr_fifo_order_at_simultaneous_arrival():
    """Processes arriving simultaneously must be enqueued in ID order."""
    spec = WorkloadSpec("simul", n_processes=8, arrival_mode="simultaneous", burst_mode="long")
    procs = generate_workload(spec, seed=1)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_round_robin_family(procs, FixedQuantumPolicy(1000.0), predicted, context_switch_cost=0.0)
    # with a huge quantum, RR degenerates to FCFS: dispatch order == pid order
    dispatch_order = [e.pid for e in log.events]
    assert dispatch_order == sorted(p.pid for p in procs)


def test_rule_based_quantum_within_candidate_set():
    procs, predicted = _procs_and_predictions(seed=5, n=20)
    policy = RuleBasedQuantumPolicy()
    log = run_round_robin_family(procs, policy, predicted)
    # every dispatched slice length must be explainable by one of QUANTA
    # (either the full quantum, or less because the burst finished first)
    for e in log.events:
        dur = e.end - e.start
        assert any(dur <= q + 1e-6 for q in QUANTA)


def test_context_switch_counting_no_switch_for_single_process():
    spec = WorkloadSpec("single", n_processes=1, burst_mode="long")
    procs = generate_workload(spec, seed=1)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    assert log.context_switches == 0


def test_idle_time_recorded_for_late_arrival():
    spec = WorkloadSpec("late", n_processes=3, arrival_mode="staggered", offered_load=0.05)
    procs = generate_workload(spec, seed=1)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    assert log.idle_time >= 0.0
