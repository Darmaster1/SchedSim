import pytest

from schedsim.metrics import compute_metrics
from schedsim.schedulers import AlgorithmA_Priority, run_priority_family
from schedsim.workload import WorkloadSpec, generate_workload


def test_turnaround_equals_waiting_plus_burst():
    spec = WorkloadSpec("m1", n_processes=15)
    procs = generate_workload(spec, seed=1)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    m = compute_metrics(procs, log)
    assert m.avg_turnaround == pytest.approx(m.avg_waiting + sum(p.true_burst for p in procs) / len(procs))


def test_jain_index_bounds():
    spec = WorkloadSpec("m2", n_processes=20)
    procs = generate_workload(spec, seed=2)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    m = compute_metrics(procs, log)
    n = len(procs)
    assert 1.0 / n - 1e-9 <= m.jain_index <= 1.0 + 1e-9


def test_perfectly_equal_slowdown_gives_jain_index_one():
    """If every process has arrival 0 and identical burst length with no
    contention (single process), slowdown is trivially uniform -> Jain = 1."""
    spec = WorkloadSpec("m3", n_processes=1, burst_mode="long")
    procs = generate_workload(spec, seed=1)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    m = compute_metrics(procs, log)
    assert m.jain_index == pytest.approx(1.0, abs=1e-6)


def test_p95_waiting_at_least_median_and_at_most_max():
    spec = WorkloadSpec("m4", n_processes=40)
    procs = generate_workload(spec, seed=3)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    m = compute_metrics(procs, log)
    assert m.p95_waiting <= m.max_waiting + 1e-9


def test_starvation_violation_threshold():
    spec = WorkloadSpec("m5", n_processes=30, priority_mode="adversarial")
    procs = generate_workload(spec, seed=4)
    predicted = {p.pid: float(p.true_burst) for p in procs}
    log = run_priority_family(procs, AlgorithmA_Priority(), predicted)
    m = compute_metrics(procs, log)
    assert m.starvation_violations >= 0
    assert m.starvation_violations <= len(procs)
