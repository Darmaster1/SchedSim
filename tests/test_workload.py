import numpy as np
import pytest

from schedsim.workload import (SEED_OFFSETS, WorkloadSpec, generate_workload,
                                in_distribution_categories, shifted_test_only_categories)


def test_determinism_same_seed_same_output():
    spec = WorkloadSpec("det_test", n_processes=20)
    a = generate_workload(spec, seed=42)
    b = generate_workload(spec, seed=42)
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert pa.arrival == pb.arrival
        assert pa.priority == pb.priority
        assert pa.true_burst == pb.true_burst
        assert np.allclose(pa.features(), pb.features())


def test_different_seeds_differ():
    spec = WorkloadSpec("det_test2", n_processes=20)
    a = generate_workload(spec, seed=1)
    b = generate_workload(spec, seed=2)
    arrivals_a = [p.arrival for p in a]
    arrivals_b = [p.arrival for p in b]
    assert arrivals_a != arrivals_b


def test_train_val_test_seed_ranges_disjoint():
    offsets = list(SEED_OFFSETS.values())
    assert len(set(offsets)) == len(offsets)
    # category index multiplier (10_000 in experiment.py) must not overlap
    # adjacent split offsets for any reasonable number of categories.
    assert SEED_OFFSETS["val"] - SEED_OFFSETS["train"] >= 10_000 * 50
    assert SEED_OFFSETS["test"] - SEED_OFFSETS["val"] >= 10_000 * 50


def test_priority_range():
    spec = WorkloadSpec("prio_test", n_processes=50, priority_mode="uniform")
    procs = generate_workload(spec, seed=1)
    for p in procs:
        assert 1 <= p.priority <= 10


def test_burst_length_positive_and_bounded():
    for mode in ("short", "long", "mixed", "heavy_tailed"):
        spec = WorkloadSpec(f"burst_{mode}", n_processes=50, burst_mode=mode)
        procs = generate_workload(spec, seed=1)
        for p in procs:
            assert p.true_burst >= 1
            assert p.true_burst <= 400


def test_feature_strength_correlation_direction():
    """A 'strong' feature-strength workload should give much better linear
    correlation between `text` and log(true_burst) than a 'misleading' one,
    and 'misleading' should be negatively correlated."""
    strong_spec = WorkloadSpec("strong", n_processes=200, feature_strength="strong")
    mislead_spec = WorkloadSpec("mislead", n_processes=200, feature_strength="misleading")

    strong_procs = generate_workload(strong_spec, seed=1)
    mislead_procs = generate_workload(mislead_spec, seed=1)

    def corr(procs):
        text = np.array([p.text for p in procs])
        log_b = np.log(np.array([p.true_burst for p in procs]))
        return np.corrcoef(text, log_b)[0, 1]

    strong_corr = corr(strong_procs)
    mislead_corr = corr(mislead_procs)

    assert strong_corr > 0.8
    assert mislead_corr < -0.3


def test_category_catalogue_sizes():
    in_dist = in_distribution_categories()
    shifted = shifted_test_only_categories()
    assert len(in_dist) == 10
    assert len(shifted) == 3
    names = {s.name for s in in_dist} | {s.name for s in shifted}
    assert len(names) == 13  # all unique
