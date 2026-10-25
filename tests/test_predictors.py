import numpy as np
import pytest

from schedsim.predictors import (MODEL_FACTORIES, _feature_matrix, evaluate_predictor,
                                  fit_predictor, out_of_fold_predictions, select_best_predictor)
from schedsim.workload import WorkloadSpec, generate_workload


def _train_val(seed_train=1, seed_val=2, n=100, strength="strong"):
    spec = WorkloadSpec("pred", n_processes=n, feature_strength=strength)
    train = generate_workload(spec, seed_train)
    val = generate_workload(spec, seed_val)
    return train, val


def test_feature_matrix_shape_excludes_leakage_fields():
    train, _ = _train_val()
    X = _feature_matrix(train, use_endcode=True)
    assert X.shape == (len(train), 4)  # vsize, text, endcode, vmlib ONLY
    X3 = _feature_matrix(train, use_endcode=False)
    assert X3.shape == (len(train), 3)


def test_all_five_models_fit_and_predict():
    train, val = _train_val()
    for name in MODEL_FACTORIES:
        fp = fit_predictor(name, train, use_endcode=True)
        preds = fp.predict(val)
        assert len(preds) == len(val)
        assert np.all(preds >= 1.0)  # floored at one time unit


def test_strong_feature_relationship_beats_weak_on_rmse():
    train_s, val_s = _train_val(strength="strong")
    train_w, val_w = _train_val(strength="weak")
    fp_s = fit_predictor("linear", train_s, use_endcode=True)
    fp_w = fit_predictor("linear", train_w, use_endcode=True)
    m_s = evaluate_predictor(fp_s, val_s)
    m_w = evaluate_predictor(fp_w, val_w)
    assert m_s["rmse"] < m_w["rmse"]
    assert m_s["r2"] > m_w["r2"]


def test_select_best_predictor_returns_lowest_val_rmse():
    train, val = _train_val()
    best_name, best_fp, all_metrics = select_best_predictor(train, val)
    assert best_name in MODEL_FACTORIES
    best_rmse = all_metrics[best_name]["rmse"]
    assert all(best_rmse <= m["rmse"] + 1e-9 for m in all_metrics.values())


def test_out_of_fold_predictions_not_equal_to_naive_in_sample_fit():
    """Out-of-fold predictions should differ from predictions of a model
    trained on the FULL set and evaluated on itself (which would be
    optimistically biased) for at least some points."""
    train, _ = _train_val(n=60)
    oof = out_of_fold_predictions(train, "tree", n_folds=5, seed=0)
    full_fp = fit_predictor("tree", train, use_endcode=True)
    in_sample = full_fp.predict(train)
    assert not np.allclose(oof, in_sample)


def test_predictor_never_sees_true_burst_as_input():
    """Structural leakage check: verify the training call signature path
    never passes burst/priority/arrival data into the feature matrix."""
    train, _ = _train_val()
    X = _feature_matrix(train, use_endcode=True)
    true_bursts = np.array([p.true_burst for p in train])
    # None of the 4 feature columns should be a trivial copy of true_burst
    for col in range(X.shape[1]):
        assert not np.allclose(X[:, col], true_bursts)
