import math

import numpy as np
import pytest

from schedsim.cwabp import (Burst, CWABPPredictor, ExponentialAveragingPredictor,
                             simulate_srtf)


def test_exponential_averaging_converges_to_stable_value():
    pred = ExponentialAveragingPredictor(beta=0.5, initial=10.0)
    b = Burst(process_id=0, burst_index=0, ready_time=0.0, true_length=20.0, feature=0.0)
    for _ in range(50):
        pred.observe(b, 20.0)
    assert pred.predict(b) == pytest.approx(20.0, abs=0.01)


def test_cwabp_predict_uses_only_past_information_no_future_leakage():
    """Changing burst n's true length (observed only AFTER predict() is
    called) must not retroactively change the prediction already made."""
    pred = CWABPPredictor(window=5, shrink=0.8, min_samples=2, beta=0.5)
    b1 = Burst(0, 0, 0.0, 10.0, feature=1.0)
    b2 = Burst(0, 1, 10.0, 12.0, feature=1.2)
    pred.observe(b1, 10.0)
    pred.observe(b2, 12.0)

    b3 = Burst(0, 2, 22.0, true_length=999.0, feature=1.3)  # "future" true value
    prediction_before = pred.predict(b3)
    # Prediction must not depend on b3.true_length at all (it isn't even
    # read by predict()); confirm by mutating true_length and re-predicting
    # with history unchanged.
    b3_mutated = Burst(0, 2, 22.0, true_length=1.0, feature=1.3)
    prediction_after = pred.predict(b3_mutated)
    assert prediction_before == prediction_after


def test_cwabp_falls_back_to_exp_avg_with_insufficient_samples():
    pred = CWABPPredictor(window=5, shrink=0.9, min_samples=3, beta=0.5, initial=15.0)
    b = Burst(0, 0, 0.0, 10.0, feature=5.0)
    # No observations yet -> must equal the exponential-averaging initial value
    assert pred.predict(b) == pytest.approx(15.0)


def test_cwabp_bounded_even_under_spurious_small_window_correlation():
    """Regression extrapolation must stay within the trust-region clamp,
    not blow up, even when a small window yields a high sample correlation
    by chance on an out-of-range feature value."""
    pred = CWABPPredictor(window=4, shrink=1.0, min_samples=3, beta=0.5, initial=10.0)
    # Feed a window with high apparent (but spurious/noisy) correlation.
    pts = [(1.0, 9.0), (1.1, 11.0), (0.9, 8.5)]
    for i, (x, y) in enumerate(pts):
        b = Burst(0, i, float(i), true_length=y, feature=x)
        pred.observe(b, y)
    # Predict for a wildly out-of-range feature value.
    b_extreme = Burst(0, 99, 100.0, true_length=0.0, feature=1_000_000.0)
    pred_val = pred.predict(b_extreme)
    exp_est = pred._exp_estimate[0]
    assert pred_val <= exp_est * 5.0 + 1e-6
    assert pred_val >= exp_est / 5.0 - 1e-6


def test_simulate_srtf_every_burst_completes():
    r = simulate_srtf(n_processes=5, bursts_per_process=6, rho=0.5, sigma=0.3, seed=1,
                       predictor=ExponentialAveragingPredictor())
    assert r.n_bursts == 30
    assert r.avg_waiting >= 0
    assert r.avg_turnaround >= r.avg_waiting  # turnaround = waiting + burst length


def test_simulate_srtf_deterministic_given_seed():
    r1 = simulate_srtf(4, 5, rho=0.6, sigma=0.3, seed=7, predictor=ExponentialAveragingPredictor())
    r2 = simulate_srtf(4, 5, rho=0.6, sigma=0.3, seed=7, predictor=ExponentialAveragingPredictor())
    assert r1.avg_waiting == pytest.approx(r2.avg_waiting)
    assert r1.preemptions == r2.preemptions


def test_cwabp_does_not_catastrophically_underperform_on_uncorrelated_feature():
    """On a genuinely uncorrelated feature, CWABP should fall back toward
    exponential averaging rather than being badly misled (RMSE should stay
    within a small factor of EA's, not blow up)."""
    r_ea = simulate_srtf(6, 10, rho=0.02, sigma=0.5, seed=3,
                          predictor=ExponentialAveragingPredictor(beta=0.5))
    r_cw = simulate_srtf(6, 10, rho=0.02, sigma=0.5, seed=3,
                          predictor=CWABPPredictor(window=8, shrink=0.8, min_samples=3, beta=0.5))
    assert r_cw.rmse <= r_ea.rmse * 3.0
