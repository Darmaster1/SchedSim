"""
Worked example (Section III-A of the professor-revised study plan): three
processes, hand-computed predictions, checked against the simulator's own
predictor implementations to machine precision.

  P0 -- three completed CPU bursts (each followed by an I/O return), with a
        perfect log-linear feature/burst relationship, so the regression
        component activates and its exact value can be hand-derived.
  P1 -- only one prior burst: below `min_samples`, so every window-based
        predictor must fall back to the exponential-averaging estimate.
  P2 -- three prior bursts with a CONSTANT feature (zero variance): the
        correlation is undefined by the usual formula and is defined to be
        zero by convention, so every predictor again falls back to EA.

All reference values below are computed by hand in the comments and
asserted to within 1e-6 relative tolerance.
"""
from __future__ import annotations

import math

import pytest

from schedsim.cwabp import (Burst, CWABPPredictor, ExponentialAveragingPredictor,
                             FixedWeightPredictor, RegressionOnlyPredictor)

BETA = 0.5
INITIAL = 10.0
MIN_SAMPLES = 3


def _feed(predictor, pid, xs, ys):
    for i, (x, y) in enumerate(zip(xs, ys)):
        predictor.observe(Burst(pid, i, float(i), true_length=y, feature=x), y)


def test_worked_example_P0_regression_activates_after_three_bursts():
    # x = [0, 1, 2], ln(y) = [1, 2, 3]  =>  y = [e^1, e^2, e^3] (exact log-linear fit)
    xs = [0.0, 1.0, 2.0]
    ys = [math.exp(1), math.exp(2), math.exp(3)]

    # --- Hand-computed exponential-averaging trajectory (beta=0.5, initial=10) ---
    est0 = BETA * ys[0] + (1 - BETA) * INITIAL      # 0.5*e^1 + 0.5*10
    est1 = BETA * ys[1] + (1 - BETA) * est0          # 0.5*e^2 + 0.5*est0
    est2 = BETA * ys[2] + (1 - BETA) * est1          # 0.5*e^3 + 0.5*est1
    assert est0 == pytest.approx(0.5 * math.e + 5.0)
    assert est1 == pytest.approx(0.5 * math.exp(2) + 0.5 * est0)
    assert est2 == pytest.approx(0.5 * math.exp(3) + 0.5 * est1)

    # --- Hand-computed regression on (x, ln y) = (0,1),(1,2),(2,3): slope=1, intercept=1 ---
    slope, intercept = 1.0, 1.0
    x3 = 3.0
    raw_log_pred = intercept + slope * x3  # = 4.0
    log_exp_est = math.log(est2)
    lo, hi = log_exp_est - math.log(5.0), log_exp_est + math.log(5.0)
    clamped_log_pred = min(max(raw_log_pred, lo), hi)
    reg_pred = math.exp(clamped_log_pred)
    # With these numbers the clamp is NOT active (4.0 is within [lo, hi]):
    assert clamped_log_pred == pytest.approx(4.0)
    assert reg_pred == pytest.approx(math.exp(4.0))

    r = 1.0  # perfect correlation by construction

    # --- Feed the same history into the real predictors and compare ---
    ea = ExponentialAveragingPredictor(beta=BETA, initial=INITIAL)
    _feed(ea, 0, xs, ys)
    assert ea.predict(Burst(0, 3, 3.0, 0.0, x3)) == pytest.approx(est2, rel=1e-9)

    reg_only = RegressionOnlyPredictor(window=8, min_samples=MIN_SAMPLES, beta=BETA, initial=INITIAL)
    _feed(reg_only, 0, xs, ys)
    assert reg_only.predict(Burst(0, 3, 3.0, 0.0, x3)) == pytest.approx(reg_pred, rel=1e-9)

    shrink = 0.5
    cwabp = CWABPPredictor(window=8, shrink=shrink, min_samples=MIN_SAMPLES, beta=BETA, initial=INITIAL)
    _feed(cwabp, 0, xs, ys)
    weight_cwabp = shrink * r ** 2  # = 0.5
    expected_cwabp_log = weight_cwabp * math.log(reg_pred) + (1 - weight_cwabp) * math.log(est2)
    assert cwabp.predict(Burst(0, 3, 3.0, 0.0, x3)) == pytest.approx(math.exp(expected_cwabp_log), rel=1e-9)

    fixed = FixedWeightPredictor(weight=0.25, window=8, min_samples=MIN_SAMPLES, beta=BETA, initial=INITIAL)
    _feed(fixed, 0, xs, ys)
    expected_fixed_log = 0.25 * math.log(reg_pred) + 0.75 * math.log(est2)
    assert fixed.predict(Burst(0, 3, 3.0, 0.0, x3)) == pytest.approx(math.exp(expected_fixed_log), rel=1e-9)


def test_worked_example_P1_insufficient_history_falls_back_to_EA():
    xs, ys = [1.0], [12.0]  # only ONE prior burst: below min_samples=3
    est0 = BETA * ys[0] + (1 - BETA) * INITIAL

    for cls, kwargs in [
        (CWABPPredictor, dict(window=8, shrink=0.5, min_samples=MIN_SAMPLES, beta=BETA, initial=INITIAL)),
        (RegressionOnlyPredictor, dict(window=8, min_samples=MIN_SAMPLES, beta=BETA, initial=INITIAL)),
        (FixedWeightPredictor, dict(weight=0.5, window=8, min_samples=MIN_SAMPLES, beta=BETA, initial=INITIAL)),
    ]:
        pred = cls(**kwargs)
        _feed(pred, 1, xs, ys)
        got = pred.predict(Burst(1, 1, 1.0, 0.0, feature=2.0))
        assert got == pytest.approx(est0, rel=1e-9), f"{cls.__name__} should fall back to EA"


def test_worked_example_P2_zero_feature_variance_falls_back_to_EA():
    # Constant feature across all three prior bursts: var_x = 0, so r is
    # DEFINED to be 0 by convention (not undefined/NaN), and every
    # predictor's regression component degenerates to the EA baseline.
    xs = [5.0, 5.0, 5.0]
    ys = [9.0, 11.0, 13.0]
    est0 = BETA * ys[0] + (1 - BETA) * INITIAL
    est1 = BETA * ys[1] + (1 - BETA) * est0
    est2 = BETA * ys[2] + (1 - BETA) * est1

    for cls, kwargs in [
        (CWABPPredictor, dict(window=8, shrink=0.5, min_samples=MIN_SAMPLES, beta=BETA, initial=INITIAL)),
        (RegressionOnlyPredictor, dict(window=8, min_samples=MIN_SAMPLES, beta=BETA, initial=INITIAL)),
        (FixedWeightPredictor, dict(weight=0.9, window=8, min_samples=MIN_SAMPLES, beta=BETA, initial=INITIAL)),
    ]:
        pred = cls(**kwargs)
        _feed(pred, 2, xs, ys)
        got = pred.predict(Burst(2, 3, 3.0, 0.0, feature=5.0))
        assert got == pytest.approx(est2, rel=1e-9), f"{cls.__name__} should fall back to EA under zero variance"
