"""
Correlation-Weighted Adaptive Burst Predictor (CWABP) for SRTF scheduling.

This is the Section 9 extension: a different process model (multi-burst,
alternating CPU/I-O) from the rest of the simulator, used only to compare
CWABP against classical exponential averaging on SRTF scheduling.

Design note: a process's *next* burst only becomes ready after its current
burst has actually finished executing (plus an I/O wait) -- which, under
queueing, happens later than it would in an uncontended system. Burst
arrival times are therefore generated online, during the simulation itself,
rather than precomputed in advance; precomputing them would let two bursts
of the *same* process appear "ready" simultaneously whenever the system is
congested, which is not a physically meaningful workload.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Burst:
    process_id: int
    burst_index: int
    ready_time: float
    true_length: float
    feature: float  # observable scalar x, known at ready_time


@dataclass
class _ProcessGenerator:
    pid: int
    rng: np.random.Generator
    rho: float
    sigma: float
    mean_log_burst: float
    bursts_per_process: int
    phase_change: bool
    correlation_break: bool
    heavy_tailed: bool
    io_mean: float = 4.0
    k: int = 0

    def next_burst(self, ready_time: float) -> Optional[Burst]:
        if self.k >= self.bursts_per_process:
            return None
        cur_rho = self.rho
        if self.correlation_break and self.k >= self.bursts_per_process // 2:
            cur_rho = -self.rho
        if self.phase_change and self.k == self.bursts_per_process // 2:
            self.mean_log_burst += 1.0

        z1 = self.rng.normal()
        z2 = self.rng.normal()
        log_burst = self.mean_log_burst + self.sigma * z1
        feature = cur_rho * z1 + math.sqrt(max(1 - cur_rho ** 2, 0.0)) * z2

        if self.heavy_tailed:
            true_len = float(np.clip((self.rng.pareto(1.5) + 1) * math.exp(self.mean_log_burst) / 5, 1, 400))
        else:
            true_len = float(np.clip(math.exp(log_burst), 1, 200))

        b = Burst(self.pid, self.k, ready_time, true_len, float(feature))
        self.k += 1
        return b

    def io_wait(self) -> float:
        return float(self.rng.exponential(scale=self.io_mean))


def _make_generators(n_processes: int, bursts_per_process: int, rho: float, sigma: float,
                      seed: int, phase_change: bool, correlation_break: bool,
                      heavy_tailed: bool) -> Tuple[List[_ProcessGenerator], List[float]]:
    master_rng = np.random.default_rng(seed)
    gens, initial_ready = [], []
    for pid in range(n_processes):
        proc_seed = int(master_rng.integers(0, 2**31 - 1))
        rng = np.random.default_rng(proc_seed)
        mean_log_burst = rng.uniform(2.0, 3.5)
        gens.append(_ProcessGenerator(
            pid=pid, rng=rng, rho=rho, sigma=sigma, mean_log_burst=mean_log_burst,
            bursts_per_process=bursts_per_process, phase_change=phase_change,
            correlation_break=correlation_break, heavy_tailed=heavy_tailed,
        ))
        initial_ready.append(float(rng.exponential(scale=5.0)))
    return gens, initial_ready


# --------------------------- Predictors -----------------------------------

class ExponentialAveragingPredictor:
    name = "exp_avg"

    def __init__(self, beta: float = 0.5, initial: float = 10.0):
        self.beta = beta
        self.initial = initial
        self._estimate: Dict[int, float] = {}

    def predict(self, b: Burst) -> float:
        return self._estimate.get(b.process_id, self.initial)

    def observe(self, b: Burst, actual: float):
        prev = self._estimate.get(b.process_id, self.initial)
        self._estimate[b.process_id] = self.beta * actual + (1 - self.beta) * prev


def _window_regression(hist: List[Tuple[float, float]], min_samples: int,
                        feature: float, exp_est: float) -> Tuple[Optional[float], float]:
    """Shared regression step for every window-based predictor below.
    Returns (clamped_regression_prediction_or_None, sample_correlation_r).
    None means "too little history" -- the caller must fall back to the
    exponential-averaging estimate. The trust-region clamp (log-space,
    within a factor of 5x the exponential-averaging estimate) guards against
    a spurious small-window correlation combining with an out-of-range
    feature value to extrapolate to an absurd (or exp()-overflowing) length;
    this is the same safeguard for every predictor that uses regression at
    all, not just CWABP, so the comparison in Section IV is apples-to-apples.
    """
    if len(hist) < min_samples:
        return None, 0.0
    xs = np.array([h[0] for h in hist])
    ys = np.array([h[1] for h in hist])
    var_x, var_y = xs.var(), ys.var()
    if var_x < 1e-9 or var_y < 1e-9:
        r = 0.0
    else:
        r = float(np.corrcoef(xs, ys)[0, 1])

    if abs(r) > 1e-9 and var_x > 1e-9:
        slope = np.cov(xs, ys, bias=True)[0, 1] / var_x
        intercept = ys.mean() - slope * xs.mean()
        log_exp_est = math.log(max(exp_est, 1e-6))
        raw_log_pred = intercept + slope * feature
        clamped_log_pred = float(np.clip(raw_log_pred, log_exp_est - math.log(5.0),
                                          log_exp_est + math.log(5.0)))
        reg_pred = math.exp(clamped_log_pred)
    else:
        reg_pred = exp_est
    return reg_pred, r


class _WindowedPredictorBase:
    """Shared history/state machinery for every predictor that maintains a
    per-process sliding window of (feature, log-burst) pairs alongside an
    exponential-averaging baseline. Subclasses implement only `predict`."""

    def __init__(self, window: int = 8, min_samples: int = 3, beta: float = 0.5,
                 initial: float = 10.0):
        self.window = window
        self.min_samples = min_samples
        self.beta = beta
        self.initial = initial
        self._history: Dict[int, List[Tuple[float, float]]] = {}  # pid -> [(x, log_y)]
        self._exp_estimate: Dict[int, float] = {}

    def _state(self, b: Burst) -> Tuple[List[Tuple[float, float]], float]:
        hist = self._history.get(b.process_id, [])[-self.window:]
        exp_est = self._exp_estimate.get(b.process_id, self.initial)
        return hist, exp_est

    def observe(self, b: Burst, actual: float):
        prev = self._exp_estimate.get(b.process_id, self.initial)
        self._exp_estimate[b.process_id] = self.beta * actual + (1 - self.beta) * prev
        self._history.setdefault(b.process_id, []).append((b.feature, math.log(max(actual, 1e-6))))

    def predict(self, b: Burst) -> float:
        raise NotImplementedError


class CWABPPredictor(_WindowedPredictorBase):
    """Correlation-Weighted Adaptive Burst Predictor: blends the regression
    prediction with the exponential-averaging baseline, weighted by shrunk
    squared sample correlation (Eq. 6 of the paper)."""
    name = "cwabp"

    def __init__(self, window: int = 8, shrink: float = 0.5, min_samples: int = 3,
                 beta: float = 0.5, initial: float = 10.0):
        super().__init__(window, min_samples, beta, initial)
        self.shrink = shrink

    def predict(self, b: Burst) -> float:
        hist, exp_est = self._state(b)
        reg_pred, r = _window_regression(hist, self.min_samples, b.feature, exp_est)
        if reg_pred is None:
            return exp_est
        weight = float(np.clip(self.shrink * (r ** 2), 0.0, 1.0))
        blended_log = weight * math.log(max(reg_pred, 1e-6)) + (1 - weight) * math.log(max(exp_est, 1e-6))
        return max(math.exp(blended_log), 1.0)


class RegressionOnlyPredictor(_WindowedPredictorBase):
    """CWABP's window-regression component in isolation: once enough
    history exists, the regression prediction is used outright (weight 1),
    with no correlation-based shrinkage -- this isolates how much of
    CWABP's behaviour comes from "having a regression at all" versus from
    adaptively trusting it only when it has recently been predictive."""
    name = "regression_only"

    def predict(self, b: Burst) -> float:
        hist, exp_est = self._state(b)
        reg_pred, _r = _window_regression(hist, self.min_samples, b.feature, exp_est)
        return exp_est if reg_pred is None else reg_pred


class FixedWeightPredictor(_WindowedPredictorBase):
    """Same blend formula as CWABP (Eq. 6), but with a constant regression
    weight instead of one derived from the observed correlation -- isolates
    whether CWABP's *adaptive* weighting specifically helps, versus just
    blending in some fixed amount of regression unconditionally."""
    name = "fixed_weight"

    def __init__(self, weight: float = 0.5, window: int = 8, min_samples: int = 3,
                 beta: float = 0.5, initial: float = 10.0):
        super().__init__(window, min_samples, beta, initial)
        self.weight = weight

    def predict(self, b: Burst) -> float:
        hist, exp_est = self._state(b)
        reg_pred, _r = _window_regression(hist, self.min_samples, b.feature, exp_est)
        if reg_pred is None:
            return exp_est
        blended_log = self.weight * math.log(max(reg_pred, 1e-6)) + (1 - self.weight) * math.log(max(exp_est, 1e-6))
        return max(math.exp(blended_log), 1.0)


# --------------------------- SRTF simulation --------------------------------

@dataclass
class SRTFResult:
    avg_waiting: float
    avg_turnaround: float
    avg_response: float
    p95_waiting: float
    preemptions: int
    context_switches: int
    mae: float
    rmse: float
    pearson: float
    n_bursts: int


def simulate_srtf(n_processes: int, bursts_per_process: int, rho: float, sigma: float,
                   seed: int, predictor, phase_change: bool = False,
                   correlation_break: bool = False, heavy_tailed: bool = False) -> SRTFResult:
    """Event-driven, preemptive SRTF over a dynamically generated multi-burst,
    alternating CPU/I-O workload. `predictor` must expose .predict(burst) and
    .observe(burst, actual_length).

    A process's next burst is only generated (and only becomes ready) after
    its current burst actually completes in simulated time plus an I/O wait,
    so ready times correctly reflect queueing delay instead of being
    precomputed under an (incorrect) no-contention assumption.
    """
    gens, initial_ready = _make_generators(n_processes, bursts_per_process, rho, sigma,
                                            seed, phase_change, correlation_break, heavy_tailed)

    heap: List[Tuple[float, int]] = []  # (ready_time, pid) -- next burst to materialise
    for pid, rt in enumerate(initial_ready):
        heapq.heappush(heap, (rt, pid))

    time = 0.0
    running: Optional[dict] = None  # {burst, remaining, est}
    ready: List[dict] = []
    completion: Dict[Tuple[int, int], float] = {}
    start: Dict[Tuple[int, int], float] = {}
    ready_time_of: Dict[Tuple[int, int], float] = {}
    true_length_of: Dict[Tuple[int, int], float] = {}
    preds, actuals = [], []
    preemptions = 0
    context_switches = 0
    last_pid: Optional[int] = None
    idle_break = True  # no "previous process" yet, so the first dispatch never counts
    n_expected = n_processes * bursts_per_process

    def key(b: Burst):
        return (b.process_id, b.burst_index)

    def set_running(entry: dict):
        """Assign `running`, counting a context switch whenever this dispatch
        follows a *different* process with no intervening idle interval --
        the same convention used by the main priority/RR engine."""
        nonlocal running, last_pid, idle_break, context_switches
        pid = entry["burst"].process_id
        if not idle_break and last_pid is not None and pid != last_pid:
            context_switches += 1
        running = entry
        start.setdefault(key(entry["burst"]), time)
        last_pid = pid
        idle_break = False

    def admit(entry: dict):
        """Insert a freshly materialised burst into running/ready, preempting
        the current running burst if its estimate is strictly smaller."""
        nonlocal preemptions
        if running is None:
            set_running(entry)
        elif entry["est"] < running["est"] - 1e-9:
            preemptions += 1
            ready.append(running)
            set_running(entry)
        else:
            ready.append(entry)

    def materialise_due(t: float):
        while heap and heap[0][0] <= t + 1e-9:
            rt, pid = heapq.heappop(heap)
            b = gens[pid].next_burst(rt)
            if b is None:
                continue
            est = predictor.predict(b)
            preds.append(est)
            actuals.append(b.true_length)
            k = key(b)
            ready_time_of[k] = b.ready_time
            true_length_of[k] = b.true_length
            admit(dict(burst=b, remaining=b.true_length, est=est))

    while len(completion) < n_expected:
        if running is None:
            if ready:
                ready.sort(key=lambda e: (e["est"], e["burst"].ready_time, e["burst"].process_id))
                set_running(ready.pop(0))
            elif heap:
                time = heap[0][0]
                idle_break = True
                materialise_due(time)
                continue
            else:
                break

        next_event = heap[0][0] if heap else math.inf
        step = min(running["remaining"], max(next_event - time, 1e-6))
        time += step
        running["remaining"] -= step

        if running["remaining"] <= 1e-6:
            b = running["burst"]
            completion[key(b)] = time
            predictor.observe(b, b.true_length)
            gen = gens[b.process_id]
            if gen.k < gen.bursts_per_process:
                heapq.heappush(heap, (time + gen.io_wait(), b.process_id))
            running = None

        materialise_due(time)

    waiting, turnaround, response = [], [], []
    for k, c in completion.items():
        rt = ready_time_of[k]
        t_len = true_length_of[k]
        s = start[k]
        t = c - rt
        w = t - t_len
        r = s - rt
        turnaround.append(t)
        waiting.append(w)
        response.append(r)

    waiting = np.array(waiting)
    preds_a, actuals_a = np.array(preds), np.array(actuals)
    mae = float(np.mean(np.abs(preds_a - actuals_a)))
    rmse = float(np.sqrt(np.mean((preds_a - actuals_a) ** 2)))
    pearson = float(np.corrcoef(preds_a, actuals_a)[0, 1]) if len(preds_a) > 1 else 0.0

    return SRTFResult(
        avg_waiting=float(np.mean(waiting)),
        avg_turnaround=float(np.mean(turnaround)),
        avg_response=float(np.mean(response)),
        p95_waiting=float(np.percentile(waiting, 95, method="linear")),
        preemptions=preemptions,
        context_switches=context_switches,
        mae=mae, rmse=rmse, pearson=pearson,
        n_bursts=len(preds),
    )
