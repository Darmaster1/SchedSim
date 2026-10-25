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


class CWABPPredictor:
    name = "cwabp"

    def __init__(self, window: int = 8, shrink: float = 0.5, min_samples: int = 3,
                 beta: float = 0.5, initial: float = 10.0):
        self.window = window
        self.shrink = shrink
        self.min_samples = min_samples
        self.beta = beta
        self.initial = initial
        self._history: Dict[int, List[Tuple[float, float]]] = {}  # pid -> [(x, log_y)]
        self._exp_estimate: Dict[int, float] = {}

    def predict(self, b: Burst) -> float:
        hist = self._history.get(b.process_id, [])[-self.window:]
        exp_est = self._exp_estimate.get(b.process_id, self.initial)

        if len(hist) < self.min_samples:
            return exp_est

        xs = np.array([h[0] for h in hist])
        ys = np.array([h[1] for h in hist])
        var_x, var_y = xs.var(), ys.var()
        if var_x < 1e-9 or var_y < 1e-9:
            r = 0.0
        else:
            r = float(np.corrcoef(xs, ys)[0, 1])

        weight = float(np.clip(self.shrink * (r ** 2), 0.0, 1.0))

        if abs(r) > 1e-9 and var_x > 1e-9:
            slope = np.cov(xs, ys, bias=True)[0, 1] / var_x
            intercept = ys.mean() - slope * xs.mean()
            # Trust-region clamp: with a small window, a spurious high sample
            # correlation can combine with an out-of-sample `feature` value
            # to extrapolate the regression component to an absurd (or even
            # exp()-overflowing) length. Clamp the LOG-space argument first
            # (so we never overflow math.exp itself), then clip the result
            # to within a factor of 5x the exponential-averaging baseline,
            # which keeps a high weight's effect meaningful without letting
            # a single noisy window dominate the estimate.
            log_exp_est = math.log(max(exp_est, 1e-6))
            raw_log_pred = intercept + slope * b.feature
            clamped_log_pred = float(np.clip(raw_log_pred, log_exp_est - math.log(5.0),
                                              log_exp_est + math.log(5.0)))
            reg_pred = math.exp(clamped_log_pred)
        else:
            reg_pred = exp_est

        blended_log = weight * math.log(max(reg_pred, 1e-6)) + (1 - weight) * math.log(max(exp_est, 1e-6))
        return max(math.exp(blended_log), 1.0)

    def observe(self, b: Burst, actual: float):
        prev = self._exp_estimate.get(b.process_id, self.initial)
        self._exp_estimate[b.process_id] = self.beta * actual + (1 - self.beta) * prev
        self._history.setdefault(b.process_id, []).append((b.feature, math.log(max(actual, 1e-6))))


# --------------------------- SRTF simulation --------------------------------

@dataclass
class SRTFResult:
    avg_waiting: float
    avg_turnaround: float
    avg_response: float
    p95_waiting: float
    preemptions: int
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
    n_expected = n_processes * bursts_per_process

    def key(b: Burst):
        return (b.process_id, b.burst_index)

    def admit(entry: dict):
        """Insert a freshly materialised burst into running/ready, preempting
        the current running burst if its estimate is strictly smaller."""
        nonlocal running, preemptions
        if running is None:
            running = entry
            start[key(entry["burst"])] = time
        elif entry["est"] < running["est"] - 1e-9:
            preemptions += 1
            ready.append(running)
            running = entry
            start[key(entry["burst"])] = time
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
                running = ready.pop(0)
                start.setdefault(key(running["burst"]), time)
            elif heap:
                time = heap[0][0]
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
        mae=mae, rmse=rmse, pearson=pearson,
        n_bursts=len(preds),
    )
