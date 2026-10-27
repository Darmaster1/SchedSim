"""
Scheduling engines and Algorithms A-E (Section 7 of the paper).

Two discrete-event engines are implemented:
  * `run_priority_family`  -- non-preemptive, used by Algorithms A, B, C.
  * `run_round_robin_family` -- preemptive FIFO RR, used by Algorithms D, E, F.

Both engines produce an EventLog (list of dispatch intervals) plus a dict of
algorithm-specific extra statistics (inversions, activations, quantum
choices, ...). Algorithm F (Q-learning) is driven through the same RR
engine via a pluggable `QuantumPolicy`; see qlearning.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol

import numpy as np

from .workload import Process

QUANTA = (2, 4, 8, 16)  # the four candidate quantum sizes q0..q3


@dataclass
class DispatchEvent:
    pid: int
    start: float
    end: float
    is_context_switch: bool


@dataclass
class EventLog:
    events: List[DispatchEvent] = field(default_factory=list)
    idle_time: float = 0.0
    start_time: Dict[int, float] = field(default_factory=dict)
    completion_time: Dict[int, float] = field(default_factory=dict)
    context_switches: int = 0


# ---------------------------------------------------------------------------
# Priority family: Algorithms A, B, C
# ---------------------------------------------------------------------------

@dataclass
class PriorityPolicyResult:
    """What a priority-family policy returns at a decision point."""
    pid: int
    used_override: bool = False


class PriorityPolicy(Protocol):
    def decide(self, ready: List[Process], predicted: Dict[int, float],
               wait_so_far: Dict[int, float], time: float) -> PriorityPolicyResult: ...


def _normalize(values: Dict[int, float]) -> Dict[int, float]:
    vals = list(values.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def score_ready_set(ready: List[Process], predicted: Dict[int, float], alpha: float
                     ) -> Dict[int, float]:
    """Equation 3-4: min-max normalised blend of priority and predicted burst."""
    prio = {p.pid: p.priority for p in ready}
    pred = {p.pid: predicted[p.pid] for p in ready}
    p_hat = _normalize(prio)
    b_hat = _normalize(pred)
    return {pid: alpha * p_hat[pid] + (1 - alpha) * b_hat[pid] for pid in p_hat}


class AlgorithmA_Priority:
    """Traditional non-preemptive Priority Scheduling (baseline)."""
    name = "A"

    def decide(self, ready, predicted, wait_so_far, time):
        best = min(ready, key=lambda p: (p.priority, p.arrival, p.pid))
        return PriorityPolicyResult(pid=best.pid)


class AlgorithmB_WeightedML:
    """Weighted ML Priority Scheduling (Equation 3-4)."""
    name = "B"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def decide(self, ready, predicted, wait_so_far, time):
        scores = score_ready_set(ready, predicted, self.alpha)
        by_pid = {p.pid: p for p in ready}
        best_pid = min(scores, key=lambda pid: (
            scores[pid], by_pid[pid].priority, by_pid[pid].arrival, pid))
        return PriorityPolicyResult(pid=best_pid)


class AlgorithmC_ProtectedML:
    """Protected ML Priority Scheduling: max-wait override + bounded inversion."""
    name = "C"

    def __init__(self, alpha: float = 0.5, w_max: float = 100.0, gap: int = 3):
        self.alpha = alpha
        self.w_max = w_max
        self.gap = gap
        self.activations = 0
        self.effective_activations = 0
        self.inversions = 0

    def decide(self, ready, predicted, wait_so_far, time):
        by_pid = {p.pid: p for p in ready}
        scores = score_ready_set(ready, predicted, self.alpha)

        # Unprotected (Algorithm B) choice, for "effective activation" bookkeeping.
        unprotected_best = min(scores, key=lambda pid: (
            scores[pid], by_pid[pid].priority, by_pid[pid].arrival, pid))

        overdue = [p for p in ready if wait_so_far[p.pid] >= self.w_max]
        if overdue:
            self.activations += 1
            chosen = min(overdue, key=lambda p: (p.arrival, p.pid))
            if chosen.pid != unprotected_best:
                self.effective_activations += 1
            if chosen.priority > by_pid[unprotected_best].priority:
                self.inversions += 1
            return PriorityPolicyResult(pid=chosen.pid, used_override=True)

        best_priority = min(p.priority for p in ready)
        eligible = [p for p in ready if p.priority - best_priority <= self.gap]
        elig_pids = {p.pid for p in eligible}
        chosen_pid = min((pid for pid in scores if pid in elig_pids),
                          key=lambda pid: (scores[pid], by_pid[pid].priority,
                                            by_pid[pid].arrival, pid))
        if by_pid[chosen_pid].priority > best_priority:
            self.inversions += 1
        return PriorityPolicyResult(pid=chosen_pid)


def run_priority_family(procs: List[Process], policy: PriorityPolicy,
                         predicted: Dict[int, float], context_switch_cost: float = 1.0
                         ) -> EventLog:
    procs_sorted = sorted(procs, key=lambda p: (p.arrival, p.pid))
    remaining = {p.pid: p.true_burst for p in procs}
    by_pid = {p.pid: p for p in procs}
    arrival_idx = 0
    n = len(procs_sorted)
    log = EventLog()
    ready: List[Process] = []
    time = 0.0
    last_pid: Optional[int] = None

    def admit_arrivals(up_to_time: float):
        nonlocal arrival_idx
        while arrival_idx < n and procs_sorted[arrival_idx].arrival <= up_to_time + 1e-9:
            ready.append(procs_sorted[arrival_idx])
            arrival_idx += 1

    admit_arrivals(time)
    inversion_set_tracker = []

    while len(log.completion_time) < n:
        if not ready:
            next_arrival = procs_sorted[arrival_idx].arrival
            log.idle_time += max(0.0, next_arrival - time)
            time = max(time, next_arrival)
            admit_arrivals(time)
            last_pid = None  # idle breaks the context-switch chain
            continue

        wait_so_far = {p.pid: time - p.arrival for p in ready}
        result = policy.decide(ready, predicted, wait_so_far, time)
        pid = result.pid
        proc = by_pid[pid]
        ready.remove(proc)

        is_switch = last_pid is not None and last_pid != pid
        if is_switch:
            time += context_switch_cost
            log.context_switches += 1

        if pid not in log.start_time:
            log.start_time[pid] = time

        duration = remaining[pid]
        start, end = time, time + duration
        admit_during = end
        time = end
        remaining[pid] = 0
        log.completion_time[pid] = time
        admit_arrivals(admit_during)
        log.events.append(DispatchEvent(pid, start, end, is_switch))
        last_pid = pid

    return log


# ---------------------------------------------------------------------------
# Round Robin family: Algorithms D, E, F
# ---------------------------------------------------------------------------

class QuantumPolicy(Protocol):
    def choose_quantum(self, queue_len: int, mean_pred_remaining: float,
                        max_wait: float, imbalance: float,
                        own_wait: float) -> int: ...


# Reference constants and default ("balanced") weights for the Q-learning
# reward (Section 8.2 / Eq. 7): scale each term to roughly order one for a
# typical event, as modelling choices rather than derived quantities.
REWARD_REF = dict(w_ref=160.0, r_ref=50.0, s_ref=10.0, s_max=50.0, w_fair=100.0)
DEFAULT_REWARD_WEIGHTS = dict(w1=1.0, w2=0.5, w3=0.5, w4=0.2, w5=0.2)


def compute_reward(delta_wait: float, response_first: Optional[float],
                    slowdown_at_completion: Optional[float], is_switch: bool,
                    max_wait_tail: float, weights: Optional[dict] = None) -> float:
    """Five-term reward (Eq. 7): waiting-time increase, first-response
    latency, completion slowdown, a context-switch penalty, and a tail-wait
    penalty once the longest queued wait exceeds w_fair. `response_first`
    and `slowdown_at_completion` are None on steps where that event (first
    dispatch / completion) doesn't occur this step, contributing zero."""
    w = weights or DEFAULT_REWARD_WEIGHTS
    ref = REWARD_REF
    term1 = w["w1"] * delta_wait / ref["w_ref"]
    term2 = w["w2"] * (response_first / ref["r_ref"]) if response_first is not None else 0.0
    if slowdown_at_completion is not None:
        excess = min(max(slowdown_at_completion - 1.0, 0.0), ref["s_max"])
        term3 = w["w3"] * excess / ref["s_ref"]
    else:
        term3 = 0.0
    term4 = w["w4"] * (1.0 if is_switch else 0.0)
    tail = float(np.clip((max_wait_tail - ref["w_fair"]) / ref["w_fair"], 0.0, 1.0))
    term5 = w["w5"] * tail
    return -(term1 + term2 + term3 + term4 + term5)


@dataclass
class FixedQuantumPolicy:
    q: float = 4.0

    def choose_quantum(self, queue_len, mean_pred_remaining, max_wait, imbalance, own_wait):
        return self.q


@dataclass
class RuleBasedQuantumPolicy:
    """Algorithm E: shrink the quantum when predicted work/queue length is
    high, grow it when the dispatched process has waited a long time."""
    queue_thresh: int = 6
    median_thresh: float = 20.0
    wait_thresh: float = 30.0
    base_idx: int = 1

    def choose_quantum(self, queue_len, mean_pred_remaining, max_wait, imbalance, own_wait):
        idx = self.base_idx
        if mean_pred_remaining > self.median_thresh or queue_len > self.queue_thresh:
            idx -= 1
        if own_wait > self.wait_thresh:
            idx += 1
        idx = int(np.clip(idx, 0, len(QUANTA) - 1))
        return QUANTA[idx]


def run_round_robin_family(procs: List[Process], policy: QuantumPolicy,
                            predicted: Dict[int, float], context_switch_cost: float = 1.0,
                            step_callback: Optional[Callable] = None,
                            reward_weights: Optional[dict] = None) -> EventLog:
    """
    step_callback(state_dict, quantum_idx, reward, next_state_dict, done) is
    invoked after every dispatch step, so that Algorithm F's Q-learning agent
    can observe transitions and update online without a second engine.
    """
    procs_sorted = sorted(procs, key=lambda p: (p.arrival, p.pid))
    by_pid = {p.pid: p for p in procs}
    remaining = {p.pid: p.true_burst for p in procs}
    executed = {p.pid: 0.0 for p in procs}
    ready_since = {p.pid: p.arrival for p in procs}
    n = len(procs_sorted)
    arrival_idx = 0
    queue: List[int] = []
    log = EventLog()
    time = 0.0
    last_pid: Optional[int] = None
    total_wait_prev = 0.0

    def admit_arrivals(up_to_time: float):
        nonlocal arrival_idx
        while arrival_idx < n and procs_sorted[arrival_idx].arrival <= up_to_time + 1e-9:
            queue.append(procs_sorted[arrival_idx].pid)
            ready_since[procs_sorted[arrival_idx].pid] = procs_sorted[arrival_idx].arrival
            arrival_idx += 1

    def pred_remaining(pid: int) -> float:
        return max(predicted[pid] - executed[pid], 1.0)

    def current_total_wait() -> float:
        return sum(time - ready_since[p] for p in queue)

    admit_arrivals(time)

    while len(log.completion_time) < n:
        if not queue:
            next_arrival = procs_sorted[arrival_idx].arrival
            log.idle_time += max(0.0, next_arrival - time)
            time = max(time, next_arrival)
            admit_arrivals(time)
            last_pid = None
            continue

        pid = queue.pop(0)
        proc = by_pid[pid]

        queue_len = len(queue) + 1
        ready_pids = [pid] + queue
        mean_pred_remaining = float(np.mean([pred_remaining(q) for q in ready_pids]))
        max_pred_remaining = float(np.max([pred_remaining(q) for q in ready_pids]))
        imbalance = max_pred_remaining / max(mean_pred_remaining, 1e-6)
        own_wait = time - ready_since[pid]
        max_wait_others = max([time - ready_since[q] for q in queue], default=0.0)

        state = dict(queue_len=queue_len, mean_pred_remaining=mean_pred_remaining,
                      max_wait=max(own_wait, max_wait_others), imbalance=imbalance,
                      own_wait=own_wait)

        if hasattr(policy, "choose_quantum_idx"):
            q_idx = policy.choose_quantum_idx(**state)
            q = QUANTA[q_idx]
        else:
            q = policy.choose_quantum(**state)
            q_idx = min(range(len(QUANTA)), key=lambda i: abs(QUANTA[i] - q))

        same_process_redispatch = (last_pid == pid and queue_len == 1
                                    and executed[pid] > 0)
        is_switch = (last_pid is not None and last_pid != pid)
        if is_switch:
            time += context_switch_cost
            log.context_switches += 1

        is_first_dispatch = pid not in log.start_time
        if is_first_dispatch:
            log.start_time[pid] = time

        run_time = min(q, remaining[pid])
        start = time
        end = time + run_time
        wait_before = current_total_wait()

        # enqueue every arrival up to and including the end of this slice
        # BEFORE the preempted process (if any) is re-enqueued
        admit_arrivals(end)
        time = end
        remaining[pid] -= run_time
        executed[pid] += run_time

        completed_this_step = remaining[pid] <= 1e-9
        if not completed_this_step:
            queue.append(pid)
            ready_since[pid] = time
        else:
            log.completion_time[pid] = time

        wait_after = current_total_wait()
        delta_wait = wait_after - wait_before
        response_first = (start - proc.arrival) if is_first_dispatch else None
        slowdown_completion = ((time - proc.arrival) / max(proc.true_burst, 1)) if completed_this_step else None
        reward = compute_reward(delta_wait, response_first, slowdown_completion, is_switch,
                                 state["max_wait"], reward_weights)
        done = len(log.completion_time) == n

        log.events.append(DispatchEvent(pid, start, end, is_switch))
        last_pid = pid

        if step_callback is not None:
            next_state = None
            if queue:
                npid = queue[0]
                nq_len = len(queue)
                n_ready = [npid] + queue[1:]
                n_mean_pred = float(np.mean([pred_remaining(q) for q in n_ready]))
                n_max_pred = float(np.max([pred_remaining(q) for q in n_ready]))
                n_imbalance = n_max_pred / max(n_mean_pred, 1e-6)
                n_own_wait = time - ready_since[npid]
                n_max_wait_others = max([time - ready_since[q] for q in queue[1:]], default=0.0)
                next_state = dict(queue_len=nq_len, mean_pred_remaining=n_mean_pred,
                                   max_wait=max(n_own_wait, n_max_wait_others),
                                   imbalance=n_imbalance, own_wait=n_own_wait)
            step_callback(state, q_idx, reward, next_state, done)

    return log
