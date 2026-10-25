"""
Per-workload evaluation metrics (Section 11 of the paper).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .schedulers import EventLog
from .workload import Process

STARVATION_THRESHOLD = 100.0


@dataclass
class WorkloadMetrics:
    avg_waiting: float
    avg_turnaround: float
    avg_response: float
    max_waiting: float
    p95_waiting: float
    context_switches: int
    utilization: float
    makespan: float
    starvation_violations: int
    jain_index: float


def compute_metrics(procs: List[Process], log: EventLog) -> WorkloadMetrics:
    by_pid = {p.pid: p for p in procs}
    waiting, turnaround, response, slowdown = [], [], [], []

    for p in procs:
        c = log.completion_time[p.pid]
        s = log.start_time[p.pid]
        t = c - p.arrival
        w = t - p.true_burst
        r = s - p.arrival
        waiting.append(w)
        turnaround.append(t)
        response.append(r)
        slowdown.append(max(t, 1e-6) / max(p.true_burst, 1))

    waiting = np.array(waiting)
    turnaround = np.array(turnaround)
    response = np.array(response)
    slowdown = np.array(slowdown)

    total_burst = sum(p.true_burst for p in procs)
    first_arrival = min(p.arrival for p in procs)
    last_completion = max(log.completion_time.values())
    elapsed = max(last_completion - first_arrival, 1e-9)
    utilization = total_burst / elapsed

    jain = (slowdown.sum() ** 2) / (len(slowdown) * (slowdown ** 2).sum() + 1e-12)

    return WorkloadMetrics(
        avg_waiting=float(waiting.mean()),
        avg_turnaround=float(turnaround.mean()),
        avg_response=float(response.mean()),
        max_waiting=float(waiting.max()),
        p95_waiting=float(np.percentile(waiting, 95, method="linear")),
        context_switches=log.context_switches,
        utilization=float(utilization),
        makespan=float(elapsed),
        starvation_violations=int(np.sum(waiting > STARVATION_THRESHOLD)),
        jain_index=float(jain),
    )


def metrics_to_dict(m: WorkloadMetrics) -> Dict[str, float]:
    return {
        "avg_waiting": m.avg_waiting,
        "avg_turnaround": m.avg_turnaround,
        "avg_response": m.avg_response,
        "max_waiting": m.max_waiting,
        "p95_waiting": m.p95_waiting,
        "context_switches": m.context_switches,
        "utilization": m.utilization,
        "makespan": m.makespan,
        "starvation_violations": m.starvation_violations,
        "jain_index": m.jain_index,
    }
