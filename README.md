# schedsim

**Current focused study:** `paper/paper.pdf` is *"Correlation-Guided CPU
Burst Prediction for SRTF Scheduling Under Changing Workloads"* -- a
narrowly-scoped, statistically rigorous comparison of CWABP against three
fair baselines (exponential averaging, regression-only, fixed-weight) on
SRTF scheduling. Run it with:
```bash
python cwabp_study.py                 # validation sweep + frozen test run
pytest tests/test_cwabp.py tests/test_worked_example.py tests/test_properties.py -v
```
This repository also contains a broader, separately-scoped six-algorithm
scheduling study (Priority Scheduling, Round Robin, Q-learning) described
below -- the two studies share the simulator's core but are evaluated and
reported independently; the current paper does not draw on the six-algorithm
results.


A discrete-event simulator comparing **ML-assisted priority scheduling** and
**RL-based adaptive Round Robin** against classical baselines, plus a
**correlation-weighted burst predictor (CWABP)** extension for SRTF
scheduling. This is the reference implementation accompanying the paper
*"Machine Learning-Assisted Priority Scheduling and Reinforcement
Learning-Based Adaptive Round Robin"* (`paper/paper.pdf`).

Six scheduling algorithms are implemented and compared on identical
synthetic workloads under identical counting conventions:

| # | Algorithm | Family | Idea |
|---|-----------|--------|------|
| A | Priority Scheduling | Priority (non-preemptive) | Classical baseline |
| B | Weighted ML Priority | Priority | Blends priority with a predicted burst length |
| C | Protected ML Priority | Priority | B + anti-starvation safeguards |
| D | Fixed-Quantum RR | Round Robin (preemptive) | Classical baseline |
| E | Rule-Based Adaptive RR | Round Robin | Hand-written quantum-selection rules |
| F | Q-Learning Adaptive RR | Round Robin | Quantum chosen by a trained Q-learning agent |

A seventh, independent component (`schedsim/cwabp.py`, Section 9 of the
paper) proposes a **Correlation-Weighted Adaptive Burst Predictor** for SRTF
scheduling and compares it against classical exponential averaging on a
separate multi-burst, I/O-alternating process model.

## Installation

```bash
git clone <this-repo-url>
cd schedsim
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Requires Python 3.9+.

## Running the experiments

```bash
# Smoke test (~2 seconds) -- confirms everything runs end to end
python run_experiment.py --quick

# Default pilot-scale run (~15-25 seconds) -- what the paper's numbers use
python run_experiment.py

# Larger-scale run closer to the full protocol described in the paper
# (bigger workload samples, bigger Q-learning training budget; several minutes)
python run_experiment.py --full

# Only the CWABP (Section 9) extension
python run_experiment.py --cwabp-only
```

Each run writes to `results/`:
- `per_workload_results.csv` -- every metric, for every algorithm, for every test workload
- `summary.json` -- metrics averaged across test workloads, per algorithm
- `statistical_tests.json` -- paired Wilcoxon signed-rank tests (with Holm
  correction) and bootstrap confidence intervals for the six head-to-head
  comparisons used in the paper (B vs A, C vs B, C vs A, E vs D, F vs D, F vs E)
- `cwabp_results.csv` -- CWABP vs. exponential averaging across 7 workload types

## Continuous training (data flywheel)

A separate CLI keeps the burst predictors and the Q-learning agent improving
across repeated rounds instead of being fit once on a fixed batch:

```bash
python train.py run --rounds 5 --workloads 50          # grow the pool, refit, resume Q-learning
python train.py run --rounds 20 --workloads 100 --episodes-per-round 2000
python train.py status                                  # pool size, leaderboard, Q-table checkpoint
```

Each round: samples new workloads into an append-only pool
(`training_pool.csv`, seeds strictly disjoint from the evaluation splits),
refits all five regression models on the accumulated pool and logs a
validation-RMSE leaderboard (`checkpoints/leaderboard.csv`), atomically
promotes the best model to `checkpoints/predictor_latest.pkl`, and resumes
Q-learning from `checkpoints/qtable_latest.npz`, continuing its
$\varepsilon$-decay schedule from the cumulative episode count rather than
restarting.

## Running the tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

52 tests: 42 unit/invariant tests plus `test_properties.py`, 10
Hypothesis property-based suites that re-check the same invariants over
randomly generated and automatically shrunk inputs instead of hand-picked
cases. Together they cover:
- **Determinism & leakage control** -- identical (spec, seed) always produces
  identical workloads; burst predictors never see true burst length,
  priority, or arrival time as inputs.
- **Algorithm identities** -- Algorithm B with `alpha=1` is pid-for-pid
  identical to Algorithm A; Algorithm C with safeguards disabled is
  pid-for-pid identical to Algorithm B; with `gap=0`, no priority inversion
  occurs outside an explicit override event.
- **Round Robin mechanics** -- FIFO tie-breaking on simultaneous arrival,
  every process completes, quantum choices stay within the candidate set.
- **Q-learning** -- state discretisation always lands in `[0, 144)`, a frozen
  policy is fully deterministic (no residual exploration at evaluation time).
- **CWABP** -- no future information leaks into a prediction; the method
  falls back cleanly to exponential averaging with too little history; a
  trust-region clamp keeps the regression component from exploding under a
  spuriously high small-window correlation (this test originally caught a
  real `OverflowError` during development -- see `schedsim/cwabp.py`).
- **Metrics** -- turnaround = waiting + burst length; Jain's index stays
  within `[1/n, 1]`; starvation-violation counts stay within `[0, n]`.

## Repository layout

```
schedsim/
  workload.py      Synthetic workload generation
  predictors.py     Five-model burst-length prediction + leakage control
  schedulers.py     Discrete-event engines + Algorithms A-E + reward function
  qlearning.py       Tabular Q-learning for Algorithm F (resumable)
  cwabp.py            CWABP predictor + event-driven SRTF simulator
  pool.py              Append-only training pool for the continuous flywheel
  checkpoint.py         Model refitting, leaderboard, atomic checkpointing
  metrics.py             Per-workload evaluation metrics
  experiment.py            Pipeline orchestration: random-search tuning, training, evaluation
  stats.py                  Paired Wilcoxon tests + bootstrap CIs
tests/                      52 tests: unit/invariant tests + test_properties.py (Hypothesis)
run_experiment.py           CLI: one-shot pilot/full/deep experiment runs
train.py                    CLI: continuous training flywheel (pool + checkpoint + resumable RL)
paper/                      IEEE-format paper source (LaTeX) and compiled PDF
results/                    Output of the most recent run (gitignored except structure)
```

## A note on scale

The default configuration here is a **genuine but intentionally small-scale**
instance of the protocol described in the paper: reduced sample sizes per
workload category (3 train / 3 validation / 5 test instead of 20/8/20), a
smaller Q-learning training budget (600 episodes instead of 4,000 x 5 seeds),
and a coarser tuning grid for Algorithm E. This keeps the full pipeline
runnable in well under a minute, which matters for reproducibility and for
actually being able to check the code works. All reported numbers in the
paper are real output from this exact code, not placeholders -- but they
should be read as a **pilot study**, not a final large-scale result. Use
`--full` to scale the sample sizes and training budget up; the code path is
identical, only the constants change.

## License

MIT. See `LICENSE`.
