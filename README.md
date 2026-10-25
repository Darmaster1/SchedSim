# schedsim

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

## Running the tests

```bash
pip install pytest hypothesis
# Run the 40 original unit & invariant tests
pytest tests/test_*.py --ignore=tests/test_properties.py -v

# Run the full test suite including randomized property-based tests (Hypothesis)
pytest tests/ -v
```

40 unit tests + 18 randomized property tests cover:
- **Property-based randomized invariants (Hypothesis)** -- thousands of randomized
  workloads test structural invariants: turnaround = waiting + burst decomposition across
  Priority and RR engines, B(α=1) ≡ A identity, C with safeguards disabled ≡ B,
  no priority inversion outside override when gap=0, Jain's index ∈ [1/n, 1],
  starvation count ∈ [0, n], Q-state discretisation ∈ [0, 144), and deterministic frozen policy.
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

## Continuous Training & Data Flywheel (`train.py`)

To continuously generate randomized workloads, accumulate training data, and progressively improve models:

```bash
# Run 1 continuous round (generate random workloads -> pool -> retrain -> eval)
python train.py run

# Run multiple rounds with custom batch sizes
python train.py run --rounds 5 --workloads 100 --q-episodes 1000

# View current pool size & trained checkpoints
python train.py status

# Inspect model leaderboard (checkpoints ranked by validation RMSE)
python train.py leaderboard

# Evaluate current checkpoint against the frozen benchmark test set
python train.py evaluate

# Reset data pool and checkpoints to try fresh training sessions:
python train.py reset -f              # Reset everything without prompt
python train.py reset --pool-only     # Reset training data pool only
python train.py reset --models-only   # Reset model checkpoints & leaderboard only
```

## Repository layout

```
schedsim/
  workload.py        Synthetic workload generation (Section 6.2 of the paper)
  random_spec.py     Combinatorial WorkloadSpec sampler for continuous random workloads
  predictors.py      Five-model burst-length prediction + leakage control (7.1)
  schedulers.py      Discrete-event engines + Algorithms A-E (7.2-7.6)
  qlearning.py       Tabular Q-learning for Algorithm F (Section 8)
  cwabp.py           CWABP predictor + event-driven SRTF simulator (Section 9)
  metrics.py         Per-workload evaluation metrics (Section 11)
  experiment.py      Pipeline orchestration: tuning, training, evaluation
  stats.py           Paired Wilcoxon tests + bootstrap CIs (Section 10.2)
  pool.py            Persistent data pool for continuous predictor retraining
  checkpoint.py      Checkpointing for predictors and Q-tables + leaderboard
  continuous.py      Continuous training loop orchestration
tests/               Unit tests + randomized property tests (Hypothesis)
run_experiment.py    CLI entry point for reproducing paper pilot numbers
train.py             CLI entry point for continuous random training & data flywheel
paper/               IEEE-format paper source (LaTeX) and compiled PDF
results/             Output of the most recent run (gitignored except structure)
data/                Persistent training pool storage
checkpoints/         Model checkpoints and versioned training log
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
