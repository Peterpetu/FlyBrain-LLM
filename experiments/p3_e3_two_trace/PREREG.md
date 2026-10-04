# PREREG — P3-E3: a learned readout over two traces, with and without the confirmation count

*Module level (docs/phase3/EXPERIMENT_PLAN.md). Frozen at the first run. Written after E2's null
and before any E3 run; the E2 numbers that motivate it are quoted here so the reader can judge the
inference.*

**Why this experiment.** E2 found that a learned write gate on ONE store does not use the admitted
per-address states (+0.0001, +0.3 seed-SD). Its per-class results showed the trade-off every
single-store gate faces: on outlier keys a low fixed rate is best (0.78; learned gates 0.58), on
re-bound keys a high one (0.70; learned gates 0.51). The decision that matters, adopt a deviating
observation after it has been repeated and protect against it when it has not, needs the previous
observation, which one store does not hold. The two-trace store holds it (labile trace at rate 1,
slow trace at 0.2) and was the best mechanism on outlier keys (0.66) but the worst overall (0.850)
because its readout is a FIXED rule (trust the labile trace after one content-based confirmation).
E3 applies the doctrine to that substrate: learn the readout and the per-trace write rates, and ask
whether the confirmation count (and the contradiction masses) add anything to what the readout can
compute from the two traces themselves (their agreement and norms).

**Hypotheses.**
- **K0-TT (primary).** On the two-trace substrate, a learned readout and learned per-trace gates
  given the per-address states (nconf, zc, a) as slot inputs beat the SAME architecture with the
  slots zeroed (equal parameters) AND a state-matched single store with a learned gate.
- **Substrate (secondary, reported whatever the verdict).** The two-trace substrate with a learned
  readout beats the state-matched single store even without the states.

**Task.** As E2 (`flyprinc.tasks.rebind_stream`, 48 keys, T = 1,500, four regimes sigma in
{0.3, 1.0} x p_outlier in {0, 0.1}, 4 held-out streams of 8 sequences per regime, fixed seed).

**Conditions.**
- `tt_learned_state`: 2 compartments (1,024 cells, 64 active, delta writes), learned per-trace
  write gates (MLP on key, value, error, log squared error, log read norm, slots nconf / zc / a),
  learned readout (MLP on cos(slow, labile), the two log norms, the same slots).
- `tt_learned_free`: identical, slots read zero in the gate and in the readout. THE PRIMARY CONTROL.
- `single_free_wide`: one compartment with 2,048 cells and 64 active (state matched to the two
  traces: 2,048 x 16 = 2 x 1,024 x 16; compute matched by active rows), learned gate, slots zero.
  It has no readout MLP, so 178 fewer parameters than the two-trace arms (3,329 vs 3,507); the
  nested K0-TT comparison (`tt_learned_state` vs `tt_learned_free`) is at equal parameters.
- `learned_free`: E2's single store (1,024 cells), for reference.
- `two_trace`: E2's fixed two-trace mechanism (rates 0.2 / 1.0, confirmation readout), for reference.
Trained conditions: 80 Adam steps on fresh T = 600 streams cycling through the four regimes,
learning rates {1e-3, 3e-3, 1e-2}, best by seed-mean of the primary metric. Seeds {0..4}.

**Metric.** Primary: `acc_classes`, the mean over the four regimes of the equal-weighted mean of
the per-class accuracies (stable / re-bound / outlier where present). E2's primary (the natural
mix, `acc_mean`) is secondary here because the classes where the mechanisms differ are minorities
of it. Per-regime and per-class accuracies are reported.

**Thresholds (exhaustive).**
- K0-TT SUPPORTED iff `tt_learned_state` beats `tt_learned_free` on `acc_classes` by > 2 seed-SD
  with the bootstrap CI excluding 0 AND beats `single_free_wide` by > 2 seed-SD with CI excluding 0.
  Otherwise FALSIFIED (a state that a learned readout over two traces does not use either).
- Substrate: `tt_learned_free` vs `single_free_wide` by the same rule; reported as a finding about
  the substrate, not about the states.
- Positive verdicts are replicated with `--variant replication --seeds 5 6 7 8 9 10` before being
  reported as supported.
