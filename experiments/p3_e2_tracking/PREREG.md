# PREREG — P3-E2 (v2): does a per-address state improve a learned write gate?

*Module level (docs/phase3/EXPERIMENT_PLAN.md). Frozen at the first run. v1 of this file was never
run; it was revised before freezing after the design panel's doctrine ("do not propose a gate,
propose a state", docs/phase3/DESIGN_PANEL.md) and the E0 signal-availability result
(experiments/p3_e0_signal/SUMMARY.md), which admitted three states. v1 compared fixed rules with a
learned gate, the comparison that the phase-1 and phase-2 nulls already answered: a fixed rule lies
inside the learned gate's hypothesis class. v2 asks the nested question instead.*

**Hypotheses.**
- **K0 (state in the gate; primary).** A learned write gate given the E0-admitted per-address
  STATES as extra inputs — the content-based confirmation count `nconf`, the contradiction mass `a`
  (relu(Z . k - rho sigma), ATTEST's evidence channel) and its familiarity-gated variant `zc` —
  tracks a non-stationary noisy stream with outliers better than the SAME gate without them (same
  architecture and initialisation, equal parameter count, the state slots read zero), because the
  adoption-versus-outlier decision at an address depends on that address's history, which the delta
  rule's own intermediates (error, read norm) carry only in part (E0: fully at beta < 1 without
  noise, not at all at beta = 1).
- **P3 (rate as inferred precision; secondary, fixed rule).** beta = sigmoid(log(pi ||e||²/n)) with
  pi the running inverse error variance (Gamma tracker, forgetting 0.9) matches the best fixed rate
  without knowing the regime.
- **P4 (non-destructive update; secondary, fixed rule).** The two-trace memory (slow trace at 0.2,
  labile at 1.0, readout trusting the labile trace in proportion to its content-based confirmation
  count) is robust to outliers without losing genuine changes.

**Task.** `flyprinc.tasks.rebind_stream`: 48 keys; each step presents one random pair (a write
with observation noise sigma, a noise-to-signal norm ratio); with probability 0.001 per key per
step a key is silently re-bound to a new value; with probability p_outlier a presentation shows a
wrong value once. Every 50 steps all keys are queried (tags: stable / re-bound within 150 steps /
last presentation was an outlier). T = 1,500. Four regimes: sigma in {0.3, 1.0} x p_outlier in
{0, 0.1}; 4 held-out streams of 8 sequences per regime, fixed seed. Memory: sparse random
addressing as in E1 (1,024 cells, 64 active), delta writes, one compartment unless stated.

**Conditions.**
- `const_0.3`, `const_0.6`, `const_1.0`: fixed rates (the best per seed-mean is the fixed-rate control).
- `precision`: P3 with a = 1, b = 0 fixed (no tuned constants).
- `two_trace`: P4.
- `learned_free`: MLP gate (32 tanh units) on (key, value, error, log squared error, log read norm,
  four zero slots) (trained).
- `learned_state`: the same MLP with the slots (nconf, a, zc, 0) (trained).
Trained conditions: 80 Adam steps on fresh T = 600 streams cycling through the four regimes (one
generalist gate), learning rates {1e-3, 3e-3, 1e-2}, best by seed-mean of the primary metric.
Seeds {0..4}.

**Metric.** Primary: overall query accuracy (natural mix of tags) averaged over the four regimes.
Secondary: per-regime overall accuracy; per-tag accuracies (stable / re-bound / outlier).

**Thresholds.**
- K0 supported only if `learned_state` beats `learned_free` on the primary by > 2 seed-SD with the
  bootstrap CI excluding 0. A difference inside the seed spread is a null and is reported as "the
  admitted states carry nothing this gate needs on this task at module level". The per-regime and
  per-tag breakdown (outlier keys, re-bound keys) is reported whatever the verdict.
- P3 supported only if `precision` beats the best fixed rate by > 2 seed-SD with CI excluding 0.
- P4 supported only if `two_trace` beats `precision` by > 2 seed-SD with CI excluding 0.
- Both learned arms are also compared with the best fixed rate and with `two_trace` (reported, no
  verdict). Positive verdicts are replicated with `--variant replication --seeds 5 6 7 8 9 10`
  before being reported as supported.
