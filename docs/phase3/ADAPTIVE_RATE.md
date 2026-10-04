# Learning rate as inferred precision: what the numbers say (axiom A4)

*Pure-numpy study (`results/adaptive_rate_numpy.json`), run before the phase-3 design. A sparse
delta-rule memory (64 of 1,024 active rows, 48 stored pairs, half its capacity) is fed a stream in
which each observation is a stored pair's value plus Gaussian noise sigma, and now and then one
pair is silently re-bound to a fresh value (change points, 0.004 per pair per step). Every write
uses M += beta · key (v_obs − read)^T with a rate policy; recall and post-re-binding recall are
measured on the true current values.*

| noise sigma | policy | recall MSE | recall accuracy | accuracy on pairs re-bound in the last 200 steps |
|---|---|---|---|---|
| 0.3 | fixed 0.2 | 0.562 | 0.63 | 0.39 |
| 0.3 | fixed 0.5 | 0.423 | 0.78 | 0.62 |
| 0.3 | fixed 1.0 | 0.428 | 0.80 | 0.69 |
| 0.3 | surprise-gated, per-pair tracker | 0.360 | 0.81 | 0.69 |
| 0.3 | surprise-gated, one global tracker | 0.370 | 0.80 | 0.68 |
| 0.3 | change-point Kalman oracle (knows sigma and the jump rate) | 0.398 | 0.81 | 0.68 |
| 1.0 | fixed 0.2 | 0.664 | 0.58 | 0.35 |
| 1.0 | fixed 0.5 | 0.716 | 0.67 | 0.51 |
| 1.0 | fixed 1.0 | 1.343 | 0.60 | 0.52 |
| 1.0 | surprise-gated, per-pair tracker | 0.706 | 0.71 | 0.56 |
| 1.0 | surprise-gated, one global tracker | 0.723 | 0.69 | 0.54 |
| 1.0 | change-point Kalman oracle | 0.651 | 0.67 | 0.49 |

The gated rule is beta = sigmoid(log(pi · ||e||²/n)), with pi the running inverse error variance
(Gamma tracker, forgetting 0.9): a large error *relative to the usual error* opens the gate. It is
the three-factor rule of docs/phase3/REFLECTION.md axiom A4 with no tuned constants.

**Reading.** The best fixed rate depends on the noise level (1.0 at low noise, 0.5 at high noise);
the gated rate is at or above the best fixed rate in both regimes without knowing the regime, and
on a par with an oracle that knows the noise and the jump rate. The advantage over a *tuned* fixed
rate is a few points, so this axiom buys robustness and tuning-freedom, not a large gain; a
learned gate with the error as input can learn the same rule (phase-1 H1b confirmed that a learned
gate is at least as good in the easy regime). A single global tracker loses almost nothing against
per-pair trackers, which is what a model can implement for keys it has never seen.
