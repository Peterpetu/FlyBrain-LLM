# PREREG — AIF-H4: edge-of-chaos regulation of the recurrent core

**Hypothesis.** Regulating the Dale-signed core's global gain so that its estimated
largest Lyapunov exponent stays near a slightly subcritical target (-0.02; driven
homeostasis settles subcritical and the biological branching ratio is ~0.98, not 1; one
Jacobian-vector product per token, gain updated by a homeostatic rule, no gradient) retains information over long distances
better than the same core with its gain fixed in the contractive regime (0.5) or the
expansive regime (2.0), and at least as well as a gain fixed at the value the regulated
core converges to (checked post hoc).

**Task.** Long-range retention (flyai/tasks.py): 4 cue-value pairs at the start, a random
filler, the 4 cues at the end; vocab 48; train length 64; held-out fixed batches at
lengths 64 (primary), 128 and 256 (extrapolation, secondary). Chance 1/16.
The mushroom-body memory is OFF in every condition so that retention must be carried by
the recurrent core and the attractor bank (a fast-weight memory would trivialise the task).

**Model.** FlyAI, K = 1, n = 128, use_memory = False, attractor on. Conditions:
`regulated` (hypothesis), `fixed_0.5`, `fixed_2.0`, `fixed_1.0`.

**Budget.** 300 AdamW steps, batch 32, LRs {1e-3, 3e-3, 1e-2}, seeds {0, 1, 2}. Loss on
the 4 final query positions.

**Thresholds.** Supported only if `regulated` beats both `fixed_0.5` and `fixed_2.0` at
length 64 by > 2 seed-SD with CI excluding 0. If `fixed_1.0` matches `regulated`, the
homeostatic mechanism adds nothing over a well-chosen constant (partial null, reported).
Lyapunov estimates and final gains are reported for every run.


## v0.2 change log (before the first run)

The core is now tanh with Rajan-Abbott-balanced, unit-spectral-radius initialisation and
log-parametrised magnitudes (divisive normalisation was removed from the core because it
made the dynamics gain-invariant); the controller is extremum-seeking toward
lambda = -0.02 and runs every 8 tokens; E/I columns are re-balanced after every optimiser
step in all conditions. Fixed-gain arms are unchanged.

## v0.2.1 change log (verification pass, before the first run)

Target alignment fix, found by the formal verification of the implementation against the
spec: this task places the target value AT the query position (the phase-1 convention for
causal models whose logits at t include x_t). FlyAI emits logits[:, t] BEFORE observing
x[:, t], so the driver as first written asked the model to answer a query it had not yet
seen, and every arm would have scored at chance. The driver now scores the prediction made
after observing the query token (`flyai.tasks.same_index_logits`: one extra filler step,
first prediction dropped). Conditions, budget, seeds, LRs, metrics and thresholds are
unchanged.

## v0.2.2 change log (formal verification pass, before the first run)

Implementation fixes found by verifying the code against docs/aif/DESIGN.md (report:
docs/aif/VERIFICATION.md). None changes the hypothesis, conditions, budget, seeds, learning
rates, metrics or thresholds registered above.
* K is exactly 1 (`adaptive_K=False`), as registered; answers are scored over the value vocabulary (chance exactly 1/16).
* Fixed-gain arms now report their (estimated, never acted on) Lyapunov exponent, as this PREREG promises for every run; before the fix they would have reported 0.0.
* The gain controller is the dithered extremum-seeking design (relative dither 0.1, demodulated slope estimate, dead-band 0.01); the previous single-step slope estimator could step away from the target. Target and cadence are unchanged.
* The post-hoc arm "gain fixed at the value the regulated core converges to" is scripted (`fixed_conv`: mean final base gain of `regulated` at its best learning rate, rounded to 0.01) and reported next to the pre-chosen fixed gains.
* Precisions are bounded ratios ([1e-3, 1e3]); a diverged run enters the tables as NaN.
