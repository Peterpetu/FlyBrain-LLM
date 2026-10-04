# PREREG — AIF-H2: inferred precision as the dopamine gate, under non-stationarity

**Hypothesis.** Gating fast-weight writes by an *inferred* precision (Gamma posterior on
the memory's own prediction errors, two learned scalars) recalls re-bound key-value pairs
better than a learned input-dependent gate (Gated-DeltaNet style) or a constant gate,
because the precision gate raises the write strength exactly when the memory is being
contradicted (surprise) and lowers it when predictions are reliable.

**Task.** Non-stationary MQAR (flyai/tasks.py): vocab 48, 8 key-value pairs, sequence
length 96; at the midpoint 4 keys are re-bound to new values; queries before the midpoint
must return the old value, queries after it the new value. Primary metric: accuracy on
post-rebind queries (chance 1/16). Secondary: pre-rebind accuracy; held-out length 192.

**Model.** FlyAI with K = 1 inference step, n = 128, memory 512 KCs; only `memory_gate`
differs: `precision` (hypothesis), `learned`, `const`. Plus `no_memory` (attractor + core
only) as a floor.

**Budget.** 300 AdamW steps, batch 32, warm-up 20, cosine, wd 0.01, clip 1; LRs
{1e-3, 3e-3, 1e-2} for all conditions; seeds {0, 1, 2}. Loss on query positions only.

**Thresholds.** Supported only if `precision` beats `learned` on post-rebind accuracy by
> 2 seed-SD with a paired-bootstrap CI excluding 0, and is not worse on pre-rebind
accuracy beyond seed spread. If `learned` matches or beats `precision`, the inferred
precision carries no advantage here (null).


## v0.2 change log (before the first run)

The v0 gate beta = sigmoid(a log pi + b) with a > 0 LOWERED writes under contradiction,
the opposite of the hypothesis. The v0.2 'precision' gate is the variance-normalised
surprise sigmoid(a log(pi_slow ||eps||^2 / n) + b): it rises when the current error is
large relative to the expected error. The ladder is now {precision (hypothesis), surprise
(raw ||eps||^2, Titans-style), learned, const, no_memory}. Threshold: supported only if
`precision` beats BOTH `surprise` and `learned` on post-rebind accuracy by > 2 seed-SD
with CIs excluding 0 (the raw-surprise control isolates the role of the inferred
precision); otherwise null. The memory now uses the delayed hetero-associative key and a
shared KC address with log-spaced decay priors in every condition.

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
* K is exactly 1 (`adaptive_K=False`), as registered; the adaptive stopping rule was on by default.
* Answers are scored as the argmax over the value vocabulary, so chance is exactly 1/16 as registered.
* The `const` gate is now exactly 0.5 (arousal no longer entered its logit), so it is the constant control it was registered as.
* Precisions are bounded ratios ([1e-3, 1e3], MODEL_SPEC section 6) everywhere they are read; an unbounded Gamma ratio could overflow.
* A run that diverges enters the tables with NaN (never a partially trained finite metric); the learning-rate selection treats NaN as worst.
* The pre-rebind guard (not worse than `learned` on pre-rebind accuracy beyond seed spread) is evaluated mechanically (`pre_rebind` in summary.json) and the held-out length-192 pre-rebind accuracy is tabulated.
* The Lyapunov exponent is estimated (not acted on) in every condition; the gain controller is the dithered extremum-seeking design described in docs/aif/VERIFICATION.md.
