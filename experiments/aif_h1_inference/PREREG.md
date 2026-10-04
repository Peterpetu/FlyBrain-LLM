# PREREG — AIF-H1: does per-token free-energy inference help at matched compute?

**Hypothesis.** A predictive-coding filter that refines its fast latent state with up to
K = 3 gradient steps on the variational free energy per token (adaptive stopping) predicts
characters better than the same filter with K = 0 (amortised bottom-up correction only),
and better than a K = 0 filter whose core is widened so its per-token FLOPs match the
average K > 0 cost.

**Model.** FlyAI (flyai/model.py): d = 64 embedding, core n = 128 (Dale-signed, 30.8 %
inhibitory, fan-in 32, divisive normalisation, Lyapunov-regulated gain), mushroom-body
memory (512 KCs, 5 % active, precision-gated delta writes, 2 heads, d_v 32), attractor
bank (4 rings x 16), inferred precisions with learned multipliers, arousal on.

**Data.** TinyShakespeare, character level, 90/10 split; training windows of 96
characters, batch 32; validation on the first 60k validation characters in
non-overlapping 96-character windows (loss on positions 2..96 of each window).

**Conditions.** `flyai_K3` (hypothesis; adaptive K <= 3), `flyai_K0` (ablation),
`flyai_K0_wide` (n = 192, compute-matched amortised control), `flyai_K3_fixed`
(K = 3 always; tests whether adaptivity matters).

**Budget.** 300 AdamW steps, warm-up 20, cosine to 0.1x, wd 0.01, clip 1; LRs
{1e-3, 3e-3, 1e-2} for every condition (compute limit: the same 3-point sweep for
hypothesis and controls; noted as a deviation from the >= 5-point baseline rule);
seeds {0, 1, 2}. Best LR per condition by seed-mean of the primary metric.

**Metric.** Validation bits/char (mean of the last two evaluations).

**Thresholds.** Supported only if `flyai_K3` beats BOTH `flyai_K0` and `flyai_K0_wide`
by > 2 seed-SD with paired-bootstrap CIs excluding 0. If `flyai_K0_wide` matches or beats
`flyai_K3`, iterative inference is not worth its compute at this scale (null). The
`flyai_K3_fixed` row is descriptive (adaptivity vs fixed K), not a threshold.


## v0.2 change log (before the first run)

The model is FlyAI v0.2 (docs/aif/DESIGN.md): level-0 precision is a learned temperature
(no Gamma tracking of one-hot errors), the Newton step carries the NLL curvature bound,
the prior is the precision-ratio fusion of the core and attractor predictions with grouped
(G = 2) precisions, level-3 precision comes from ring amplitude, the memory uses the
delayed hetero-associative key with a shared KC address and the surprise-driven gate,
arousal scales the sensory precision and the write gate (not F), and optimiser hygiene
(no weight decay on log-parameters, E/I re-balance) applies to every condition. The
conditions, budget, metric and thresholds are unchanged.
Added before the first run, after phase-1 H2 reported that the ring attractor loses to an
unstructured layer on state tracking: a descriptive arm `flyai_K3_noattr` (attractor bank
removed) to measure the bank's contribution to LM loss. Not a threshold.