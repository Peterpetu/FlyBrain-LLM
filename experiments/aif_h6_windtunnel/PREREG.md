# PREREG — AIF-H6: Bayesian wind tunnel (does the belief machinery track a real posterior?)

**Question.** A model that claims to hold beliefs with precisions should reproduce the
exact Bayesian posterior predictive on data where it is known in closed form. Small
transformers do this to ~1e-3 bits (Agarwal, Dalal & Misra 2025); capacity-matched MLPs
fail by orders of magnitude. Where does FlyAI sit, and does iterative inference help?

**Task.** Random HMMs (6 hidden states, 16 symbols, Dirichlet(0.3) rows), freshly sampled
per sequence, length 64. The exact forward-algorithm predictive is the target. Metric:
mean KL(true predictive || model predictive) in bits over positions 8-64 (after the
model has seen enough to identify the HMM), on 8 fixed held-out batches of 32.

**Conditions.** `flyai_K2` (hypothesis; 2 inference steps), `flyai_K0` (amortised only),
`gdn` (Gated-DeltaNet mixer, 1 block, d 64) and `attention` (2 layers, reference), all
trained with next-token cross-entropy (the log-loss is minimised by the true predictive,
so training points at the right target without ever seeing it).

**Budget.** 400 AdamW steps, batch 32, LRs {1e-3, 3e-3, 1e-2} for all conditions, seeds
{0, 1, 2}.

**Thresholds.** AIF-H6a (belief tracking): `flyai_K2` reaches KL < 0.05 bits (an order of
magnitude above the transformer reference level, generously). AIF-H6b (inference helps):
`flyai_K2` beats `flyai_K0` by > 2 seed-SD with CI excluding 0. Descriptive: KL of `gdn`
and `attention` at the same budget.


## v0.2 change log (before the first run)

Model is FlyAI v0.2. Added the descriptive precision-tracking diagnostic: Spearman
correlation between the model's inferred level-1 precision and the true posterior
predictive's concentration (negative entropy) over positions 8-64 (the design panel's
suggestion; not a threshold).
Added before the first run (after phase-1 H2's negative on the ring attractor): a
descriptive arm `flyai_K2_noattr` (attractor bank removed) to measure whether the slow
belief state contributes to posterior tracking. Not a threshold.

## v0.2.2 change log (formal verification pass, before the first run)

Implementation fixes found by verifying the code against docs/aif/DESIGN.md (report:
docs/aif/VERIFICATION.md). None changes the hypothesis, conditions, budget, seeds, learning
rates, metrics or thresholds registered above.
* `flyai_K2` runs exactly 2 inference steps (`adaptive_K=False`) and K_mean is logged per run.
* The precision-tracking diagnostic pairs the level-1 precision ENTERING step t (before o_t is observed) with the concentration of the true predictive for o_t; the first version used the precision after observing o_t.
* Precisions are bounded ratios ([1e-3, 1e3]); a diverged run enters the tables as NaN; AIF-H6a (absolute KL < 0.05 bits) is evaluated mechanically by the results builder.
