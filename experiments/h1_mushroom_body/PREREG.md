# PREREG v2 — H1 (supersedes v1 below; see 'Change log')


**Hypothesis.** A residual fast-weight layer whose keys pass through a fixed sparse
expansion with k-winner-take-all (PN -> KC -> APL), whose write is a prediction-error
(delta) update, and whose learning rate is gated by the error itself (DAN-like) recalls
in-context key-value associations better than a compute-matched dense fast-weight baseline.

**Task.** Multi-query associative recall (MQAR; Arora et al. 2023). Vocab 64 (21 keys, 21
values, 22 fillers). Train: sequence length 64, 8 key-value pairs, 8 queries. Held-out
evaluation on fixed seeded batches: (a) in-distribution T=64 / 8 pairs — **primary**;
(b) T=128 / 16 pairs (capacity, out of distribution); (c) T=128 / 8 pairs (length).

**Model.** 1 block of [mixer + MLP], d_model 64, embedding + head; only the mixer differs.
Every fast-weight mixer has a depthwise causal convolution (kernel 4, SiLU) before its
projections, the standard short-conv of Gated DeltaNet / Mamba, so a single layer can pair
a key with the value that follows it. Fast-weight mixer: 2 heads (compartments), d_value 16.
`attention` keeps 2 layers (softmax attention needs a second layer for the shift).

**Conditions.**
- `mb` (hypothesis): sparse_random expansion, 61 PN channels, 1024 KCs, fan-in 5,
  k-WTA 5 %, delta write, error gate, learned decay.
- `mb_connectome`: same, expansion = measured PN-type -> KC contact matrix of the right
  hemisphere (1024 KCs sampled with the run seed).
- `mb_shuffled`: same, expansion = degree-preserving shuffle of the measured matrix.
- `abl_hebb`: `mb` with Hebbian (write-only) update.
- `abl_learned_gate`: `mb` with an input-dependent learned gate instead of the error gate.
- `abl_no_kwta`: `mb` without k-WTA (all 1024 KCs active).
- `abl_dense_expanded`: a dense *learned* projection to 1024 keys with no k-WTA (the
  falsification ablation: same state size, no fixed sparse expansion).
- Baselines (dense keys, key dim chosen so per-token cost matches `mb`'s effective
  cost, see `fast_weight_flops`): `gdn_matched` (delta, learned alpha/beta),
  `deltanet_matched` (delta, learned beta, no decay), `linattn_matched` (Hebb,
  beta=1, no decay). `attention` (2-head softmax, RoPE) is a reference, not a control.

**Budget.** 800 AdamW steps, batch 32, warm-up 20, cosine to 0.1x, weight decay 0.01,
gradient clip 1, evaluation every 100 steps. Seeds {0, 1, 2}. Learning rates: `mb`,
`mb_connectome`, `mb_shuffled`, `abl_dense_expanded` {1e-3, 3e-3, 1e-2}; baselines
{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}; `abl_hebb`, `abl_learned_gate`, `abl_no_kwta` at the LR
selected for `mb` (compute limit; this can only favour the hypothesis, so an ablation that
*matches* `mb` is conservative evidence against the feature, while an ablation that loses
is weaker evidence because it may be under-tuned — stated as such in RESULTS). Best LR
per condition by seed-mean of the primary metric.

**Metric.** Accuracy on query positions. Mean +/- SD over seeds; paired bootstrap 95 % CI
of the seed-paired difference vs each control.

**Falsification thresholds (pre-registered).**
- H1 is *supported* only if `mb` beats `gdn_matched` on the primary metric by more than
  2 seed-SDs with a bootstrap CI excluding 0, AND `abl_dense_expanded` does not match
  `mb` (i.e. the fixed sparse expansion is necessary).
- H1a (sparse expansion): falsified if `abl_no_kwta` or `abl_dense_expanded` matches
  `mb` within seed spread.
- H1b (error gate): falsified if `abl_learned_gate` matches or beats `mb`.
- H1c (delta/depression write): falsified if `abl_hebb` matches or beats `mb`.
- H1d (real wiring): supported only if `mb_connectome` beats `mb_shuffled` by > 2 seed-SD;
  otherwise the measured PN->KC pattern carries no extra information beyond its degree
  sequence for this task.
Any difference smaller than the seed spread is reported as null.

**Compute.** 4 CPU threads, no GPU. Cost matching is by analytic multiply-adds per token
(`recurrent_effective + projections`), not wall-clock.


## Change log (why there is a v2)

v1 (hash 4a…, 2 layers, 400 steps) was frozen and 4 runs of the `mb` condition at its two
lowest learning rates completed before any other condition ran. Their learning curves were
still rising at step 400 (accuracy 0.09-0.16 vs chance 0.048), showing the budget would
compare under-trained models. No cross-condition comparison had been seen. v2 doubles the
steps, halves the depth (one mixer layer with the standard short convolution) to pay for
it, moves the width-512 ablations to full width, and fixes the ablation LR policy. The v1
text is kept verbatim below for the record.

---

## Superseded v1

# PREREG — H1: Mushroom-body memory as a sparse, surprise-gated fast-weight layer

**Hypothesis.** A residual fast-weight layer whose keys pass through a fixed sparse
expansion with k-winner-take-all (PN -> KC -> APL), whose write is a prediction-error
(delta) update, and whose learning rate is gated by the error itself (DAN-like) recalls
in-context key-value associations better than a compute-matched dense fast-weight baseline.

**Task.** Multi-query associative recall (MQAR; Arora et al. 2023). Vocab 64 (21 keys, 21
values, 22 fillers). Train: sequence length 64, 8 key-value pairs, 8 queries. Held-out
evaluation on fixed seeded batches: (a) in-distribution T=64 / 8 pairs — **primary**;
(b) T=128 / 16 pairs (capacity, out of distribution); (c) T=128 / 8 pairs (length).

**Model.** 2 blocks of [mixer + MLP], d_model 64, embedding + head; only the mixer differs.
Fast-weight mixer: 2 heads (compartments), d_value 16.

**Conditions.**
- `mb` (hypothesis): sparse_random expansion, 61 PN channels, 1024 KCs, fan-in 5,
  k-WTA 5 %, delta write, error gate, learned decay.
- `mb_connectome`: same, expansion = measured PN-type -> KC contact matrix of the right
  hemisphere (1024 KCs sampled with the run seed).
- `mb_shuffled`: same, expansion = degree-preserving shuffle of the measured matrix.
- `abl_hebb`: `mb` with Hebbian (write-only) update.
- `abl_learned_gate`: `mb` with an input-dependent learned gate instead of the error gate.
- `mb_512`, `abl_no_kwta_512`, `abl_dense_expanded_512`: at 512 KCs, `mb`, `mb` without
  k-WTA (all KCs active), and a dense *learned* projection to 512 keys with no k-WTA
  (the falsification ablation: same state size, no fixed sparse expansion).
- Baselines (dense keys, key dim chosen so per-token cost matches `mb`'s effective
  cost, see `fast_weight_flops`): `gdn_matched` (delta, learned alpha/beta),
  `deltanet_matched` (delta, learned beta, no decay), `linattn_matched` (Hebb,
  beta=1, no decay). `attention` (2-head softmax, RoPE) is a reference, not a control.

**Budget.** 400 AdamW steps, batch 32, warm-up 20, cosine to 0.1x, weight decay 0.01,
gradient clip 1. Seeds {0, 1, 2}. Learning rates: hypothesis and ablations {1e-3, 3e-3,
1e-2}; baselines {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}. Best LR per condition by seed-mean of the
primary metric.

**Metric.** Accuracy on query positions. Mean +/- SD over seeds; paired bootstrap 95 % CI
of the seed-paired difference vs each control.

**Falsification thresholds (pre-registered).**
- H1 is *supported* only if `mb` beats `gdn_matched` on the primary metric by more than
  2 seed-SDs with a bootstrap CI excluding 0, AND `abl_dense_expanded_512` does not match
  `mb_512` (i.e. the fixed sparse expansion is necessary).
- H1a (sparse expansion): falsified if `abl_no_kwta_512` or `abl_dense_expanded_512`
  matches `mb_512` within seed spread.
- H1b (error gate): falsified if `abl_learned_gate` matches or beats `mb`.
- H1c (delta/depression write): falsified if `abl_hebb` matches or beats `mb`.
- H1d (real wiring): supported only if `mb_connectome` beats `mb_shuffled` by > 2 seed-SD;
  otherwise the measured PN->KC pattern carries no extra information beyond its degree
  sequence for this task.
Any difference smaller than the seed spread is reported as null.

**Compute.** 4 CPU threads, no GPU. Cost matching is by analytic multiply-adds per token
(`recurrent_effective + projections`), not wall-clock.
