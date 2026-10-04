# PREREG — H2: Central-complex ring attractor as a length-extrapolating state tracker

**Hypothesis.** A recurrent state built from the extracted EPG / PEN_a / PEN_b / PEG /
Delta7 matrix (148 neurons; contact counts signed by predicted neurotransmitter; one
learned gain per population pair; angular velocity = learned scalar function of the token
that multiplicatively modulates left vs right PEN populations) tracks a cyclic latent
state and extrapolates to sequences >= 4x longer than training, better than an
unstructured recurrent layer of identical size with the identical velocity channel.

**Task.** Compass state tracking: tokens {forward, L1, R1, L2, R2} plus 4 distractors; the
target at every position is the current heading in Z_8. Train length 32. Evaluate on fixed
seeded batches at lengths 32 (in-dist), 128 (4x, **primary**), 256 (8x), 512 (16x).

**Model.** 1 block [mixer + MLP], d_model 32, 8-way classification head at every position.

**Conditions.**
- `ring_connectome` (hypothesis): measured 148-neuron matrix, learned block gains.
- `ring_canonical`: idealised ring of the same size (positive control for the mechanism).
- `rnn_unstructured` (pre-registered falsifier): same 148 units, same velocity channels
  and direct drive, fully learned recurrent matrix.
- `rnn_plain`: 148-unit tanh RNN without any velocity channel (classic baseline).
- `gdn`: Gated-DeltaNet-style fast-weight mixer (2 heads, key 32, value 16).
- `attention`: 2-head softmax attention with RoPE (reference: transformers are known to
  fail to extrapolate on state tracking).

**Budget.** 600 AdamW steps, batch 32, warm-up 20, cosine, weight decay 0.01, clip 1.
Seeds {0, 1, 2}. LRs: hypothesis/canonical {1e-3, 3e-3, 1e-2}; all baselines
{3e-4, 1e-3, 3e-3, 1e-2, 3e-2}. Best LR per condition by seed-mean of the primary metric.

**Metric.** Per-position heading accuracy (chance 0.125).

**Falsification thresholds.**
- H2 falsified if `rnn_unstructured` matches `ring_connectome` at 4x within seed spread,
  or beats it. Supported only if `ring_connectome` exceeds `rnn_unstructured` at 4x by
  > 2 seed-SD with a bootstrap CI excluding 0.
- If `ring_canonical` succeeds but `ring_connectome` fails, the mechanism works but the
  measured matrix (with our sign/normalisation assumptions) does not implement it: report
  as a negative for the connectome, not for the principle.
- Ring topology ablation: `rnn_unstructured` *is* the ablation (same size, same inputs).
