# PREREG — H4: Dale's law / E-I balance and recurrent stability at long context

**Hypothesis.** A sign-constrained recurrent layer (separate excitatory and inhibitory
populations, non-negative magnitudes, inhibitory fraction f) keeps hidden-state and
gradient norms bounded when run at 4-16x the training context, better than the tuned
unconstrained layer, without a loss penalty at training length.

**Setup.** Dense 256-unit tanh recurrent mixer in the char-LM backbone (TinyShakespeare,
train seq 128, batch 32, 400 steps, seeds {0,1,2}). Conditions: `unconstrained`;
`dale_f0.2`, `dale_f0.3` (MaleCNS: 30.8 %), `dale_f0.5`. Dale conditions have the sign
vector fixed and magnitudes learned. LRs: unconstrained {3e-4, 1e-3, 3e-3, 1e-2, 3e-2};
Dale {1e-3, 3e-3, 1e-2}.

**Metrics.** (1) validation bits/char at seq 128 (primary for LR selection); (2) at
sequence lengths 512, 1024, 2048: validation bits/char, mean ||h_t|| over the last 128
positions relative to the first 128, and the ratio of the gradient norm of the loss at
the last 128 positions w.r.t. the embedding of the first 128 tokens (exploding /
vanishing diagnostic).

**Thresholds.** Supported only if at least one Dale fraction matches the unconstrained
layer at 128 within seed spread AND has strictly better (lower) bits/char at 2048 by
> 2 seed-SD. A stability metric improvement without a loss improvement is reported as a
partial result, not support.
