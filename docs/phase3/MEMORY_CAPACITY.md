# The one hard number the pivot rests on: capacity of sparse one-shot memories at equal compute

*Pure-numpy study (`results/capacity_numpy.json`, `results/figures/capacity_numpy.png`), run before
the phase-3 design was written. No learning, no task: P random (key, value) pairs are written once,
in order, and every key is queried; recall is correct if the read-out is nearest to its own value.*

**Setting.** A fast-weight memory M reads r = key^T M and writes with either the Hebbian outer
product (M += key v^T) or the delta rule (M += key (v − key^T M)^T; the error-correcting form that
Gated DeltaNet, DeltaNet and the fly's dopamine-gated rule all use). Keys are either dense random
vectors of dimension d_key, or sparse k-of-n codes (k active units of n, as the Kenyon-cell code:
n = 2,048 cells, k ≈ 100 at a 5 % coding level). **Per-token compute is equal when d_key = k**,
because a read or write touches d_key rows in the dense case and k rows in the sparse case
(gather/scatter). The sparse memory's *state* is n/k times larger (16-20x); that is the trade.

**Result (value dimension 16, 5 trials, one-shot recall accuracy).**

| stored pairs P | dense delta, d_key = 64 | sparse Hebb, 64 of 1,024 | sparse delta, 64 of 1,024 | dense delta, d_key = 100 | sparse delta, 100 of 2,048 |
|---|---|---|---|---|---|
| 32 | 0.81 | 0.96 | 0.97 | 0.88 | 0.97 |
| 64 | 0.54 | 0.84 | 0.94 | 0.72 | 0.95 |
| 128 | 0.25 | 0.57 | 0.90 | 0.40 | 0.92 |
| 256 | 0.11 | 0.33 | 0.75 | 0.17 | 0.82 |
| 512 | 0.04 | 0.13 | 0.56 | 0.07 | 0.70 |

**Reading.**
1. A dense delta-rule memory holds about d_key associations (it is a rank-d_key object); past
   that, recall collapses. This is the regime phase-1 H1 never entered (8 pairs ≪ 65).
2. A sparse code with the same number of active rows per token keeps recalling far beyond d_key:
   at P = 4 d_key the sparse delta memory is at 0.90 where the dense one is at 0.25. The
   mechanism is the near-orthogonality of random sparse codes (the expected overlap of two k-of-n
   codes is k²/n ≈ 4-6 active units, against k = 64-100), i.e. the fly-hashing argument
   (Dasgupta et al. 2017) and the sparse-distributed-memory capacity argument (Kanerva 1988).
3. The delta rule matters as much as the sparsity: sparse Hebb decays from 0.84 to 0.33 between
   P = 64 and P = 256 while sparse delta goes from 0.94 to 0.75. Error-corrected writes are what a
   prediction-error-gated dopamine signal implements; the biology and the mathematics agree on the
   rule, and phase 1 confirmed the delta write over Hebb (H1c, 0.942 vs 0.981).

**What this does and does not show.** It shows a compute-matched capacity advantage for sparse
addressing in the one-shot regime, with numbers. It does not show that a language model needs
that capacity, that a learned dense key of larger dimension at more compute would not do better,
or that the advantage survives learned keys and gates. Those are exactly the questions of the
phase-1 capacity experiment H1b (pre-registered, running now at 40 pairs against a
compute-matched Gated-DeltaNet twin) and of the phase-3 design.
