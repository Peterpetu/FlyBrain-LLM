# PREREG — H5: the connectome graph as a recurrent mask (calibration / negative control)

**Hypothesis under test (the community default; we predict it is weak).** A character-level
recurrent language model whose recurrent connectivity is masked by the real MaleCNS core
graph (Dale's-law signs fixed, magnitudes learned) beats the same model masked by a
degree-preserving shuffle or an Erdos-Renyi graph with the same N and E, and does *not*
beat a tuned dense recurrent layer at matched parameters or matched compute. We test how
the real-vs-shuffle gap behaves with N.

**Data.** TinyShakespeare (1.1 M characters), 90/10 split, character-level. Sequence
length 128, batch 32.

**Model.** Embedding (64) -> 1 block [recurrent mixer + MLP] -> head. Recurrent mixer:
h_t = tanh(x_t W_in + h_{t-1} W_rec + b), W_rec = mask * |W| * sign (Dale) for graph
conditions, unconstrained dense for dense conditions.

**Core graphs.** Top-N neurons by contact-weighted degree inside the largest strongly
connected component of the whole-CNS graph, N in {256, 512, 1024}. Magnitudes initialised
from log(1 + contacts) column-normalised.

**Conditions per N.** `real`; `shuffle` (directed degree-preserving swaps, 10 |E|
attempts, same node signs); `er` (G(N, E), permuted weights, same node signs);
`dense_param` (unconstrained dense with N_p = round(sqrt(E)) hidden units: same number
of learnable recurrent weights); `dense_N` (unconstrained dense with N hidden units: same
wall-clock/dense FLOPs as the masked implementation).

**Budget.** 400 AdamW steps, warm-up 20, cosine, weight decay 0.01, clip 1. Seeds
{0, 1, 2}. LRs: graph conditions {1e-3, 3e-3, 1e-2}; dense conditions {3e-4, 1e-3, 3e-3,
1e-2, 3e-2}. Best LR per condition by seed-mean validation bits/char.

**Metric.** Validation bits per character on the first 100k validation characters
(non-overlapping windows of 128).

**Pre-registered predictions and thresholds.**
- Expect real < shuffle (better) by a small margin that does not grow with N. The gap is
  "real" only if it exceeds 2 seed-SD with a bootstrap CI excluding 0 at >= 2 of 3 sizes.
- Expect dense_param and dense_N to beat all graph conditions. If any graph condition
  beats dense_N at any N, that is a positive worth following up (H3/H4 unlock).
- If real ~ shuffle ~ er within seed spread at all N, the graph is inert as a mask here.
