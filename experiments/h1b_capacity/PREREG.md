# PREREG — H1b: mushroom-body memory in the capacity regime of MQAR

**Why this experiment exists.** Written while H1 was still running, before any baseline
result was seen. H1's task (8 pairs in 64 tokens) is the easy regime: the MB layer reaches
~98 %, and a compute-matched Gated DeltaNet is expected to saturate too, which would make
H1 uninformative about *capacity*. The hypothesis behind the sparse KC code is precisely
about interference when many associations must coexist (Litwin-Kumar 2017; sparse
distributed memory), so it must be tested where interference bites.

**Hypothesis.** With many more key-value pairs than the dense baseline's key dimension can
hold orthogonally, the sparse-expansion memory (1024 KCs, 51 active) recalls better than
the compute-matched dense fast-weight memory (key dim 65), and the gain disappears when the
fixed sparse expansion is replaced by a dense learned projection of the same width.

**Task.** MQAR, vocab 192 (64 keys, 64 values, 64 fillers). Train: 40 pairs, sequence
length 160, 40 queries. Held-out fixed batches: (a) 40 pairs / T=160 — **primary**;
(b) 60 pairs / T=256 (harder, out of distribution); (c) 8 pairs / T=160 (easy, sanity).

**Model.** As H1 v2 (1 block, d_model 64, short conv, 2 heads, d_value 16).

**Conditions.** `mb` (sparse_random expansion, k-WTA 5 %, delta write, error gate,
learned decay) — LRs {1e-3, 3e-3, 1e-2}; `gdn_matched` (dense key 65, delta, learned
gates) — LRs {3e-4, 1e-3, 3e-3, 1e-2, 3e-2}; `abl_dense_expanded` (dense learned 1024-dim
keys, no k-WTA, otherwise as `mb`) — LRs {1e-3, 3e-3, 1e-2}. Seeds {0, 1, 2}.

**Budget.** 800 AdamW steps, batch 32, warm-up 20, cosine to 0.1x, wd 0.01, clip 1, eval
every 100 steps. 4 CPU threads at most.

**Metric.** Accuracy on query positions (chance 1/64 = 0.016).

**Falsification thresholds.**
- Supported only if `mb` beats `gdn_matched` on the primary metric by > 2 seed-SD with a
  paired-bootstrap 95 % CI excluding 0, AND `abl_dense_expanded` does not match `mb`
  within seed spread.
- If `gdn_matched` matches or beats `mb`, the sparse-expansion hypothesis is falsified at
  this scale. If `abl_dense_expanded` matches `mb`, the gain (if any) is due to state
  size, not the fixed sparse expansion.
- Differences inside the seed spread are nulls.
