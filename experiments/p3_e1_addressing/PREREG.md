# PREREG — P3-E1: sparse random addressing raises one-shot capacity at equal active rows

*Module level (docs/phase3/EXPERIMENT_PLAN.md). Frozen at the first run.*

**Hypothesis (P1).** A fast-weight memory addressed by a fixed random sparse expansion (the
Kenyon-cell code: 1,024 cells, fan-in 7 of 64 input channels, 64 active per event after k-WTA,
divisive normalisation of the input) recalls one-shot associations at a higher load than a
dense delta-rule memory whose key dimension equals the number of active rows (64), and the
advantage is not reproduced by a learned dense expansion of the same width without k-WTA.

**Why this is the right test.** docs/phase3/MEMORY_CAPACITY.md (numpy) predicts the effect; this
experiment measures it in the module that the phase-3 design would use, with learned controls
trained by backpropagation through the module.

**Task.** `flyprinc.tasks.capacity_stream`: P distinct keys written once with random values, then
every key queried; keys are fixed random Gaussian vectors (vocabulary 2,048, d_key 64), values
fixed random unit vectors (vocabulary 64, d_val 16). P in {32, 64, 128, 256, 512}; 6 held-out
streams of 8 sequences per P, generated with a fixed seed. Recall = argmax cosine over the value
vocabulary.

**Conditions** (all delta-rule writes, constant gate 1.0, one compartment):
- `sparse_random` (hypothesis): fixed random expansion, fan-in 7, k = 64 of 1,024, normalisation on.
- `dense`: keys = normalised input, 64 rows, all active (equal active rows; 16x less state).
- `dense_expanded`: learned 1,024 x 64 expansion, no k-WTA, all 1,024 rows active (equal state, 16x more active rows), trained.
- `sparse_learned`: learned expansion with k-WTA (descriptive: is the fixed random expansion necessary?), trained.
- `sparse_nonorm`: hypothesis without divisive normalisation (P9 ablation).
- `sparse_hebb`: hypothesis with Hebbian writes (P2 sanity row).
Trained conditions: 100 Adam steps on fresh P = 128 streams, learning rates {1e-3, 3e-3, 1e-2},
best by seed-mean of the primary metric; fixed conditions have no learning rate. Seeds {0..4}
(the seed sets the expansion and the learned initialisation; the held-out streams are fixed).

**Metric.** Primary: recall accuracy at P = 128 (chance 1/64). Secondary: the full P curve;
state size and active rows per condition are reported in every table.

**Thresholds.** Supported only if `sparse_random` beats `dense` at P = 128 by > 2 seed-SD with
a paired-bootstrap 95 % CI excluding 0 AND beats `dense_expanded` beyond seed spread. If
`dense_expanded` matches `sparse_random`, the gain is state size, not sparse addressing. If
`sparse_learned` beats `sparse_random` beyond threshold, the fixed random expansion is not
necessary (reported, not a support condition). Differences inside the seed spread are nulls.
A positive verdict is replicated with `--variant replication --seeds 5 6 7 8 9 10` before it
is reported as supported.
