# PREREG — P3-S1b (rung 1b, v2): does the Kenyon-cell sparse address raise the capacity of Gated DeltaNet's memory inside trained sequence models?

*System level, GPU (RTX 4080 SUPER, 16 GB). Frozen by its hash at the first plain run of `run.py`
(never by `--smoke` or `--probe`). Second version; v1 was never frozen or run.*

**Why v2.** v1 tested the address where every query comes more than 512 tokens after its key, in
2,048-token sequences, with a hybrid model. The rung-1 learnability work
(`experiments/p3_s1_hybrid/diagnostics`, docs/phase3/EXPERIMENT_PLAN.md) then showed that long,
filler-heavy recall tasks are not learned by any model within an affordable budget, while dense
recall tasks are learned by memory-only models. v2 tests the address where models demonstrably
learn: memory-only models on the capacity form of the dense recall task, in which dozens of pairs
load the memory at every query. v1 also used this repository's own gated layer as the control; v2's
primary comparison uses Gated DeltaNet's gating, so that the control IS Gated DeltaNet.

**Hypothesis H-S1b.** In a trained memory-only sequence model whose memory layers are Gated DeltaNet
(a delta-rule fast-weight store whose write strength, erase strength (equal to the write strength)
and decay are computed from the input), replacing the dense 16-dimensional key and query by the
Kenyon-cell code (a FIXED random expansion of the key to 256 cells, each cell averaging 4 randomly
chosen key dimensions, then winner-take-all keeping the 16 most active cells, unit norm) raises
held-out recall accuracy when the memory holds more pairs than the key has dimensions, at equal
active rows per token (16) and equal parameters (the expansion is a fixed buffer). The mechanism is
the one E1 measured at module level (0.993 against 0.368 recall at 128 stored pairs, replicated):
random sparse codes are nearly orthogonal (expected overlap k^2/n = 1 of 16 cells), so the store is
not limited to d_k near-orthogonal keys. (The key width 16 was chosen by the rule below, because at
32 the dense store was not the bottleneck of this task.)

**Model.** `flyprinc.hybrid.HybridLM`, memory-only: 2 memory layers, d_model 128, 4 heads,
d_k = 16, d_v = 32, SwiGLU MLP, untied output head, embedding std 1 (the configuration the rung-1
diagnosis validated), one content store per head, no history states. Main arms: Gated DeltaNet's
gating (`x_only_gate`, `tie_gamma`: the gate sees only the input; per-head decay ladder from 8 to
2,048 tokens scaled by the learned, input-dependent forget drive); chunked form with chunks of 64
tokens, exact in one solve round at any chunk size because the gates do not read the store (tested
against the sequential definition for every address), so the chunk size is only a speed choice.
`--compile` compiles the per-chunk step (identical arithmetic).

**Task.** `flyprinc.lmtasks`, the dense recall generator in its contradiction-free capacity form:
T = 8P tokens, P distinct keys per sequence, 2-5 consistent presentations each ("key value"), 1-2
queries per key ("Q key value", loss on the value) placed 4-256 tokens after a presentation; no
re-bindings, no outliers. P in {32, 64, 128} (T = 256, 512, 1,024); P = 256 is not used because
the placement cannot fill it (1.4 presentations per key, 55% filler). Measured on 64 sequences per
P (`diagnostics/task_load.py`): queries per sequence 16.8 / 38.0 /
89.6; distinct keys already presented when a query comes (the memory load) median 27 / 55 / 90
(10th-90th percentile 12-31 / 24-61 / 42-96); distance from a query to its key's last presentation
median 25 / 45 / 102 tokens. The load is thus about 1.7, 3.4 and 5.6 times the dense key's 16
dimensions.
Held-out evaluation: 8 batches of 64 sequences, fixed seed.

**Wiring.** The fixed random expansion is drawn per seed and per layer (a generator seeded from the
run's seed and the layer index; `rewire` in `run.py`, tested), so the five seeds sample five
wirings as well as five training runs, as E1 drew one wiring per seed; a wiring is never tuned or
selected. (The pilots used the layer's default draw, one wiring shared by both layers.)

**Arms (same code).** At every P:
- `sparse`: Gated DeltaNet with the Kenyon-cell address (256 cells, fan-in 4, 16 on). THE HYPOTHESIS ARM.
- `dense`: Gated DeltaNet with dense 16-dimensional keys. THE PRIMARY CONTROL: equal active rows,
  equal parameters.
- `dense_c`: Gated DeltaNet with dense keys widened to 37 dimensions, the width at which its
  arithmetic per token equals the sparse arm's with a kernel that touches only the active rows and
  gathers the fixed expansion (6,432 against 6,464 multiply-adds per head and token; computed by
  `compute_matched_dk` from the cost table). The equal-compute control.
- `dense_wide`: Gated DeltaNet with 256-dimensional dense keys: the same memory state as `sparse`
  (256 x 32 per head), 16 times its active rows and wider key and query projections (more
  parameters). The equal-memory control.
- `product`: Gated DeltaNet with product-key addressing, the addressing of product-key memory
  layers: learned sub-key codebooks of 16 entries each for the two 8-dimensional halves of the key,
  one cell per pair (256), exact top-16 of the summed scores (identical to the two-stage search;
  tested), softmax weights, unit norm; 256 extra parameters per layer. The closest relative (learned,
  structured sparse addressing). Product-key memory layers keep their values as learned parameters,
  which cannot hold in-context bindings; what is compared here is their addressing, on the in-context store.
At P = 64 only (reported, not in the verdict):
- `sparse_learned`: the expansion learned (4,096 extra parameters per layer): is the random wiring what matters?
- `expand_dense`: the fixed expansion without winner-take-all (256 dense cells): is it the sparsity?
- `full_dense`, `full_sparse`: the address comparison in this repository's full layer (the gate
  senses the store, block-delayed with 16-token blocks; erase decoupled from write), the layer rung 1 uses.
- `attn_only`: two attention layers with full context: reference.
Training: 6,000 AdamW steps, batch 64, learning rate 1e-3, cosine schedule with 200 warm-up steps,
gradient clip 1.0, loss on query answers. Seeds {10, 11, 12, 13, 14}. Training batch s of seed n is
drawn with its own generator seeded by (n + 1, s) (`flyprinc.lmtasks.batch_at`) and generated ahead
by worker processes; the batches do not depend on the number of workers (tested).

**Metrics.** Primary: `acc`, held-out recall accuracy over all queries. Reported: accuracy by
memory load (0-15, 16-31, 32-63, 64-127, 128 or more keys already presented), single and confirmed
queries, final training accuracy, learning speed (`steps_to_0.9`: the first logged step at which
the 100-step training accuracy reaches 0.9; added before freezing because at P = 32, where both
arms end at ceiling, the full-layer pilot's sparse arm reached 0.95 training accuracy by step 1,000
and the dense arm 0.12; one seed, reported, not in the verdict), step time, peak GPU memory,
parameters, memory state per layer, active rows, arithmetic as implemented and with a kernel that
touches only the active rows.

**The pilots (exploratory, recorded before freezing).** Seed 0 (not a grid seed), 6,000 steps,
batch 64, learning rate 1e-3. (1) This repository's full layer (sensing gate, 16-token blocks), at
P = 32 (T = 256), dense vs sparse keys: both converged, held-out 0.998 and 1.000; training accuracy
at step 1,000 was 0.12 (dense) and 0.95 (sparse). (2) The grid's own arms and code path (Gated
DeltaNet gating, 64-token chunks): dense at P = 64 (T = 512) converged (training accuracy 0.91 by
step 2,000), held-out 0.983 (by memory load: 1.00 below 32 keys, 0.98 at 32-63 and at 64 or more);
GPU time 0.068 s per step; chunked-form error 6e-7. The same at P = 128 (T = 1,024): held-out 0.952
(0.94 at loads of 64 or more). (3) At d_k = 16 (the revision below), dense at P = 64: held-out
0.671, training accuracy 0.63-0.67 from step 2,000 on; 11 minutes of wall time with five data
workers while another job shared the GPU. (4) At d_k = 16, sparse at P = 64: held-out 1.000 (1.00
at every memory load, 64 or more keys included), training accuracy 0.35 at step 1,000 and 1.00 from
step 2,000; GPU time 0.254 s per step while rung 1's grid shared the GPU, 25 minutes; chunked-form
error 4e-7. This preview, a paired difference of +0.33 on one seed, is recorded before freezing; the
key width had been chosen from the dense arm alone before it existed. It does not separate the
explanations the controls exist for: the sparse arm also holds 16 times the dense arm's memory
state, which `dense_wide` matches, and needs 1.46 times its arithmetic, which `dense_c` matches.
(5) At d_k = 16, dense at P = 128 (T = 1,024): held-out 0.779 (1.00 / 0.99 / 0.87 / 0.75 by memory
load), training accuracy still rising at step 6,000 (0.39 / 0.58 / 0.75 / 0.78 at steps 2,000 /
3,000 / 4,000 / 6,000); 16 minutes. This is higher than the same arm at P = 64, where training had
stalled at 0.63-0.67 from step 2,000: in this regime a single dense run's outcome varies a lot, and
the rule's working assumption that accuracy does not rise with P does not hold for single seeds.
The choice does not depend on it: P = 64 and P = 128 each measured below 0.95. (6) At d_k = 16,
sparse at P = 128: held-out 0.999 (1.00 at every memory load), training accuracy 0.14 at step 1,000
and 0.99 at 2,000; GPU time 0.740 s per step while rung 1's grid shared the GPU, 74 minutes; peak
GPU memory 6.9 GB; chunked-form error 7e-7. In sum, at d_k = 16 the sparse arm was at ceiling at
both P piloted (1.000, 0.999) and the dense arm was not (0.671, 0.779), one seed each; which of the
explanations holds is what the controls decide. Every arm type piloted learned, which settles the
learnability gate.

**How the P values were chosen (rule fixed before any Gated DeltaNet-gated pilot result was
seen).** A P at which the dense control is at ceiling cannot show an advantage. The candidates are
P in {32, 48, 64, 96, 128} at 8 tokens per key (larger P cannot be placed at this density: 1.4
presentations per key at P = 256; placing them needs 70% filler and 4,096 tokens, the regime rung 1
found unlearnable in budget). The pre-registered P values are the three smallest candidates at
which the pilot's Gated DeltaNet-gated dense arm (the primary control) reaches held-out accuracy
below 0.95, taking accuracy as non-increasing in P (a P above one that qualified qualifies without a
pilot run of its own). If only two qualify, both are used and the verdict needs both; if fewer than
two qualify, this regime cannot test the hypothesis at d_k = 32, the design is revised before
freezing, and the revision is recorded here. The sparse arm's pilot results play no part in the
choice. (History: a first version of this rule, written when the sensing-gate pilot's dense arm
reached 0.998 at P = 32, listed {32, ..., 512} with longer sequences for large P; it was replaced by
this one before any Gated DeltaNet-gated result existed, once the placement limit was measured.)

**Revision under that rule (recorded before any result at a smaller key width existed).** No
candidate qualified at d_k = 32: the Gated DeltaNet-gated dense arm reached 0.983 at P = 64 and
0.952 at P = 128 (0.94 at memory loads of 64 or more keys), so two layers of four heads with
32-dimensional keys hold this task's loads almost perfectly, and the regime could not test the
hypothesis. As the rule prescribes, the design is revised: the key width is lowered until the dense
store is below its capacity, and the sparse code scales with it. The P values are fixed at
{32, 64, 128}; d_k is the largest of {16, 8} at which the pilot's dense arm reaches held-out
accuracy below 0.95 at P = 64 (accuracy is taken as non-increasing in P, so P = 128 then qualifies
too, and at least two of the three P values are below ceiling). With the key width the sparse code
keeps its expansion ratio and coding level: 16 d_k cells, d_k of them on, each averaging
round(7 d_k / 32) key dimensions (4 at d_k = 16); product keys over the same cells (codebooks of
the largest power of two not above sqrt(16 d_k) entries and its complement: 16 x 16 at d_k = 16);
the equal-memory control has 16 d_k dense dimensions; the equal-compute control is recomputed by
`compute_matched_dk`. The sections above that name d_k = 32, 512 cells and their costs are updated
to the chosen width before freezing. **Outcome:** at d_k = 16 the dense arm reached 0.671 at P = 64
(training accuracy stuck at 0.63-0.67 from step 2,000; held-out 0.99 / 0.86 / 0.63 / 0.67 at memory
loads below 16 / 16-31 / 32-63 / 64 or more keys), so d_k = 16, with 256 cells, 16 on, fan-in 4,
product codebooks 16 x 16, `dense_c` at 37 and `dense_wide` at 256 dimensions.

**Correctness gates (a failure voids the run).** (i) Unit tests: the chunked form reproduces its
sequential definition for every address; product keys equal the two-stage search; the fixed
expansion adds no parameters. (ii) The trained model's chunked layers reproduce their sequential
definition on a held-out batch to < 1% (measured at the end of every run). (iii) Learnability:
settled by the pilots. A voided run keeps its seed with undefined accuracy; a comparison containing
it is undefined and passes nothing (conservative); voided runs are reported with their cause and
never re-run with another seed. The verdict is computed only when every arm has every
pre-registered seed at every P; otherwise it reads INCOMPLETE.

**Thresholds (exhaustive).**
- SUPPORTED iff `sparse` - `dense` on `acc` > 2 x seed-SD with the paired-bootstrap 95% CI excluding
  0 at >= 2 of the 3 P values; FALSIFIED otherwise. A positive verdict is replicated with
  `--variant replication --seeds 20 21 22 23 24 25` before it is reported as supported.
- Reported with fixed rules, not part of the verdict: (B) `sparse` vs `dense_c` (equal compute),
  (C) `sparse` vs `dense_wide` (equal memory) and (D) `sparse` vs `product` (closest relative) at
  every P, and (E) `full_sparse` vs `full_dense` at P = 64, each classified BEATS (CI above 0),
  LOSES (CI below 0), MATCHES (CI within +/- 0.02) or INCONCLUSIVE.
- The plain-language statements are fixed in advance, each at the P values where A passed: "the fly
  address beats Gated DeltaNet at equal active rows and parameters" iff SUPPORTED; "and at equal
  compute" iff B is BEATS; "and at equal memory" iff C is BEATS; "matches the equal-memory store
  with a quarter of its arithmetic" iff C is MATCHES; "beats its closest relative" iff D is BEATS.

**What would count against the fly-specific reading, stated in advance.** If `sparse_learned` beats
`sparse` (CI above 0), the random wiring is not what matters; if `expand_dense` matches or beats
`sparse`, the expansion and not the sparsity is what matters; if `product` beats `sparse`, learned
product keys do the same job better. All are reported whatever the primary verdict.

**Cost accounting, stated in advance.** Equal active rows is not equal compute: at this width the
address itself (gathering 4 key dimensions into each of 256 cells, for the query and the key:
2,048 multiply-adds per head and token) costs more than the memory's reads and writes (1,536), so
even with a kernel that touches only the active rows the sparse arm needs 1.46 times the dense
arm's arithmetic per token (6,464 against 4,416 multiply-adds per head, the gate's 2,880 included;
the winner-take-all selection, comparisons rather than multiply-adds, is not counted). That is why
`dense_c` exists. In this implementation the expansion is a dense product and the 256-row store is
updated in full (35,648 multiply-adds per head and token), so the sparse arms run slower than the
dense arm; measured step times are reported as they are. Parameters (whole model): `dense` and
`sparse` 717,958, `product` 718,470, `sparse_learned` 726,150, `dense_c` 762,646, `dense_wide`
1,228,678. Memory state per layer: `dense` 2,048, `dense_c` 4,736, every 256-cell arm and
`dense_wide` 32,768. The equal-memory store needs 27,456 multiply-adds per head and token, 4.2 times
the sparse arm's with a sparse kernel (hence "a quarter of its arithmetic" above).

**Out of scope, stated in advance.** The full fly combination (the sparse address with two stores
and the history states) needs contradictions and memory load at once; it is rung 1c, designed after
rungs 1 and 1b report.

**Cost.** 5 arms x 3 P x 5 seeds = 75 runs plus 5 arms x 5 seeds at P = 64 = 25 runs. Measured in
the pilots while another job shared the GPU: 11-16 minutes per dense-key run at P = 64-128, 25
minutes per 256-row run at P = 64 and 74 at P = 128 (the 256-row store is updated in full here);
about 45 GPU-hours with the GPU shared, roughly 30 alone. Rung 1's grid is paused between its runs
(a STOP file) while this grid runs; sharing the GPU gained little throughput, and pausing a
resumable grid does not change its runs.
