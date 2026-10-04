# PREREG — P3-S1 (rung 1, v3): do the per-address states improve the two-store memory inside a trained sequence model?

*System level, GPU (RTX 4080 SUPER, 16 GB). Frozen by its hash at the first plain run of `run.py`
(never by `--smoke` or `--probe`). This is the third version of the rung-1 design; none was ever
frozen or run as a grid. Their history is part of the record (docs/phase3/EXPERIMENT_PLAN.md).*

**Why v3.** v1 (single store) was superseded when the module ladder found the states useful only
on two stores (E4). v2 (two stores in a 3:1 hybrid on 2,048-token sequences) failed its own
learnability gate: every model, full attention included, stayed at chance for 3,000 steps. The
diagnosis (scripts in `diagnostics/`) found two things. (1) A model-configuration defect (rotary
positions with an output head tied to a 0.02-std embedding) that kept even plain attention on a
plateau; fixed (untied head, embedding std 1). (2) Long, filler-heavy sequences with sparse queries
put recall learning far beyond any affordable budget, while dense recall tasks are learned; on the
dense rung-1 task memory-only models learn and hybrids with a short attention window do so
erratically. v3 therefore tests the states where the models demonstrably learn: memory-only models
(no attention layer can bypass the memory) on the dense rung-1 task.

**Hypothesis K0-S.** In a trained memory-only sequence model whose layers each hold TWO content
stores (a slow and a labile trace with their own learned write, erase and forget gates, and a
learned readout mixing the two reads at the query), giving the gates and the readout the
per-address history states (confirmation count `nconf`, familiarity-gated contradiction mass
`zc`) improves recall of the CURRENT binding of keys that were re-bound or shown a one-off outlier,
compared with the same model with the state slots zeroed (equal parameters, same initialisation)
and with a single store widened to the same memory state. Expected from E4 (module level,
replicated): a small effect, a re-allocation from re-bound toward after-outlier keys.

**Model.** `flyprinc.hybrid.HybridLM`, 2 memory layers, d_model 128, 4 heads, d_k = d_v = 32,
d_c = 8, SwiGLU MLP, rotary positions (no attention layer in the main arms, so they do not act),
untied output head, embedding std 1. Two stores per head (the gate biases start the first slow and
the second labile), learned readout (16 tanh units). Block-delayed sensing with blocks of 8
tokens (chunk = 8; the gates sense the stores and the states as of each block start; the writes,
the history updates and the readout stay per token; exact in one solve round per chunk, tested
against its sequential definition): in this task 0.3% of key re-occurrences fall in the same block
as the key's previous occurrence (68 of 19,591 measured; 5.5% at 16-token blocks, 17.4% at 32,
which is why the pilot's 32 was not kept). `--compile` compiles the per-chunk step (identical
arithmetic).

**Task.** `flyprinc.lmtasks`, dense setting: T = 256 tokens, 24 keys per sequence, 2-5
presentations per key, 1-2 queries per key placed near their key (about 15 queries per sequence,
46% filler tokens), re-binding probability 0.2 per eligible presentation (always confirmed by the
next presentation before any query), one-off outliers at rate r, r in {0.15, 0.30}. Strata:
single, confirmed, rebound, after_outlier. Held-out evaluation: 64 batches of 64 sequences, fixed
seed, plus the contradiction-free stream (no re-bindings, no outliers). (Raised from 8 batches
before freezing, on measurement grounds only: 8 batches held 90 re-bound queries, whose sampling
error of about 0.05 alone would match the pilot's whole seed spread; 64 hold about 720.)

**Arms (same code).**
- `state`: two stores, slots (nconf, zc, 0, 0) in the gates and the readout. THE HYPOTHESIS ARM.
- `free`: two stores, slots zeroed. THE PRIMARY CONTROL (equal parameters).
- `single_wide`: one store with d_k raised from 32 to 73 so that its memory state per layer equals
  the state arm's (2 d_k d_v + d_k d_c + d_k); rank- and state-matched, not parameter-matched.
- `single`: one store, d_k = 32 (the Gated DeltaNet-like layer with a learned gate). Reported.
- `hebb_column`: two stores plus one uncorrected accumulator of beta |e| per address shown as a
  slot (the minimal alternative). Reported.
- `best_fixed_0.5`, `best_fixed_1.0`: two stores with constant gates (slow 0.2, labile b), learned
  readout, no learned gate. Reported.
- At r = 0.15 only (reported): `single_state` (the states on one store), `state_slots_only`,
  `state_feats_only` (the two paths separately), `gdn_x_gate` (two stores with Gated DeltaNet's
  input-only gate), `attn_only` (two attention layers, full context: the reference).
Training: 6,000 AdamW steps, batch 64, learning rate 1e-3 (the rate at which every model learned
in the diagnosis), cosine schedule with 200 warm-up steps, gradient clip 1.0, loss on query
answers. Seeds {10, 11, 12, 13, 14} (fresh: seeds 0-2 were used by the pilot below). Training
batch s of seed n is drawn with its own generator seeded by (n + 1, s) (`flyprinc.lmtasks.batch_at`)
and generated ahead by worker processes; the batches do not depend on the number of workers
(tested). (The pilot drew its batches from one sequential generator; same distribution.)

**Metrics.** Primary: `acc_decisive`, the equal-weighted mean of held-out accuracy on the rebound
and after_outlier strata. Secondary, reported whatever the verdict: each stratum, `acc_inert` (mean
of single and confirmed), `acc_all`, contradiction-free accuracy `acc_cf_all`, final training
accuracy, learning speed (`steps_to_0.9`: the first logged step at which the 100-step training
accuracy reaches 0.9; added before freezing after a rung-1b pilot seed learned much faster with
the sparse address), dead-feature fractions, step time, peak GPU memory.

**The pilot (exploratory, recorded before freezing).** Memory-only models on this task, two stores
with and without the states, block 32, seeds 0-2, 6,000 steps: every seed of both arms converged
(training accuracy 0.97 by step 4,000), which settles the learnability gate. Its held-out decisive
accuracy was: with states 0.756 / 0.827 / 0.823 (mean 0.802), without 0.766 / 0.840 / 0.878 (mean
0.828) (seeds 0 / 1 / 2): the states arm was slightly WORSE at every seed (paired differences
-0.010, -0.013, -0.055), better after outliers at two seeds and worse after re-bindings at all
three. The pilot used a different block (32, with 17.4% of gate decisions missing the key's
previous occurrence) and other seeds; it is recorded here so that the grid cannot be read as a
first look, and it makes a FALSIFIED verdict the expected outcome. A convergence check at the
pre-registered 8-token blocks (seed 0, rate 0.15) confirmed learnability for both arms: training
accuracy 0.958 at step 3,000 and 0.976 at step 5,600 with the states (a GPU driver reset, caused by
killing another GPU process, ended that run before its evaluation), 0.942 and 0.969 without (first
reaching 0.9 at step 2,500; held-out decisive accuracy 0.729 on the 8-batch evaluation).

**Correctness gates (a failure voids the run).** (i) Unit tests: exactness of the chunked form and
of block-delayed sensing, the frozen single-store reference, equal parameters of `state` and
`free`. (ii) The trained model's chunked layers reproduce their sequential definition on a
held-out batch to < 1% (measured at the end of every run). (iii) A state arm whose nconf or zc
feature reads 0 at more than 90% of query positions at any log after step 500 is stopped and
voided. (iv) Learnability: settled by the pilot. A voided run (a failed gate, or divergence) keeps
its seed in the grid with undefined accuracies; a comparison containing it is undefined and passes
no conjunct (the verdict is conservative); voided runs are reported with their cause and never
re-run with another seed. The verdict is computed only when every arm has every pre-registered
seed at every rate; otherwise it reads INCOMPLETE (so a partial grid, or one of several processes
sharing the GPU on disjoint seeds, cannot print a verdict).

**Thresholds (exhaustive).**
- SUPPORTED iff, at EVERY outlier rate, `state` - `free` on `acc_decisive` > 2 x seed-SD with the
  paired-bootstrap 95% CI excluding 0; AND at every rate `state` - `single_wide` > 0 with the CI
  excluding 0; AND `state` does not lose to `free` on `acc_inert` or on `acc_cf_all` by more than
  1 seed-SD at any rate. The verdict is computed by `run.py` and rendered by
  `scripts/build_results.py`.
- FALSIFIED otherwise. There is no third outcome. The `hebb_column` comparison is reported
  alongside; if it matches `state`, the finding is the minimal-alternative one.
- A positive verdict is replicated with `--variant replication --seeds 20 21 22 23 24 25` before it
  is reported as supported.

**Cost.** 7 arms x 2 rates x 5 seeds = 70 runs plus 5 arms x 5 seeds at r = 0.15 = 25 runs;
measured on the target GPU (compiled, batch 64, T = 256): 0.12-0.25 s per step alone and about
0.25-0.36 s while another job shares the GPU, 15-30 minutes per 6,000-step run, about 35-45
GPU-hours in total (the grid shares the GPU with rung 1b).
