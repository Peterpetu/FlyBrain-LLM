# Phase 3 experimental ladder: let the data pick the design

*Written before any phase-3 model code, after the biology briefs and three numerical studies
(docs/phase3/MEMORY_CAPACITY.md, ADAPTIVE_RATE.md, CONSOLIDATION_TOY.md). Every experiment below
is pre-registered in its own `experiments/p3_*/PREREG.md` before it runs, with matched controls,
>= 5 seeds, the 2 seed-SD + CI rule and a replication grid.*

*How to read this file. It is a chronological log, appended to and never rewritten: each section
was written at the time it describes, and words such as "now", "queued" or "running" refer to
that time. The opening sections are the plan as it stood before any result (P5 and P6 left the
module ladder after the smoke tests, see "Revisions"; P8 stayed deferred and was never run; the
multi-time-scale idea of P5 survives only as the per-head decay ladder of the system-level layer,
untested on its own); the results follow in the order they came; the current state of every claim
is summarised in "C, synthesis" near the end, and the verdict table in REPORT.md section 4 gives
the final status of every pre-registered test.*

## Two levels, one reason

Phase-1 H1b showed that a 40-pair recall task does not train at all in 800 CPU steps (every arm
at chance, loss at log 64): at CPU scale the *trainer* is the bottleneck, so a system-level test
cannot separate "the principle is wrong" from "the model never learned to use its memory".
The ladder therefore runs at two levels:

1. **Module level (CPU, now).** The memory module receives write events (key, value) and query
   events explicitly and is measured directly: capacity, interference, tracking under
   non-stationarity, retention across time scales, consolidation, savings after reversal. Only the
   module's own parameters (gate, decay, learned address, learned readout) are trained, by
   backpropagation through the module on the recall loss, so learned controls are as strong as
   they can be. Fixed principles need no training. This isolates the mechanism.
2. **System level (16 GB GPU, later).** The surviving principles are composed into a memory block
   inside a modern hybrid decoder and compared with Gated DeltaNet / Titans / LaCT twins at
   matched compute on language, long-context recall and continual-learning benchmarks
   (`benchmarks/PREREG_tier2.md`, to be revised for phase 3).

A principle that fails at the module level is dead; one that survives has earned a slot on the
GPU, not a claim.

## The principles and their killer experiments

| id | principle (from the biology) | regime where it must matter | matched control | primary metric | pre-registered threshold |
|---|---|---|---|---|---|
| P1 | sparse random addressing (expansion, fan-in 7, 5 % k-WTA) raises one-shot capacity at equal active rows per token | many associations (P from 16 to 512) | dense delta-rule keys with d_key = k active rows; learned dense expansion of the same width without k-WTA; learned sparse expansion | recall accuracy at P = 128 (curve secondary) | sparse_random beats dense by > 2 seed-SD, CI excludes 0; the learned dense expansion does not match it |
| P9 | divisive normalisation before the thresholded expansion improves the sparse code | same as P1 | P1 without normalisation | as P1 | ablation reported; not a support condition |
| P2 | error-corrected (delta) writes beat Hebbian writes | P1 stream | Hebb | as P1 | already supported (phase 1, numpy); re-measured as a sanity row |
| P3 | write rate as inferred precision (three-factor gate) tracks non-stationary, noisy streams without tuning | re-binding stream with observation noise 0.3 / 1.0 | best fixed rate from a sweep; a learned gate (MLP on key, value, read error) trained by backprop | post-re-binding recall (primary), overall recall | supported only if it beats the best fixed rate AND matches or beats the learned gate within seed spread |
| P5 | compartments with log-spaced time constants beat a single time scale at equal state | mixed recurrence intervals (items recurring every 4 / 32 / 256 events) | single compartment with learned decay and 4x value width (equal state); four compartments with learned decays | recall by interval class | supported only if the fixed log-spaced compartments beat the equal-state single compartment on the long-interval class beyond threshold without losing the short-interval class |
| P6 | spaced consolidation from a fast sparse store into a slow store retains beyond the fast horizon | 600-item stream, fast horizon ~200 | fast store alone with the same total state; consolidation with random selection; consolidation with one replay pass | retention of items older than 300 events | supported only if selected, multi-pass consolidation beats the equal-state fast store beyond threshold |
| P4 | non-destructive re-binding (new trace in a second compartment, readout by recency competition) gives savings after a reversal | A→B→A reversal stream | overwrite (delta rule, one compartment) | accuracy on reversed pairs within the first 3 presentations after the reversal | supported only if the two-compartment memory shows savings beyond threshold with no loss on unreversed pairs |
| P8 | phasor (rotation) addressing supports relational recall ("what came k after X") | relational queries over a stored stream | concatenated learned position keys | relational recall accuracy | deferred until P1 and P5 have results |

Closed on evidence, not tested again: connectome wiring as a prior; the ring-attractor matrix;
reservoirs as backbones; spiking; backprop-free slow weights; per-token iterative inference.

## Order and budget

P1 (+P9, P2 row) → P3 → P5 → P6 → P4 → P8. Each experiment is minutes to an hour on 4 CPU
cores at module level, so every grid runs 5 seeds and a full hyper-parameter sweep for the
controls. `scripts/build_results.py` gets one block per experiment with the mechanical verdict.

## What "supported" buys

A supported principle is composed into the phase-3 memory block (`docs/phase3/DESIGN.md`) with
its exact numbers; an unsupported one is dropped from the design and reported. The GPU
pre-registration is written only from supported principles.

## Revisions after the smoke tests (before any pre-registered run)

* **E2 (P3 + P4) merged.** Change detection, noise and outliers are one problem with two candidate
  mechanisms (rate as inferred precision; a labile trace trusted only after confirmation), so they
  are compared on the same streams (`experiments/p3_e2_tracking`). The smoke test showed the
  intrinsic trade-off: the precision gate wins under noise but is fooled by one-off outliers; the
  two-trace readout is robust to outliers but adopts genuine changes more slowly. The primary
  metric is therefore the overall accuracy on the natural query mix.
* **Confirmation must be content-based.** A per-row confirmation counter is polluted by other keys'
  writes on shared rows (rows are shared under sparse coding); confirmations are accumulated as
  vectors along the observed value and counted along the labile trace's own value.
* **P5 (time scales) and P6 (consolidation) leave the module ladder.** Under heavy one-shot load no
  memory can protect long-interval items without a relevance signal (repetition within the labile
  window, or a dopamine-like importance signal), and a fast store with more rows and no decay
  retains everything a consolidation scheme could; at module level both comparisons are either
  uninformative or ill-posed. Both are system-level questions (a language model has surprise and
  loss as relevance signals and a parametric slow store that generalises) and move to the GPU plan.
* **P8 (phasor addressing)** stays deferred until E1 and E2 have results.

## Results of the two pre-flight diagnostics, and the E2 revision they forced

Both were run before E1/E2 started (E1 and E2 are queued behind the phase-2 H1b grid on the same
CPU). Both are pre-registered and frozen; their full tables are in the experiment directories.

* **E0, signal availability (`experiments/p3_e0_signal`, numpy, 8 seeds, no training).** The red
  team's "K-minus-1" test: which candidate per-address states carry information about an address's
  teaching history that a learned gate could not already compute from the delta rule's own
  intermediates (error, read norm, key covariance)? Every address receives the same current event
  (a fresh value) after a history of {no write, two consistent writes, two inconsistent writes}.
  - On sparse 32-of-2048 addresses at write rate beta < 1 and without observation noise, the
    control already separates the histories (3-way accuracy 0.92-1.00): the delta store's residual
    read norm is smaller after a contradiction, so the store itself remembers inconsistency. At
    beta = 1 (full overwrite) the control is at the novel-vs-taught ceiling (0.64-0.68) and the
    states add +0.21 to +0.37: the confirmation count `nconf` most (clean-vs-contradicted AUROC
    1.00 at every beta), then the familiarity-gated contradiction mass `zc` (0.92-1.00), then the
    ATTEST evidence channel `a` at its stated floor (0.65-0.98, rising with beta). Last-touch age
    adds nothing (+0.007).
  - On dense signed keys (d = 128) the evidence channel is dead at the stated floor (dead fraction
    1.00, AUROC 0.50; kill conditions K2 and K3 fire in 24 of 24 configurations), as the red team
    predicted; `zc` survives weakly (0.64-0.73) and `nconf` moderately (+0.12 to +0.24 accuracy at
    beta >= 0.8). Anisotropic dense keys degrade everything, including the control.
  - The exact-cancellation "separation proposition" does not cover the gated delta rule: after
    +v, -v the residual is beta² ||v|| (K4 fires for beta >= 0.5). Only linear attention cancels.
  - Admitted to E2 by the pre-registered rule (> 0.05 incremental accuracy on sparse keys, mean
    over beta and mode): `a` (+0.065), `zc` (+0.065), `nconf` (+0.099). Two caveats stated with the
    result: E0 has no observation noise, and at beta < 1 it is a ceiling effect for the control,
    not evidence that the states are redundant under noise. E2 tests exactly that.
* **D1, the corpus regime (`experiments/p3_d1_corpus`, v2).** v1 was retracted on its first run
  as an instrument defect (36% of tokens counted as keys; 92-97% of repeats counted as
  "re-bindings" for keys and control alike; the frequency-matched control was empty on PTB, so
  the control clause fired vacuously). v1's files stay in the directory as the record. v2 counts
  PERSISTENT re-bindings (a key's next token changes and the new binding is used again) against a
  within-key permutation null. At W = 8192, far (> 512-token gap) persistent re-bindings occur at
  1.5-2.0 per 1,000 tokens on WikiText-2 and 1.4 on PTB, 1.5-1.9x the null (z 21-39), so text does
  contain temporally clustered re-binding beyond a 512 window; consistent repeats (the copy regime a
  context window serves) are 1.5-1.9x the null and about four times more frequent. The language
  rung stays in scope under the pre-registered rule, with the proportion on record: the mutable-
  memory regime is real but is the minority of long-range repeats.
* **E2 revised to the nested question (v2, frozen before its first run).** v1 compared fixed rules
  (precision gate, two-trace) with a learned gate, which the phase-1 and phase-2 nulls already
  answered (a fixed rule lies inside the learned gate's hypothesis class; docs/phase3/DESIGN_PANEL.md,
  "do not propose a gate, propose a state"). v2 asks whether the E0-admitted STATES improve a learned
  gate: `learned_state` (MLP gate on key, value, error, log squared error, log read norm, plus the
  slots nconf, a, zc) against `learned_free` (the same MLP, the slots read zero; equal parameters,
  same initialisation). The fixed rules stay as references with their original thresholds.
  `flyprinc.memory` gained the state accumulators (`Z`, `sig`, `Zc`) and the gate input slots; the
  E1 configurations are untouched (no state is added unless a gate asks for it).

## System level (GPU): the scaffold is built, the run waits on E2

`flyprinc/hybrid.py` implements the design's memory layer (docs/phase3/DESIGN.md 3.3-3.5) in a
hybrid decoder, with a sequential reference form and a chunked form whose gate-input approximation
is iterated to convergence (tests/test_hybrid.py); an independent verification pass against the
specification is recorded in docs/phase3/VERIFICATION_HYBRID.md, with every finding resolved. `flyprinc/lmtasks.py`
generates the rung-1 stream (re-binding and outliers with half of the queries beyond the 512-token
window). `experiments/p3_s1_hybrid/` holds the DRAFT pre-registration (frozen only by the first
real run) and a GPU-ready grid script with the five arms and a CPU smoke mode. The decision to run
it is gated on E2 (rung 0d); its cost is about 40 GPU-hours.

## Results of the module ladder (E1, E2)

Both grids ran on the pre-registered configurations (5 seeds; mechanical verdicts in RESULTS.md).

* **E1 (P1, sparse addressing): SUPPORTED and REPLICATED (six fresh seeds: +0.625, CI excluding 0).** At 128
  stored pairs and equal active rows, the fixed random sparse expansion with k-WTA recalls 0.993
  against 0.368 for dense keys (+0.625, CI excluding 0) and 0.377 for a dense learned expansion of
  the same width without k-WTA (the gain is the sparsity of the address, not the width of the
  state). The delta rule matters as much (sparse Hebb 0.708). Two descriptive results: a LEARNED
  sparse expansion is slightly better than the fixed random one (0.998, +2.8 seed-SD), so the
  random wiring is not necessary, only the sparse non-negative code; and divisive normalisation is
  inert on these Gaussian keys (exactly 0.000 difference). At 512 pairs the sparse store still
  recalls 0.74 (dense 0.10).
* **E2 (K0 at module level): NULL.** The learned gate given the E0-admitted states (nconf, zc,
  a) reached 0.8972 against 0.8971 for the same gate without them (+0.0001, +0.3 seed-SD, CI
  including 0). The prediction that the state arm would win on the outlier regimes was wrong
  (0.8597 vs 0.8591). Every learned gate landed within half a point of the best fixed rate
  (0.892) and of the precision gate (0.896, which beats the best fixed rate by 0.35 points: P3
  supported, small), and the four regimes' ceilings dominate the overall metric. Per key class the
  mechanisms differ enormously and in the predicted direction of the trade-off: on outlier keys a
  LOW fixed rate is best (0.78; the learned gates 0.58; the precision gate 0.006, which adopts every
  surprise) and on re-bound keys a HIGH fixed rate is best (0.70; the learned gates 0.51); the
  two-trace mechanism holds the previous observation and reaches 0.66 on outlier keys, but its fixed
  readout costs it everywhere else (0.850 overall, P4 not supported). The learned gates split the
  difference, as a single-store per-write gate must: with one store, "was the previous observation
  also deviating" is not in any of its inputs, states included.
* **What the null means and where it leads.** By the pre-registration this is "the admitted states
  carry nothing this gate needs on this task at module level". The doctrine's premise held (E0:
  the signal exists and is measured) but the candidate states did not convert into decisions,
  because the decision that matters (adopt after two consistent deviations, protect after one)
  needs a substrate that holds the previous observation, which one store does not. The two-trace
  substrate holds it and its fixed readout loses; the next module-level question is therefore the
  nested one on THAT substrate: a learned readout and learned per-trace gates over two traces, with
  and without the confirmation count (E3, pre-registered before it runs). Rung 1 on the GPU is not
  run on the E2 states, as pre-registered.

## E3 result: NULL, and every learned mechanism converges to the same value

`tt_learned_state` 0.6906 vs `tt_learned_free` 0.6896 (+0.0010, +1.0 seed-SD, CI marginal) on the
class-balanced primary; vs the state-matched single store 0.6919 (-0.0013, -1.2 seed-SD). The
two-trace substrate with a learned readout beats the FIXED two-trace readout by 3 points (0.690 vs
0.660: the fixed rule was the loss) but does not beat a state-matched single store (-2.2 seed-SD).
Four learned mechanisms (one store, one wide store, two traces with and without the states) land
within 0.2 points of each other on the class metric and within 0.2 points on the natural mix, and
they make the same trade: 0.57 on outlier keys (fixed two-trace 0.66), 0.52 on re-bound keys (fixed
two-trace 0.45). That is the signature of an information ceiling of the stream rather than of the
mechanisms: a query on a key whose last presentation deviated once cannot be resolved by any policy
until the key is presented again, and E2/E3's queries come every 50 steps while a key is presented
about once per 48 steps. A post-hoc oracle (exact Bayesian filtering per key with the generator's own rates,
`scripts/e2_oracle_ceiling.py`) was computed to check this reading, and it does NOT hold: the
oracle reaches 0.781 class-balanced and 0.929 on the natural mix, 9 and 3 points above every learned
arm. The stream has headroom; the four learned mechanisms converge to the same sub-optimal policy.
Per class the fixed mechanisms bracket the oracle (the precision gate matches it on the stable
regimes, the fixed two-trace matches it on outlier keys), so the representations can reach it; what
the learned arms share is the training procedure (80 Adam steps, three learning rates). The
discriminating follow-up, run as an EXPLORATORY variant (`--variant long800`, 800 steps, one
learning rate, three seeds, three arms; reported beside the pre-registered grid, never merged), asks
whether longer training moves the learned arms toward the oracle and separates them. If it does,
E3's grid was under-trained and the pre-registered E4 uses the longer budget; if the arms stay
co-located, the failure is representational and the states are not the missing piece at module
level. Rung 1 stays unrun.

## The exploratory long-training variant separates the arms; E4 pre-registered on fresh seeds

At 800 steps (lr 3e-3, seeds 0-2, exploratory) the state arm reached 0.6993 class-balanced against
0.6870 without the slots and 0.6924 for the state-matched single store, the gain on outlier keys
(0.62 vs 0.51 / 0.59) at a cost on re-bound keys. Small (about one point on a nine-point gap to the
oracle), but the first separation of a state arm from its nested control in this repository, and
the reason E3's 80 steps were a null: the learned arms were under-trained. This is an exploratory
observation and is not evidence; `experiments/p3_e4_two_trace_long` pre-registers the same test at
800 steps on seeds 5-9 with the learning rate pre-selected, and runs now.

## E4 result: SUPPORTED at the discovery seeds (small effect), replication running

On fresh seeds 5-9 at 800 steps: `tt_learned_state` 0.6987 class-balanced vs 0.6875 for the same
architecture with the slots zeroed (+0.0112, +6.2 seed-SD, CI excluding 0) and vs 0.6929 for the
state-matched single store (+0.0058, +3.2 seed-SD, CI excluding 0). Both pre-registered conjuncts
hold. The natural-mix accuracy is unchanged (0.898 for every learned arm), so the effect is a
re-allocation across key classes: the state arm is the best learned mechanism on outlier keys (0.62
vs 0.51 / 0.59 at sigma 0.3; 0.56 vs 0.43 / 0.40 at sigma 1.0) and the worst on re-bound keys (0.48
vs 0.53 / 0.51). The substrate alone does not beat the single store (-0.0054); it is the states on
the substrate that do. Against the oracle ceiling (0.781) the gain is about one point of a nine-point
gap: real, replicable so far, and small. The six-fresh-seed replication (seeds 10-15) is running;
until it reports, the result is "supported at the discovery seeds", not "supported".

## E4 REPLICATED (six fresh seeds): the first supported state result, and it is small

Seeds 10-15: `tt_learned_state` 0.6987 vs `tt_learned_free` 0.6867 (+0.0120, +13.3 seed-SD) and
vs `single_free_wide` 0.6920 (+0.0067, +8.8 seed-SD), CIs excluding 0; the same per-class pattern
as the discovery grid (outlier keys 0.61 vs 0.50 / 0.58; re-bound keys 0.49 vs 0.54 / 0.52; natural
mix 0.898 for all). Reported as SUPPORTED. What it establishes, no more: on the two-trace substrate
with a learned readout, the per-address states (confirmation count, contradiction masses) let the
learned mechanism protect confirmed bindings against one-off outliers at a small cost on fresh
re-bindings, for a net gain of about one point of class-balanced accuracy, one ninth of the gap to
the Bayes-optimal oracle. On a single store the same states did nothing (E2). The substrate alone
does nothing (E3/E4 substrate clause). It is the combination.

Consequence for the system level: the rung-1 draft tests the single-store form, which the module
ladder says is the wrong form. Before rung 1 can be run, the hybrid layer needs the two-trace form
(two content stores at different rates, per-trace learned gates, a learned readout over the two
reads with the state slots), with the same nested control and the same exact reduction. That is the
next implementation item; its pre-registration inherits E4's thresholds and the small-effect caveat.

## On the GPU (RTX 4080 SUPER, 16 GB): cost, a configuration defect, and rung 1b

The two-store layer was built (and adversarially verified) as planned. The first sessions on the
target GPU then established five facts before any pre-registered system-level run.

* **The step is launch-bound, and per-token sensing is too slow.** At full size (batch 16,
  T = 2,048) the two-store state arm took 9.9 s per step eager: the gate reads the store inside
  each chunk, which forced up to seven refinement rounds of dozens of small kernels. Measured cures:
  compiling the per-chunk step once (3.5x, identical arithmetic to 5e-7), and block-delayed sensing
  (the gates sense the stores as of each 64-token block start, which makes the chunked form exact in
  one round; 2.4% of key re-presentations fall in the same block, so almost no gate decision
  changes). Together: 0.43 s per step, 23 times faster. Memory: 8-9 GB.
* **The parity gate could not hold.** The state arm costs 1.7 times the free arm per step with
  per-token sensing, not the design's +11% MACs, because operation count and not arithmetic sets the
  time. The gate was revised before freezing to "recorded and reported, not voiding"; the hypothesis
  is tested at equal steps.
* **The learnability gate caught a real defect.** Full-context attention reached 0.8% on the rung-1
  task after 3,000 steps. A standard reference transformer learns the textbook MQAR task to 100% by
  step 2,000 on the same batches; the repository's model plateaued at 22%. Ablation located it in
  the input/output configuration (rotary positions, output head tied to a 0.02-std embedding);
  learned absolute positions, an untied head and embedding std 1 reproduce the reference, and with
  them memory-only models learn MQAR (98% one store, 100% two stores with the states). The memory
  layer itself was never at fault. Without the gate, every arm of the grid would have trained at
  chance for days.
* **One-seed hint, not evidence:** on textbook MQAR the hybrid with two stores and the states
  learned (100% by step 2,000) while the same hybrid without the states stayed on the plateau (22%).
  Escaping this plateau is partly luck; the seeded grid decides.
* **Rung 1b is built.** The Kenyon-cell address (a fixed random expansion of each 64-dimensional key
  to 1,024 cells, 7 key dimensions per cell, winner-take-all keeping 64 on) is an option of the
  memory layer, exact in the chunked form and with no extra parameters. Its draft pre-registration
  (`experiments/p3_s1b_sparse`) tests the largest module-level effect (E1) where it matters: P key-
  value pairs stored beyond the attention window, P in {64, 128, 256}, sparse against dense keys at
  equal active rows and parameters, with the equal-state dense store, a learned expansion, the
  expansion without sparsity, and the full fly combination (sparse address, two stores, history
  states) reported alongside.

## Learnability: the rung-1 task as drafted cannot be learned in budget; a dense task can

With the corrected configuration the real-size learnability check still failed: after 3,000 steps
at T = 2,048 every arm was at chance (full attention 0.11%, two stores with states 0.23%, without
0.25%; chance 0.20%). A controlled sequence of diagnostics (scripts and README in
`experiments/p3_s1_hybrid/diagnostics`) located the obstacle in the task and budget, not in the
fly mechanisms:

| test | result after 3,000 steps |
|---|---|
| textbook MQAR at 64 tokens, 16 pairs | learned (100%) by attention, memory-only and hybrid models |
| the same at 256 tokens, 64 pairs | not learned by attention (2-3%, the in-context plateau) |
| the rung-1 generator at 256 tokens, 8 keys, four position schemes | plateau (16-17%) or chance (learned absolute positions) |
| a length curriculum 128 to 2,048 tokens, 1,000 steps per stage | chance at every stage |
| five model types on two DENSE tasks (below) | learned by some, erratically |

The dense sweep (one seed each; d = 128; the hybrid's attention window shrunk to 32 tokens so the
memory layers must carry most answers):

| model | textbook MQAR, 128 tokens, 32 pairs | dense rung-1 task, 256 tokens, 24 keys |
|---|---|---|
| attention only | 3% | 5% |
| memory only, one store | 10% | 76% |
| memory only, two stores + states | 12% | 72% |
| hybrid, two stores | 75% | 9% |
| hybrid, two stores + states | 86% | 10% |

Reading: recall tasks of this kind have a long plateau before an abrupt transition (the standard
benchmark study trains on about 6 million sequences; this budget was about 100,000). Long,
filler-heavy sequences with sparse queries delay the transition beyond any affordable budget; a
dense format does not. Whether a given model escapes within budget is erratic, so arms can only be
compared at convergence, with seeds. Memory-only models learn the dense rung-1 task reliably
enough to be the clean test of the memory mechanisms (no attention layer can bypass the memory).
A three-seed pilot at 6,000 steps (memory-only, two stores with and without the states) runs now;
rung 1 is re-pre-registered around its outcome.

## Rung 1 re-pre-registered (v3), rung 1b redesigned (v2), and what their pilots showed

**Rung 1 (the history states).** The three-seed pilot settled learnability: every seed of both arms
converged (training accuracy 0.97 by step 4,000). Its preview was negative: with the states the
decisive-strata accuracy was lower at every seed (means 0.802 against 0.828). v3 is pre-registered
on memory-only models on the dense task, with 8-token sensing blocks (0.3% of key re-occurrences
fall in the same block, against 17.4% at the pilot's 32), fresh seeds 10-14, and the pilot's preview
written into the PREREG before freezing, so a FALSIFIED verdict is the expected outcome and the grid
cannot be read as a first look. An 8-token-block check (seed 0) confirmed that the state arm
converges (0.976 training accuracy at step 5,600).

**Rung 1b (the Kenyon-cell address).** Redesigned to test the address where models learn:
memory-only models on the capacity form of the dense task, where dozens of key-value pairs load
the memory at every query (median load 27 / 55 / 90 keys at P = 32 / 64 / 128). The control is
Gated DeltaNet itself (input-only gating, erase tied to write). Four controls separate the claims:
equal active rows and parameters (`dense`), equal arithmetic with an ideal sparse kernel
(`dense_c`), equal memory state (`dense_wide`) and the closest relative, product-key addressing
(`product`, new in `flyprinc.hybrid`: learned sub-key codebooks, exact top-k, identical to the
two-stage search). One accounting fact is stated in advance: equal active rows is not equal
compute, because gathering the key into the expanded cells costs about as much as the memory's
reads and writes at these widths.

**The pilots chose the regime, by rules fixed before the relevant results.** A P at which the dense
control is at ceiling cannot show an advantage, so the P values were to be chosen from the dense
arm alone. At key width 32 no P qualified: Gated DeltaNet held the task almost perfectly (0.983 at
P = 64, 0.952 at P = 128), because two layers of four heads give eight differently projected
stores. As the rule prescribed, the key width was lowered: at 16 the dense arm reached 0.671 at
P = 64, with accuracy falling from 0.99 to 0.63 as the load grows. So rung 1b runs at d_k = 16
(256 cells, 16 on). One observation is recorded as a hint, not evidence: in this repository's full
layer at P = 32, where both address types end at ceiling, the sparse address learned far faster
(0.95 training accuracy at step 1,000 against 0.12). Learning speed is therefore reported in both
rungs (not part of any verdict).

**Operations facts learned on the way (each fixed and tested).**
* The task generator, pure Python, was the bottleneck, not the GPU (0.8 s of wall time per step
  against 0.09 s of GPU time at P = 64). Batches are now drawn from per-step seeds and generated
  ahead by worker processes (identical batches for any number of workers), and the generator is
  1.65 times faster with draw-for-draw identical output.
* `torch.compile` keeps 8 graphs per function and then silently runs it uncompiled (about 5 times
  slower). A grid trains many configurations in one process, so the limit is now raised.
* On this Windows machine, killing a process that is using the GPU made the driver reset channels
  and crashed the other GPU job. Grids and pilots now stop gracefully on a STOP file.
* Several processes can share one grid (atomic per-run claims; a final pass collects everything
  and re-runs anything a crashed worker left).

## An interruption, recorded (30 September - 1 October 2026)

The session that had launched the jobs ended at about 22:00 on 30 September and took them with
it; the GPU then sat idle for 19 hours. State found on 1 October: rung 1b had 80 of its 100 runs
finished and intact; one run (`sparse`, P = 128, seed 10) had been cut at step 4,300 of 6,000 and
left no result. It is run again from the start with the same seed (the same batches and the same
wiring), which is the grid's ordinary resume behaviour and not a voided run: nothing about it was
observed beyond its training log. No frozen file or code changed; `scripts/export_runs.py` checks
that every run of the grid executed identical code. Long jobs are now started detached from the
launching session (`scripts/launch_detached.ps1`, logs in `logs/`).

## Rung 1b result (2 October 2026): FALSIFIED under its pre-registered rule; what the controls show

The grid completed: 100 runs, seeds 10-14, one PREREG hash, the code identical in every run, no
voided run (`experiments/p3_s1b_sparse/SUMMARY.md`, `runs_table.csv`, `figures/`). Held-out recall,
mean over five seeds (SD):

| arm | P = 32 | P = 64 | P = 128 |
|---|---|---|---|
| `dense`: Gated DeltaNet, 16-dimensional keys | 0.786 (0.327) | 0.809 (0.098) | 0.789 (0.034) |
| `sparse`: + Kenyon-cell address | 0.9995 (0.0004) | 0.9993 (0.0005) | 0.9972 (0.0022) |
| `dense_c`: dense, equal compute (37 dimensions) | 0.988 (0.009) | 0.953 (0.040) | 0.942 (0.026) |
| `dense_wide`: dense, equal memory (256 dimensions) | 1.0000 | 0.9999 | 1.0000 |
| `product`: product keys | 0.9999 | 0.9998 | 0.9990 |
| `sparse_learned`: learned expansion + winner-take-all | | 1.0000 | |
| `expand_dense`: the fixed expansion without winner-take-all | | 0.810 (0.060) | |
| `full_dense` / `full_sparse`: this repository's full layer | | 0.939 / 0.9999 | |
| `attn_only`: two attention layers | | 0.040 | |

**Verdict (A), computed by `run.py`: FALSIFIED.** The rule needed `sparse` - `dense` above 2
seed-SDs with the confidence interval excluding 0 at two of the three P values. It held at P = 128
(+0.208, 6.1 SD) and not at P = 32 (+0.213, CI [+0.052, +0.508], 0.7 SD) or P = 64 (+0.190, CI
[+0.132, +0.278], 1.9 SD). The intervals exclude 0 at every P and the sparse arm was higher at all
fifteen seed pairs, but the dense control's own spread is large (at P = 32 one seed never left the
plateau: 0.203), and the rule measures the gain against that spread. The rule was frozen with the
pilot's warning about that spread on record, and it stands: no replication grid is run, and the
verdict is reported as FALSIFIED.

**The fixed-rule comparisons, as classified by the script.** B, equal compute: `sparse` BEATS
`dense_c` at every P (+0.012, +0.046, +0.055). C, equal memory: `sparse` LOSES to `dense_wide` at
every P, by 0.0004, 0.0007 and 0.0027. D, product keys: `sparse` LOSES at every P, by 0.0004,
0.0006 and 0.0018. E, the full layer at P = 64: `full_sparse` BEATS `full_dense` (+0.060).

**What was stated in advance to count against the fly-specific reading, and what happened.** The
learned expansion beat the random one (by 0.0007, interval above 0): the random wiring is not what
matters. Product keys beat the Kenyon-cell address at every P (by under 0.002): a learned,
structured sparse address does the same job at least as well. The fixed expansion without
winner-take-all gained nothing over dense keys (0.810 against 0.809): the sparsity, not the
expansion, is what matters.

**Reading.** A sparse address over a larger store removes Gated DeltaNet's capacity limit at this
key width, reliably and at every seed. It is not a fly-specific advantage: a wider dense key with
the same memory and product keys reach the same ceiling, a few thousandths higher. What the sparse
forms offer over the wide dense key is cost: the same recall while touching one sixteenth of the
rows (a quarter of the arithmetic with a kernel that exploits it; in this implementation they run
slower). Learning speed, reported only: the arms that reach the ceiling reach 0.9 training accuracy
in about 1,100-1,300 steps; `dense` reached it in 4 of its 15 runs (all at P = 32, after 2,300 to
5,400 steps).

## Exploratory language-model prototype (not pre-registered; one seed; no verdict)

Asked for before the rung verdicts, as a prototype: does the hybrid train on real text, what does
the sparse store cost at language-model size, and does anything differ on long-range recall?
Model: 8 layers (6 Gated DeltaNet-gated memory layers, 2 sliding-window attention layers of 512
tokens), d 384, 33.8M parameters (14.5M outside the embedding), WikiText-103, 2,048-token context,
49M training tokens per arm (`experiments/p3_s2_lm/prototype.py`).

* **Dense keys (finished):** validation perplexity 55.8. Recall probes (top-1 accuracy of the
  value's first token): 0.790 when the previous occurrence is within 512 tokens (5,074 probes),
  0.548 when it is beyond the attention window (1,521 probes covered by a 2,048-token window);
  after a persistent re-binding 0.602 near and 0.513 far. So the model trains, and long-range
  recall has headroom: far is 24 points below near.
* **Sparse keys (1,024 cells, 64 on; finished 2 October after two interrupted attempts, run from
  the start without a resume):** validation perplexity 58.3, against 55.8 for dense keys (4.5%
  worse). Recall probes: beyond the attention window 0.550 against 0.548 (1,521 probes: no
  difference; one probe set's sampling error is about 0.013), near 0.755 against 0.790; after a
  persistent re-binding 0.516 against 0.513 far (634 probes) and 0.574 against 0.602 near.
* **Reading (one seed per arm, so a description, not a finding):** at this size the Kenyon-cell
  address did not improve long-range recall and was slightly worse on perplexity and on nearby
  recall. That fits what the rung-1b pilots showed for 32-dimensional keys: with 64-dimensional
  keys in six heads the dense store does not appear to be the bottleneck here, and where the
  memory is not the bottleneck the sparse address has nothing to add.
* **Cost:** in this implementation the sparse arm takes 2.4 times the dense arm's time per update
  (11.2 against 4.7 seconds for 16,384 tokens with the GPU shared; dense alone: 2.5).
* **A first dry run at batch 8 needed 20-25 GB on the 16 GB GPU** (the 50,257-way output is over
  3 GB per copy) and spilled into system memory; the prototype uses micro-batches of 2 with
  gradient accumulation and activation checkpointing (4.3-4.5 GB).

## Exploratory TinyStories prototype (not pre-registered; one seed; a fluency check, not a memory test)

Asked for as a second prototype. TinyStories' stories are about 205 tokens long, so nothing here
bears on long-range memory; it shows whether the model trains into fluent text and whether the
two address types differ in perplexity. Same 33.8M-parameter hybrid as above, 1,024-token context
(about five stories), 256-token attention window, 49M training tokens from the first 100M of the
training split (`scripts/fetch_tinystories.py`); runs and samples in
`experiments/p3_s2_lm/prototype_runs/tinystories_*`.

* **Dense keys (finished 2 October):** validation cross-entropy 1.765 nats per token, perplexity
  5.84 (307,712 validation tokens); training loss 1.73 at the end; 10,600 tokens per second with
  rung 1 sharing the GPU; 80 minutes. Three samples from "Once upon a time" (temperature 0.8,
  top-40) are in `samples.txt`: grammatical, story-shaped English with dialogue, and the lapses in
  consistency usual at this size (a character talking to himself, a present kept in the freezer).
* **Sparse keys (1,024 cells, 64 on; finished 2 October):** validation cross-entropy 1.765 nats,
  perplexity 5.84: the same as dense keys to three decimals (1.7647 against 1.7645). Its samples
  read the same way. It took 2.7 times the dense arm's time per update (4.1 against 1.5 seconds;
  208 against 80 minutes).
* **Reading (one seed per arm):** on short stories the address makes no measurable difference to
  language-model quality, which is what a task with nothing to remember across long distances
  should show; the sparse arm only costs more in this implementation.

## A 12-hour diagnostic block (3 October 2026, 01:00-13:00): plan fixed before starting

Rung 1b is falsified and the language prototypes show no gain, so the work now is to understand
why, from first principles, and to test the one advantage the sparse address could still have.
Three items, each with its criterion stated here first.

**A. Why the memory did not matter in language (exploratory diagnostic).** The WikiText-103
prototype recalled a repeated fact 0.79 of the time when its previous occurrence was within the
attention window and 0.55 beyond it, and the address made no difference. Two explanations: the
memory layers do the far recall and are capacity-limited (then a bigger store should have helped,
and it did not), or the far recall is not done by the memory at all (then no address can help).
Test: train the dense and sparse WikiText models again with a second seed and their weights
saved, then evaluate each with (1) the memory layers' outputs zeroed, (2) the attention window
shrunk to 64 tokens, (3) accuracy by distance to the previous occurrence. Criterion: if zeroing the
memory leaves far recall within 0.05 of its value, the memory is not what recalls far facts in
this model, and the capacity result of rung 1b could not have transferred. Also a second seed of
the dense-sparse comparison itself.

**A, result for the dense model (05:00; seed 1, WikiText-103, `prototype_runs/dense_s1/diagnostics.json`).**
Second seed of the dense arm: perplexity 56.3 (seed 0: 55.8), recall near 0.775 / far 0.550 (seed
0: 0.790 / 0.548), after a re-binding 0.592 / 0.511. Recall by distance to the previous occurrence,
as trained: 0.80 (under 64 tokens), 0.78 (64-256), 0.72 (256-512), 0.55 (512-1,024), 0.55
(1,024-2,048): it falls at the attention window and is flat beyond it. The ablations:

| condition | perplexity | recall near | recall far | re-binding near | re-binding far |
|---|---|---|---|---|---|
| as trained | 56.3 | 0.775 | 0.550 | 0.592 | 0.511 |
| memory shortened (half-life 7 tokens) | 80.6 | 0.563 | 0.518 | 0.439 | 0.465 |
| attention window 64 | 68.7 | 0.618 | 0.537 | 0.545 | 0.494 |
| both | 86.8 | 0.517 | 0.506 | 0.440 | 0.462 |

The criterion is met: with the memory shortened, far recall stays within 0.05 of its trained value
(0.550 to 0.518), so the memory layers are not what recalls far facts in this model; the 0.55 beyond
the window is what the language itself makes guessable (it is 0.51 with both memory and attention
cut). The memory does carry a great deal inside the window (near recall 0.775 to 0.563, perplexity
56 to 81), and its learned decays are long: the slowest head of every layer has a median half-life
of 800-4,000 tokens, with 1-2 fast heads per layer under 10 tokens. So the capacity to hold far
facts is there and the model does not retrieve them. That is why no address could help: rung 1b's
gain is in capacity, and this model's far recall is not capacity-limited but unlearned. Write
strengths are 0.2-0.6 for the median token: every token is written.

**A, result for the sparse model (11:00; seed 1, `prototype_runs/sparse_s1/diagnostics.json`).** Perplexity
59.3 (seed 0: 58.3), recall near 0.718 / far 0.551 (seed 0: 0.755 / 0.550), after a re-binding 0.556
/ 0.495; 6.9 s per update against the dense arm's 2.7-4.7 (337 minutes). Recall by distance, as
trained: 0.74 / 0.71 / 0.69 / 0.55 / 0.56. Ablations: memory shortened, far 0.498 (-0.053) and near
0.497, perplexity 89; attention window 64, far 0.532 and near 0.595; both, far 0.498. So the sparse
model's memory carries a little more of its far recall than the dense model's (0.05 against 0.03),
and that is all either carries; its learned decays are at least as long (slow heads with median
half-lives of 1,000-19,000 tokens). Two seeds now agree on the dense-sparse comparison in language:
the same far recall (0.550 / 0.548 and 0.551 / 0.550), lower nearby recall and higher perplexity
with the sparse address (0.775 / 0.790 against 0.718 / 0.755; 56.3 / 55.8 against 59.3 / 58.3),
and 2-2.7 times the time per update.

**Selective-writes pilot (05:00; `experiments/p3_k2_selective_writes/pilot_results.jsonl`, one seed).**
The hypothesis that the sparse gain depends on selective writing is NOT supported. On rung 1b's
task at P = 64 with every token written at full strength (a constant gate): dense 0.678, sparse
0.998; with the learned gate: dense 0.671, sparse 0.9999. The constant gate costs the sparse arm
nothing and the dense arm nothing either. The gain is a property of the address, whatever the
write policy; the language result is explained by the diagnostic above, not by write selectivity.

**B. The efficiency claim (engineering, measured).** Rung 1b showed the sparse address recalling
as much as the 16-times-wider dense store while touching one sixteenth of its rows; the
implementation does not realise this (it updates every row). Build an exact chunked step for
k-sparse codes (gathers and scatters over the active rows only), test it against the dense
implementation to 1e-4, and measure time per token on the GPU against the dense key of equal
width and the dense key of equal memory, at rung-1b sizes and at language-model sizes.
Criterion stated in advance: the claim "the fly address matches the equal-memory store at lower
cost" is upheld only if the sparse step is faster than the equal-memory dense step in wall-clock
time on this GPU, not only in arithmetic; the arithmetic counts are reported either way.

**B, result (01:20).** The exact k-sparse step is built (`experiments/p3_k1_sparse_kernel/sparse_chunk.py`:
the two n x n products from the active cells' values, the state reads as gathers at the active cells,
the update as a scatter-add; tests: matches the layer's chunked and sequential outputs to 1e-4 for the
random, learned and product-key addresses). The criterion is NOT met: on this GPU the step is slower
than the equal-memory dense step, eager and compiled alike (`bench.py`, medians of 15 interleaved
measurements, the GPU shared with two other jobs; one layer's forward pass):

| form (compiled) | rung-1b size: ms / forward | language-model size: ms / forward |
|---|---|---|
| dense, equal width (16 / 64 rows) | 7.4 | 26.6 |
| dense, equal memory (256 / 1,024 rows) | 21.4 | 77.2 |
| sparse, as implemented (every row updated) | 30.1 | 87.1 |
| sparse, exact k-sparse step | 43.8 | 129.7 |

Arithmetic per head and token tells the opposite story (rung-1b size: equal-memory dense 27,456
multiply-adds, the sparse step 6,464; language-model size: 201,536 against 45,888), but the dense
products run on tensor cores while gathers and scatter-adds in PyTorch do not, and the sparse arm
also pays for the expansion and the winner-take-all selection. So the efficiency claim stands only in
arithmetic, not in time, on this hardware with this software; a hand-written kernel might change
that, and that is where the claim would have to be made. (The eager figures and all counts are in
`bench.json`.)

**C. Synthesis.** A first-principles account of what held and what did not across phase 3, for
the front page and the report, written after A and B; and a lesson for the pre-registration rule
(a gain measured against an erratic control's own spread penalises the hypothesis for the
control's failures; a paired rule would not), proposed for future grids, never applied to past ones.

Rung 1's grid keeps running throughout (verdict expected after this block).

**Measured on the way (01:25): the memory load of natural text.** In the WikiText-103 validation
split, counting D1's key words (rare or capitalised), a 512-token window holds a median of 125
distinct keys (26 of them recurring within the window), a 2,048-token window 397 (121 recurring),
an 8,192-token window 1,182 (518 recurring). And the memory layer writes every token, keys or not.
So a language model's store is always oversubscribed (2,048 writes per window into 64 or 1,024
cells); it works as a recency-weighted memory governed by its learned decay and write gates, which
is why the gate statistics in block A are the decisive measurement, and why a capacity gain for a
few selectively written pairs need not transfer.

**C, the lesson for the verdict rule (written 01:30, before rung 1's verdict).** Rung 1b's primary
conjunct required the gain to exceed twice the larger of the two arms' seed spreads. When the
control is erratic (one of five Gated DeltaNet runs never left the learning plateau at P = 32, and
the others ranged from 0.90 to 0.96), its spread grows and the hypothesis is penalised for the
control's failures: at P = 32 the gain was +0.21 with a confidence interval excluding 0, yet 0.7
spreads. A paired rule would have read the same data differently: the paired-bootstrap interval
excluding 0, the mean paired gain above a minimum effect fixed in advance (0.02), and the
hypothesis arm ahead at all but at most one seed pair. Under that rule rung 1b would have passed at
every P. It is proposed for future pre-registrations; it is not applied to rung 1b, whose rule was
frozen with the pilot's warning about the control's spread on record, and whose verdict stands.

**C, synthesis (11:30).** What phase 3 established, from first principles, with the numbers behind it:

1. **The Kenyon-cell address is a real capacity mechanism and not a fly-specific one.** For a
   fast-weight delta-rule store with keys of a given width, a fixed random expansion followed by
   winner-take-all removes the capacity limit: module level 0.99 against 0.37 (E1, replicated),
   trained models 0.997-0.9995 against 0.79-0.81 at every seed (rung 1b). The ingredient is the
   sparsity: the same expansion without winner-take-all gains nothing (0.810 against 0.809). The
   wiring does not matter: a learned expansion does as well. And the gain is not unique: a dense key
   with the same memory (1.000) and product keys (0.999-1.000) reach the same ceiling. The verdict is
   FALSIFIED under its frozen rule because the control is erratic, a fact about the rule as much as
   about the mechanism (see the lesson above); the mechanism itself held at every seed pair.
2. **Its remaining advantage, cost, is arithmetic only.** The sparse address matches the equal-memory
   dense store while touching one sixteenth of the rows, 4 times less arithmetic, but an exact
   k-sparse step is slower than the dense store on this GPU (44 against 21 ms per layer forward,
   compiled), because dense products run on tensor cores and gathers do not.
3. **The history states gave nothing inside a trained model.** The module-level gain (E4, +1.2
   points, replicated) did not survive: in rung 1's completed first setting the states add +0.016
   (0.3 seed spreads, interval through 0) and a single wider store is better than the two-store
   design with or without them (0.833 against 0.793). Rung 1's verdict is expected FALSIFIED.
   [Added 18:50, when the grid completed: FALSIFIED on all three conjuncts at both rates; see
   "Rung 1 result" below.]
4. **In language, at this size and budget, far recall is not done by the memory.** Recall of a
   repeated fact falls at the attention window (0.80 inside, 0.55 flat beyond) and shortening the
   memory to a 7-token half-life moves the far figure by 0.03-0.05; the memory carries a great deal
   inside the window and in perplexity, and its learned decays are long enough to hold far facts.
   So the model has the capacity and has not learned the retrieval; the sparse address, a capacity
   mechanism, therefore changes nothing far (0.550 against 0.551, two seeds) and costs nearby recall,
   perplexity and time. Selective writing is not the reason (the pilot).
5. **Why, in one sentence.** The mushroom body solves a problem of few associations written on
   reinforcement and read by sparse codes; a small language model's memory writes every token and is
   limited by what it has learned to retrieve, not by how many addresses it can tell apart; and where
   address capacity does bind, engineering already has equivalent answers.

What would change these conclusions, stated so that it can be tested rather than hoped: a model
that has learned far retrieval (longer training, a larger model or an auxiliary recall objective)
becoming capacity-limited, where the address could then matter; a hardware-level sparse kernel
making sparse access cheaper than dense at equal recall; or a setting with selectively written
facts in very long contexts. Each is a real project with modest odds, and none is a fly-specific
result in prospect. The method held up throughout: the pre-registered rules, the learnability
gates and the controls each caught something that would otherwise have been reported wrongly.

## Rung 2 (language modelling): how its arms follow from rungs 1 and 1b, fixed before their verdicts

Written on 2026-09-30, while both grids run and before either verdict exists, so that rung 2
cannot be designed around the results. Infrastructure is built and tested, nothing is trained:
WikiText-103 tokenized with GPT-2 BPE (117.9M training tokens; `scripts/fetch_wikitext103.py`),
and the D1 regime's events as scoring positions for a language model (`experiments/p3_s2_lm/`,
D1's own word-level definitions): recall_far (a consistent repeat whose previous occurrence lies
beyond a 512-token attention window; 3,308 in validation, 3,690 in test) and rebind_far (the
confirming occurrence of a persistent re-binding beyond it; 1,358 and 1,469).

* **Rung 1b SUPPORTED and replicated:** rung 2 tests the Kenyon-cell address in a hybrid language
  model (Gated DeltaNet memory layers with sliding-window attention): sparse against dense keys at
  equal active rows and parameters, with the equal-compute and equal-memory controls and product
  keys. Primary metric: recall_far accuracy. Perplexity must not regress (a pre-registered margin).
* **Rung 1 SUPPORTED and replicated:** rung 2 includes the history states against the free
  control, primary metric rebind_far accuracy (the design's original rung 2).
* **Both SUPPORTED:** the combination (sparse address, two stores, states) is rung 1c first, on a
  task with contradictions and load together, before it enters a language model.
* **Neither SUPPORTED:** no fly mechanism has earned a language-model test. Rung 2 is not run on
  these mechanisms, and the report says so plainly.
* **Open feasibility question, measured before rung 2 is pre-registered:** the throughput of the
  sparse store at language-model size. In this implementation the whole expanded store is updated,
  so a sparse kernel may be needed for a fair cost comparison at scale.

## Rung 1 result (3 October 2026, 18:50): FALSIFIED on every conjunct; a single store with the same memory is ahead

The grid completed as pre-registered: 19 arm-rate settings, seeds 10-14, 95 runs, one PREREG hash
(`dfa2d0e2...`), identical code in every run (12 commits recorded, all with the same `flyprinc/`,
`flybrain/`, `run.py` and `PREREG.md`; `scripts/export_runs.py`), no voided run, no divergence, and
the dead-feature gate never fired (at the end of training the `nconf` feature was dead at no query
position in any arm that carries the states; `zc` at most 23% in `state` and 51% in `single_state`,
against the 90% stop condition). The chunked layers reproduced their sequential definition to
< 1e-5 in every run with a memory layer. 33 GPU-hours in total, 7-58 minutes per run depending on
what else shared the GPU. Numbers: `SUMMARY.md`, `summary.json`, `runs_table.csv`, `figures/` (all in
`experiments/p3_s1_hybrid/`) and RESULTS.md.

**The primary metric (held-out accuracy on the decisive strata, equal-weighted re-bound and
after-outlier queries), seed mean (SD) over 5 seeds.**

| arm | outlier rate 0.15 | outlier rate 0.3 |
|---|---|---|
| `state`: two stores, the history states in the gates and the readout (hypothesis) | 0.793 (0.049) | 0.758 (0.022) |
| `free`: the same model with the states zeroed (primary control, equal parameters) | 0.777 (0.047) | 0.752 (0.023) |
| `single_wide`: one store with the state arm's memory size (second control) | 0.833 (0.026) | 0.851 (0.020) |
| `single`: one store, key width 32 (Gated DeltaNet-like, learned gate) | 0.797 (0.024) | 0.781 (0.061) |
| `hebb_column`: two stores plus an uncorrected accumulator shown as a slot | 0.786 (0.058) | 0.756 (0.016) |
| `best_fixed_0.5` / `best_fixed_1.0`: two stores, constant gates | 0.703 / 0.634 | 0.737 / 0.648 |
| `single_state`: the states on one store (r = 0.15 only) | 0.763 (0.045) | |
| `state_slots_only` / `state_feats_only`: the two state paths separately (r = 0.15 only) | 0.778 / 0.760 | |
| `gdn_x_gate`: two stores with Gated DeltaNet's input-only gate (r = 0.15 only) | 0.710 (0.043) | |
| `attn_only`: two attention layers, full context (reference, r = 0.15 only) | 0.043 (0.007) | |

**Verdict, computed by `run.py`: FALSIFIED.** Every conjunct fails at both rates, and the rule needed
all three at every rate.
- A, `state` - `free` above 2 seed-SD with the interval excluding 0: +0.016 [-0.015, +0.049], 0.3 SD
  at r = 0.15; +0.006 [-0.018, +0.030], 0.3 SD at r = 0.3. Inside the noise at both rates.
- B, `state` above `single_wide` with the interval excluding 0: -0.040 [-0.082, +0.002], -0.8 SD at
  r = 0.15; -0.094 [-0.115, -0.073], -4.2 SD at r = 0.3. The single store with the same memory is
  ahead at both rates, with the interval excluding 0 at the higher one.
- C, no loss to `free` beyond 1 SD on the inert strata or the contradiction-free stream: at r = 0.3
  the states lose 0.006 on contradiction-free accuracy (0.986 against 0.993; 1.2 SD) and 0.004 on
  the inert strata (0.9 SD). At r = 0.15 both are flat (+0.001, +0.002).

**Reported alongside, with no verdict attached.**
- The pre-registered minimal alternative, `hebb_column`, lies between `free` and `state` at both rates,
  0.007 and 0.002 below `state` (intervals through 0); with the states' own effect inside the noise,
  the minimal-alternative reading has nothing to explain.
- The plain single store (`single`, key width 32, half the memory of the two-store arms) is within
  the noise of `state` at both rates (`state` - `single` = -0.004 [-0.045, +0.029] and -0.023
  [-0.053, +0.018]): in a trained model the second trace with a learned readout buys nothing
  measurable, with or without the states.
- Per stratum (the seed means): at r = 0.3 the states raise accuracy after a re-binding (0.713
  against 0.644 for `free`) and lower it after an outlier (0.802 against 0.859), the opposite split
  from E4 at module level, where the gain was in resisting outliers; at r = 0.15 both strata move by
  under 0.02. `single_wide` is ahead on both strata at both rates (0.763 / 0.902 and 0.816 / 0.886).
- The two state paths alone (`state_slots_only` 0.778, `state_feats_only` 0.760) are no better than
  `free` (0.777); the states on one store (`single_state` 0.763) are below the plain single store
  (0.797). The prediction of DESIGN.md section 5, item 5 (that `nconf` contributes more than the
  other states) cannot be tested when the states together contribute nothing.
- Gated DeltaNet's input-only gate on the two stores (`gdn_x_gate` 0.710) is 0.067 below the learned
  gate that also sees the stores (`free`), and the constant gates are lower still (0.63-0.74): the
  learned, store-sensing gate is what the two-store design needs, and it is what E3/E4 built.
- Learning speed (`steps_to_0.9`, the first logged step at 0.9 training accuracy): `single_wide`
  1,840 / 1,980, `free` 2,160 / 2,420, `state` 2,400 / 2,920, `single` 2,400 / 2,780, `hebb_column`
  2,080 / 2,340 (means over the five seeds at r = 0.15 / 0.3); the fixed-gate and input-only-gate
  arms reached it in 4 of their 25 runs, late (3,100-4,300 steps), and the attention reference in
  none. The state arm reached it later than `free` at both rates (by 240 and 500 steps on average)
  and ended in the same place.
- Size and cost: `state` and `free` have identical parameters (754,670); `state` holds 9,344 numbers
  of memory per layer (two stores plus the slots), `free` 8,192; `single_wide` holds 9,344 with
  839,254 parameters (state-matched, as pre-registered, not parameter-matched: 11% more parameters);
  `single` holds 4,096 with 752,006. Step time depended on what else shared the GPU (one arm varied up
  to 1.8-fold between its seeds); at the lowest load observed, `state` took about 0.25 s per step,
  `free` 0.16 s and either single store 0.14 s, so sensing the stores' history in 8-token blocks costs
  about 1.5 times the free step (recorded and reported, as the revised parity gate prescribed).

**Reading, in one paragraph.** The module-level gain of E4 (+0.012 class-balanced, replicated) did
not transfer into a trained sequence model. With the whole model learned end to end on streams of
the same kind (keys re-bound, and contradicted by one-off outliers), the history states add nothing
that the learned gates and readout do not already extract (+0.016 and +0.006, inside the seed
spread), and the two-store substrate they sit on is no better than a plain single store with half its
memory and is behind a single store with the same memory (by 0.040 and 0.094; that store has 11%
more parameters). This continues a pattern the module level already showed: there the two-trace
substrate without the states was below the state-matched single store too (E4: -0.005), and the
states lifted it above (+0.006, and +0.007 on the replication seeds); in the trained model they no
longer do. The pilot's preview, written
into the PREREG before freezing, predicted this verdict (there the states were slightly worse at
every seed); the grid has them slightly ahead, inside the noise: the effect is null in either
direction. This was the rung the phase-3 design was built for ("propose a state, not a gate"); the
odds stated in DESIGN.md section 7 gave it 0.4 of passing. No replication grid is run; no mechanism
from this phase has earned a pre-registered language-model test, and rung 2 is not run, as the rules
above prescribed.
