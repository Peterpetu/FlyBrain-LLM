# REPORT — Does the fly connectome contain principles that improve sequence models?

<!-- Sections 1-3 and 3b were written before the results of their phase existed and are kept as
     written. Sections 4-7 were filled from experiments/*/summary.json (via RESULTS.md) as the
     pre-registered grids completed; section 0 is the current summary. -->

## 0. Where this stands, in plain language

Whether a principle from the fly's memory system makes the gated delta rule better. Two phases
null. The third, from first principles and with the rule "propose a state, not a gate", found:
(1) sparse addressing is a large, replicated win (0.99 vs 0.37 recall at equal active rows, which
is not equal compute: the expansion itself costs arithmetic);
(2) the fly-derived history states do nothing on a single store, and nothing at first on a
two-store memory; (3) a Bayes-optimal oracle showed every learned variant nine points below the
achievable, i.e. under-trained; (4) with ten times the training, the states on the two-store
memory give a small, replicated gain (+1.2 over the identical model without them, +0.7 over a
state-matched single store), entirely in protecting confirmed memories from one-off wrong
observations, with no change to overall accuracy. (5) Inside trained sequence models (rung 1b,
GPU), the Kenyon-cell address against Gated DeltaNet with 16-dimensional keys: recall 0.999 /
0.999 / 0.997 against 0.79 / 0.81 / 0.79 at 32 / 64 / 128 stored pairs, higher at all fifteen seed
pairs, but the pre-registered rule (a gain above two seed-spreads at two of three settings) passed
at one, so the verdict is FALSIFIED; and the controls show the gain is not specific to the fly's
code: a dense key with the same memory, product keys and a learned expansion all reach the same
ceiling, while the expansion without sparsity gains nothing. (6) In language (exploratory, two
seeds, 34M parameters, WikiText-103) the memory layers are not what recalls far facts: recall
beyond the attention window is 0.55 and stays at 0.52 when the memory is made to forget within 7
tokens, although the memory's learned decays could hold far facts; so the sparse address, a
capacity mechanism, changes nothing there (0.550 against 0.551) and costs perplexity and time. Its
cost advantage is arithmetic only: an exact sparse step is slower than the dense store on this GPU.
(7) Inside trained sequence models (rung 1, GPU, 95 runs), the history states are FALSIFIED on
every conjunct: +0.016 and +0.006 over the identical model without them at the two outlier rates
(0.3 seed-spreads, intervals through 0), a single store with the same memory ahead of the
two-store design by 0.040 and 0.094, and a plain single store with half the memory as good as
either two-store arm. The module-level gain of (4) did not survive training end to end. No fly
mechanism earned the pre-registered language-model test (rung 2), which was therefore not run.
Details and what would change these conclusions: docs/phase3/EXPERIMENT_PLAN.md ("C, synthesis",
"Rung 1 result") and RESULTS.md; the verdict of every pre-registered test is in section 4.

## 1. Question and stance

We asked whether any principle recoverable from the complete male *Drosophila* CNS
connectome (MaleCNS v1.0: 166,700 neurons, 10.5 M edges with >= 3 contacts, 101.4 M
contacts) improves a sequence model at matched compute, which principle, by how much, and
whether it scales. We did not try to build a competitive language model from a fly, and no
model here is "a fly brain". The whole graph is ~10^8 synapses, GPT-2-Medium scale, and
cannot be scaled without inventing wiring. A clean, controlled negative is a valid outcome
and several of ours are negatives.

## 2. First principles that shaped the design

**A connectome is structure, not computation.** Electron microscopy gives the wiring;
dynamics, plasticity and neuromodulatory state are supplied by the modeller. We list every
such assumption in LIMITATIONS.md.

**Wiring is meaningful only relative to neuron identity.** Once fly neurons are mapped to
arbitrary hidden units of a text model, only graph statistics can survive. This one axiom
predicts the pattern of prior results: FlyGM (graph as a body controller, identity and
task preserved) found real > shuffled; a frozen 499-neuron reservoir on speech emotion
(identity discarded) found real = scrambled (-0.04 pp); the char-RNN pilot (identity
discarded) found real > shuffle by 0.08 bits/char but real < dense. We therefore treated
"graph as recurrent mask" as a calibration experiment (H5) and put the weight of the
programme on *identified circuits with experimentally established functions* (H1, H2).

**Active inference as objective, not slogan.** Every test-time-memorisation layer of the
2025-26 frontier (Titans, DeltaNet, Gated DeltaNet, TTT, the Miras family) updates a
fast-weight memory by a gradient step on a prediction error plus a decay toward a prior,
i.e. a variational free-energy descent. The mushroom body is, experimentally, exactly
such a device: dopamine neurons carry a prediction error because output neurons feed back
onto them (Bennett 2021; Springer & Nawrot 2021), that error gates depression of the
active KC->MBON synapses (Hige 2015), and the alpha'3 compartment turns this into a
novelty detector (Hattori 2017). The central complex is a circular Kalman filter whose
bump amplitude encodes certainty (Kutschireiter 2023). The fly's *choices* inside the
fast-weight design space (sparse expansion keys with k-WTA, error-derived precision,
erase-type writes, one global inhibitory normaliser) are the hypotheses; the connectome
supplied their numbers.

## 3. What was extracted and verified (both hemispheres)

Mushroom body: 2,019 / 2,045 Kenyon cells; 60 / 61 PN types; mean 5.1 PN types per KC
(Caron 2013: 5-7; Litwin-Kumar 2017 predicts this fan-in maximises dimension); one
GABAergic APL contacting > 99.9 % of KCs; 48 / 49 MBONs in 32 labelled compartment
strings; 158 PAM and 8 PPL1 dopaminergic neurons; MBON->DAN feedback (339 / 300 edges,
2,861 / 2,605 contacts). Central complex: 46 EPG, 20 PEN_a, 22 PEN_b, 18 PEG, 42 Delta7
(hemibrain counts exactly). EPG->EPG recurrence is local; EPG->Delta7 input peaks at
180 deg while Delta7 output is local, so effective EPG->Delta7->EPG inhibition is 16x
weaker at the bump than opposite it; left- and right-bridge PEN neurons shift the bump in
opposite directions. The L_i = R_(9-i) glomerulus pairing was read off the Delta7 output
labels rather than assumed. One item could not be verified: the canonical ~45 deg PEN
offset, because this mirror has no per-synapse region labels and EPG->PEN contacts occur
both in the bridge (input) and the ellipsoid body (axo-axonal, at the shifted location).
The mirror itself was checked against the official Janelia annotation table (SHA-256
match; 98.7 % exact type-string agreement).

## 3b. Phase 2: the active-inference model (why, and what changed)

The phase-1 modules were ordinary sequence-model parts with a fly-derived constant or
matrix inside; the results (below) showed that the *matrices* carry nothing beyond their
statistics. Phase 2 (`flyai/`, `docs/aif/`, `experiments/aif_*`) starts from the dynamical
principles instead: perception as free-energy descent with inferred precisions,
plasticity gated by precision-weighted surprise, a belief state whose amplitude is its
certainty, homeostasis and near-critical recurrent dynamics, and action by expected free
energy. The connectome now enters as numbers and motifs only (fan-in by the sqrt rule,
5 % coding level, 30.8 % inhibitory units, 8-column rings, compartments with distinct
decay), never as wiring.

Two verification steps changed the design before any experiment ran. A structured
literature sweep (docs/aif/LITERATURE.md) established the dead ends we must not repeat:
predictive coding as a learning rule at scale, unbounded precisions, fixed connectome
masks, per-token expected-free-energy rollouts, and constant per-head precision that
disperses with context length. A three-proposal design panel with two judges
(docs/aif/DESIGN_PANEL.md) then found, and we confirmed in the code, that our residual
block's free energy was exactly quadratic (so its inner loop approximated a closed form
at several times the cost), that the arousal scalar nearly cancelled in the preconditioned
update (an experiment that would have compared the model with itself), and that the
write gate had the wrong sign. The v0.2 model (docs/aif/DESIGN.md) fixes all three and
adds the panel's other agreed changes. We report this because it is the kind of error a
pre-registration protocol exists to catch, and because it happened.

## 4. Results

Every pre-registered test in the repository, with its mechanical verdict (RESULTS.md carries the
tables, the per-arm learning rates and seeds, and the confidence intervals; the rule for a positive
verdict is a gain beyond two seed-SD with a paired-bootstrap 95 % CI excluding 0 against every
named control, replicated on fresh seeds before it is reported as supported). "Effect" is the
hypothesis arm minus its primary control, in seed-SD of that difference.

| test | hypothesis arm vs primary control (metric) | effect | verdict |
|---|---|---|---|
| H5 (phase 1) | real connectome mask vs degree-preserving shuffle, 256 / 512 / 1,024 units (bits/char) | -0.6 / -0.2 / -0.4 SD | FALSIFIED (needed 2 of 3 sizes; 0 of 3) |
| H1 | mushroom-body memory vs compute-matched Gated DeltaNet (MQAR, 8 pairs) | -5.6 to -6.3 SD | FALSIFIED, opposite direction |
| H1a / H1b / H1c | fixed sparse expansion / error gate / delta write vs their ablations | inside spread / -2.6 SD / inside spread | all three FALSIFIED (feature not necessary) |
| H1d | measured PN->KC wiring vs its degree-preserving shuffle and vs random fan-in-5 | -0.1 SD / -1.1 SD | FALSIFIED (the wiring adds nothing) |
| H2 | extracted ring attractor vs equal-size learned recurrent layer (compass tracking at 4x length) | -9.1 SD | FALSIFIED, opposite direction |
| H3, H4 | dcSBM prior; Dale's law and stability | | not run (pre-registered in phase 1; set aside after H5 showed the graph inert as a mask) |
| AIF-H1 (phase 2) | K = 3 free-energy inference steps vs K = 0, and vs a wider K = 0 (bits/char) | +0.7 SD / -0.6 SD | NOT SUPPORTED (null; the wider control had 1.94x the parameters) |
| AIF-H2 to H6 | precision gate, arousal, Lyapunov regulation, wind tunnel, EFE decoding | | not run: stopped for the pivot to phase 3 after AIF-H1 and the verification pass (docs/aif/VERIFICATION.md); their drivers and pre-registrations are kept |
| E0, D1 (phase 3) | signal availability of the candidate states; the re-binding regime in text | diagnostics | three states admitted; the regime exists at ~3 far re-bindings per 2,048 tokens |
| E1 | sparse random address vs dense keys at equal active rows (recall at P = 128) | +408 SD (0.993 vs 0.368) | SUPPORTED, REPLICATED on 6 fresh seeds |
| E2 | learned write gate with vs without the states, single store (accuracy, 4 regimes) | +0.3 SD | NOT SUPPORTED (null) |
| E3 | the same on the two-store substrate with a learned readout, 80 steps | +1.0 SD; -1.2 SD vs the state-matched store | NOT SUPPORTED (under-trained: 9 points below the Bayes-optimal oracle) |
| E4 | E3 at 800 steps, fresh seeds (class-balanced accuracy) | +6.2 SD (+0.011) vs free; +3.2 SD (+0.006) vs the state-matched store | SUPPORTED, REPLICATED on 6 fresh seeds (+0.012, +0.007) |
| rung 1b (GPU) | Kenyon-cell address inside a trained Gated DeltaNet layer vs GDN, P = 32 / 64 / 128 (recall) | +0.7 / +1.9 / +6.1 SD (0.9995 / 0.9993 / 0.9972 vs 0.786 / 0.809 / 0.789) | FALSIFIED (needed 2 of 3; 1 of 3). Fixed-rule comparisons: BEATS equal-compute GDN at every P; LOSES to the equal-memory dense key and to product keys by 0.0004-0.0027 at every P; the full layer with the sparse address BEATS the dense one at P = 64 |
| rung 1 (GPU) | two-store memory with vs without the history states, trained memory-only models, outlier rates 0.15 / 0.3 (accuracy on the decisive strata) | +0.3 / +0.3 SD (+0.016 / +0.006, CIs through 0); vs the equal-memory single store -0.8 / -4.2 SD (-0.040 / -0.094) | FALSIFIED on all three conjuncts (A: states vs free; B: states vs equal-memory store; C: a 1.2 SD regression on contradiction-free accuracy at rate 0.3). No replication grid |
| rung 2 | language modelling with embedded re-binding probes | | not run: reserved by rules fixed in advance for a mechanism rungs 1 or 1b supported |

Not pre-registered, reported as exploratory with no verdict (EXPERIMENT_PLAN.md, "Exploratory
language-model prototype" and "A 12-hour diagnostic block"): the 34M-parameter hybrid on WikiText-103 (dense
55.8 / 56.3 perplexity, sparse 58.3 / 59.3, two seeds each; recall beyond the attention window
0.548 / 0.550 against 0.550 / 0.551) and on TinyStories (5.84 for both arms, one seed each); the
memory-shortening diagnostic (far recall 0.550 to 0.518 with the memory forced to forget within
seven tokens); the selective-writes pilot; the natural-text load measurement (~400 distinct keys
per 2,048-token window); and the sparse-kernel timing (an exact k-sparse step at a quarter of the
arithmetic is 2.0 and 1.7 times slower than the equal-memory dense step at the two sizes measured).

## 5. Negative results

**The whole-graph connectome is inert as a recurrent mask (H5, pre-registered, final).**
At 256, 512 and 1,024 units, a character-level model whose recurrent layer is masked by
the real MaleCNS core graph (Dale's-law signs fixed, magnitudes learned) is
indistinguishable from the same layer masked by a degree-preserving shuffle (effects of
-0.6, -0.2 and -0.4 seed-SD; every 95 % CI includes 0) and from an Erdos-Renyi graph
(at 512 the random graph is in fact better by 2.1 seed-SD). A dense unconstrained layer of
the same width beats all graph conditions (by 21 and 4 seed-SD at 256 and 512; at 1,024 the
dense layer is better by 0.03 bits/char inside its own seed spread). The pre-registered
threshold, a real-vs-shuffle gap beyond 2 seed-SD at two of three sizes, was met at none.
The pilot's 0.08 bits/char real-vs-shuffle gap did not replicate under a tuned learning
rate per condition and three seeds. This is the outcome axiom A2 predicted: once neuron
identity is discarded, only degree statistics can survive, and those are matched by the
shuffle. All numbers: RESULTS.md, `experiments/h5_graph_mask/summary.json` (the per-run
directories of the CPU experiments are not committed; each summary carries, per condition and
learning rate, the seed mean, SD, minimum and maximum, the seeds and the rates swept).

**The measured PN->KC wiring adds nothing over random wiring with the same fan-in (H1d,
pre-registered, final).** With the real right-hemisphere PN-type-to-Kenyon-cell contact
matrix as the fixed expansion of the mushroom-body memory, multi-query recall is
0.977 +/- 0.004; with its degree-preserving shuffle 0.978 +/- 0.007; with a uniformly random
fan-in-5 expansion 0.981 +/- 0.003 (in-distribution; the harder 16-pair length-128 set gives
0.983, 0.990, 0.989). Real vs shuffled: -0.1 seed-SD; real vs random: -1.1 seed-SD, the
wrong direction and below threshold. This is the cleanest test of "does the actual wiring
matter" that the data allow, because here neuron identity is preserved at the level of
input channels; it reproduces, in a learning setting, Caron et al.'s anatomical finding
that the convergence is effectively random, and it means the connectome enters phase 2 as
numbers and motifs only.

**The mushroom-body memory does not beat dense fast weights at matched compute in the
easy recall regime (H1, pre-registered, final).** On multi-query associative recall with
8 pairs in 64 tokens, the sparse-expansion memory (1,024 KCs, 5 % active, error-gated
delta write) reaches 0.981 +/- 0.003, while the compute-matched Gated DeltaNet reaches
0.998 +/- 0.002, DeltaNet 1.000 and linear attention 1.000 (key dimension 65, chosen so
multiply-adds per token match): the hypothesis loses by 5.6 to 6.3 seed-SD, with CIs
excluding 0, on the primary metric, and by 1.3 to 5.9 seed-SD on the harder 16-pair,
length-128 set. The pre-registered support threshold (beat Gated DeltaNet by 2 seed-SD)
is therefore failed in the opposite direction. Within the family, judged by the pre-registered rules: replacing the error gate by a
learned input-dependent gate is *better* (+2.6 seed-SD; H1b falsified); the Hebbian write
(0.942 +/- 0.044) and the dense learned projection in place of the fixed sparse expansion
(0.940 +/- 0.096) both fall inside the seed spread of the hypothesis, which the
pre-registration defined as falsification of H1c and H1a. Both spreads are inflated by
one seed that failed to learn within the budget, so the honest reading is "the
biological feature is not necessary; whether it is harmful is not resolved". Removing
k-WTA (0.915 +/- 0.061) is worse by 1.1 seed-SD, the only ablation in the predicted
direction, and below threshold. Softmax attention at this size and budget (0.83 +/- 0.22) is the only
reference the mushroom-body layer beats, and only because one attention seed failed to
learn within 800 steps. Interpretation: with 8 associations and 65-dimensional dense
keys there is no interference to remove, so the sparse code's only claimed advantage
(low interference between many stored keys) cannot show; the pre-registered capacity
regime (40 pairs, `experiments/h1b_capacity/`) is where it would, and it remains unrun.

**The extracted ring attractor does not track state better than an unstructured layer of
the same size (H2, pre-registered, final; the strongest negative in phase 1).** On compass
state tracking trained at length 32 and tested at 128 (4x), the recurrent state built
from the 148-neuron EPG/PEN/Delta7 matrix with 25 learned block gains reaches
0.244 +/- 0.036 (chance 0.125); the pre-registered falsifier, the same 148 units with the
same velocity channels and a fully learned recurrent matrix, reaches 0.669 +/- 0.047
(-9.1 seed-SD against the hypothesis; CI excludes 0). The idealised canonical ring is no
better than the connectome ring (0.216 +/- 0.016), so this is not a defect of the measured
matrix but of the fixed-topology parametrisation itself: it cannot even fit the training
length (0.62 at length 32 vs 0.96 for the unstructured layer). A plain tanh RNN without
velocity channels (0.390) and two-layer attention (0.319) also beat both rings. Our
reading: a ring whose only free parameters are five-by-five block gains cannot absorb the
token-to-velocity mapping and the bump integration jointly, and the unit-spectral-radius
normalisation with bounded rates prevents the stable integration that the fly circuit
achieves with tuned synaptic strengths. The pre-registered threshold is failed in the
opposite direction. This finding propagated forward: the phase-2 model keeps an attractor
bank as a belief state, and descriptive no-attractor arms were added to AIF-H1 and
AIF-H6 before they ran.

**Per-token free-energy inference does not beat amortised inference (AIF-H1, pre-registered,
final; the only phase-2 grid that ran).** On TinyShakespeare, K = 3 inference steps per token
reach 2.800 +/- 0.011 bits/char against 2.826 +/- 0.035 for K = 0 at the same size (+0.7 seed-SD,
inside the spread) and 2.793 +/- 0.011 for the wider K = 0 control (-0.6 seed-SD, the control
ahead), with three seeds and a three-point learning-rate sweep per condition. The pre-registered
support condition, beating both controls beyond two seed-SD, is failed. The verification pass
(docs/aif/VERIFICATION.md) recorded two facts that bound the reading: the wider control was
over-provisioned (1.94x the parameters, 1.3x the measured FLOPs per token), so it is a stronger
model rather than a matched one; and the adaptive step count never stopped early (K = 3.00 in
every run), so the adaptive arm and the fixed-K arm are the same computation. The other five
phase-2 claims (AIF-H2 to H6) were stopped before completion for the pivot to phase 3; the first
AIF-H2 runs sat at chance because of a scoring defect the verification pass found and fixed
(F-1), and no result from them is reported. Phase 2's finding is the structural one: every
fly-derived mechanism it proposed was a fixed functional form inside a learned gate's hypothesis
class, and the design panel found three of its forms redundant or wrong before any grid ran.

## 5b. Phase 3, module level: the pivot to first principles

Both phases returned nulls for the same structural reason: every fly-derived mechanism was a fixed
functional form inside a backprop-trained model, and a learned gate with the same inputs contains
such a form as a special case (H1b: the learned gate beat the fly-derived error gate by 2.6
seed-SD; AIF-H1: +0.7 seed-SD for iterative inference against a control with 1.94x the
parameters). Phase 3 (docs/phase3/REFLECTION.md, AXIOMS.md, DESIGN.md) resets the axioms and
adopts one methodological rule learned from those nulls: *propose a state, not a gate*. A research
panel (three independent designs, two judges, a red team; docs/phase3/DESIGN_PANEL.md) chose a
minimal, nested extension of the gated delta rule with per-address states that record teaching
history, and the red team demanded two measurements before anything is trained. Both are done:

* **E0 (signal availability).** On sparse addresses the confirmation count, the familiarity-gated
  contradiction mass and the evidence channel carry history the delta rule's own intermediates do
  not, but only in the full-overwrite regime (beta = 1) without noise; on dense signed keys the
  evidence channel is dead at its stated floor (the red team's prediction, confirmed), and last-touch
  age carries nothing. Three states were admitted to the module-level test by the pre-registered rule.
* **D1 (the corpus regime).** Persistent re-binding of key-like tokens beyond a 512-token window
  occurs in WikiText-2 and PTB at about three per 2,048 tokens, 1.5-1.9x a within-key permutation
  null; consistent repeats are four times more frequent. The regime exists, as a minority.

The module-level grids are done. **E1 supports sparse addressing** at the discovery seeds (0.993 vs
0.368 recall at 128 pairs and equal active rows; the gain is the sparse code, not the state width;
replicated on six fresh seeds). **E2 is a null**: the learned write gate given the admitted states did not
beat the same gate without them (+0.0001, +0.3 seed-SD); per key class the mechanisms differ as the
predicted trade-off (a low fixed rate protects against outliers, a high one adopts re-bindings, the
learned gates split the difference), and the one mechanism that holds the previous observation, the
two-trace store, is the best on outlier keys and the worst overall because its readout is a fixed
rule. **E3 put a learned readout on that substrate**: a null at 80 training steps, with every learned
mechanism co-located nine points below a Bayes-optimal oracle, i.e. under-trained; **E4**, the same
nested test at 800 steps on fresh seeds, is **supported at its discovery seeds**: the states add
+0.011 over the identical architecture without them and +0.006 over a state-matched single store on
the class-balanced metric (the gain on outlier keys, the natural mix unchanged; small against the
nine-point gap to the oracle). Its six-seed replication passed (+0.012 and +0.007, CIs excluding 0),
so this is the repository's first supported and replicated result in favour of a fly-derived
state, and it is small: about one point of class-balanced accuracy, one ninth of the gap to a
Bayes-optimal oracle, with no change on the natural query mix. The odds stated in advance in
DESIGN.md section 7 were low, and the property the phase has that the earlier phases lacked is
that a null is now informative (a measured signal that a learned gate does not exploit). Both
module-level findings were then taken into trained sequence models (section 5c).

## 5c. Phase 3, system level: the two findings inside trained models (GPU)

**The Kenyon-cell address against Gated DeltaNet (rung 1b, pre-registered, final).** The
module-level win of E1 was tested where it has to hold to matter: inside a trained two-layer
memory-only model whose memory layer is a Gated DeltaNet layer (the exact reduction of section
3.3 of DESIGN.md; the same code with the states zeroed), on the capacity form of the recall task
(P = 32, 64 and 128 stored pairs in 8P tokens, key width 16, chosen by a rule from a one-seed
dense pilot before any sparse result existed), five seeds per arm with a fresh random wiring per
seed, 100 runs. The sparse address recalled 0.9995, 0.9993 and 0.9972 against Gated DeltaNet's
0.786, 0.809 and 0.789, and was higher at every one of the fifteen seed pairs. The frozen rule
asked for a gain above two seed-SD with the interval excluding 0 at two of the three P; it held
at P = 128 (6.1 SD) and not at P = 32 (0.7 SD) or P = 64 (1.9 SD), because the dense control's
spread is large (at P = 32 one of its five runs never left the plateau, 0.203 against 0.90-0.96
for the other four; the same seed is its worst at P = 64, 0.64), and the rule measures the gain
against that spread. The verdict is FALSIFIED and is reported as
such, with the warning about the control's spread that was on record before the rule was frozen.
The controls, classified by fixed rules, say what the gain is and is not: the sparse address
BEATS Gated DeltaNet at equal compute (key width 37) at every P (+0.012, +0.046, +0.055); it
LOSES to a dense key with the same memory (width 256) and to product keys by 0.0004-0.0027 at
every P, i.e. all three reach the same ceiling; a learned expansion is marginally better than the
random one (+0.0007); and the fixed expansion without winner-take-all gains nothing over dense
keys (0.810 against 0.809). So a sparse address over a larger store removes the capacity limit of
a 16-dimensional Gated DeltaNet store, reliably, and any of several ways of getting a larger store
does the same; the sparsity, not the fly's random wiring, is the operative ingredient. What the
sparse forms offer over the wide dense key is cost in arithmetic (one sixteenth of the rows
touched), and an exact kernel that exploits it (`experiments/p3_k1_sparse_kernel`) runs 1.7-2.0
times slower than the dense step on this GPU because gathers do not use tensor cores. The
efficiency claim the rung was built to test is therefore not met in wall-clock either.

**The history states on the two-store memory (rung 1, pre-registered, final).** E4's
small module-level gain was tested the same way, in trained memory-only models on the dense
rung-1 task (T = 256, 24 keys, two outlier rates, five fresh seeds, 95 runs: `state` against
`free`, the identical model with the states zeroed; `single_wide`, one store with the same memory
state; and, reported alongside, a plain single store, a Hebbian accumulator column, two fixed-gate
forms, the states on one store, each state path alone, Gated DeltaNet's input-only gate and an
attention-only reference). The hybrid form planned in DESIGN.md (attention plus memory,
contradictions beyond a 512-token window, T = 2,048) was drafted for the single store, rebuilt for
the two-store form, and then set aside because no arm, including full attention, learned it within
budget; the diagnostics that located the obstacle in the task format rather than in any mechanism
are in EXPERIMENT_PLAN.md ("Learnability"). A three-seed pilot of the memory-only form was
negative at every seed and was written into the pre-registration before it was frozen, so the grid
cannot be read as a first look. The grid completed with one pre-registration hash, identical code in
every run, no voided or diverged run and no dead-feature stop. The verdict is FALSIFIED, and every
one of its three conjuncts fails. (A) On the decisive strata (re-bound and after-outlier queries,
equal-weighted) the states add +0.016 at outlier rate 0.15 (0.793 against 0.777) and +0.006 at 0.3
(0.758 against 0.752), 0.3 seed-SD each, intervals through 0. (B) The single store with the same
memory is ahead of the state arm at both rates (0.833 and 0.851; -0.040, -0.8 SD, and -0.094, -4.2
SD with the interval excluding 0). (C) At rate 0.3 the states cost 1.2 seed-SD on contradiction-free
accuracy (0.986 against 0.993). Reported alongside: a plain single store with half the memory is
within the noise of the state arm at both rates (0.797 and 0.781); the Hebbian column lies between
`free` and `state`; neither state path alone (0.778, 0.760) nor the states on one store (0.763)
does better than its control (`free` 0.777, `single` 0.797) by more than 0.002; and at rate 0.3 the
states shift accuracy from after-outlier queries to re-bound ones (0.802 against 0.859, 0.713
against 0.644), the opposite of where E4 found its gain. The module-level result stands where it was
measured and does not transfer. A reading consistent with these numbers, not separately tested: at
module level only the memory's own gates and readout were learned, on explicit write and query
events, so the states supplied history those gates could not compute; in a model trained end to
end, the layers that produce the keys, values and gate inputs can carry that history themselves, and
the two-trace substrate the states need is itself no better than one store. Per-arm tables, the
per-stratum figures and the learning curves: `experiments/p3_s1_hybrid/` and RESULTS.md; the full
account: EXPERIMENT_PLAN.md, "Rung 1 result".

**Language (exploratory, not pre-registered, two seeds; no verdict).** A 34M-parameter hybrid
(six Gated-DeltaNet-gated memory layers and two sliding-window attention layers of 512 tokens,
context 2,048) was trained on WikiText-103 with a dense and with the Kenyon-cell address, and the
D1-regime probes (recall and re-binding of key-like tokens, near and far from the attention
window) were scored. The findings are diagnostic rather than comparative: the model recalls a
repeated fact 0.78 of the time inside the window and 0.55 beyond it, and the figure beyond the
window hardly moves when the memory is forced to forget within seven tokens (0.550 to 0.518),
although the learned decays could hold facts across the whole context. The memory contributes
inside the window and to perplexity; far recall at this size is not retrieved from it, so a
capacity mechanism has nothing to act on there, and the sparse address changes nothing far
(0.550 against 0.551), costs perplexity (58.3 / 59.3 against 55.8 / 56.3) and runs 2-2.7 times
slower. On TinyStories both arms reach 5.84 perplexity. A pilot on rung 1b's task with every
token written at full strength left the sparse gain intact (0.998 against 0.678), so that gain
does not depend on selective writing, and the language result is explained by the unlearned
retrieval rather than by the write policy. None of this is rung 2: the rules fixed in advance
reserved that test for a mechanism rungs 1 or 1b supported.

## 6. What was done after the first two phases, what remains, and what we would not do

**What the earlier version of this section proposed, and what happened to it.** After phase 2
this section named two next steps: the FlyAI GPU ladder frozen in `benchmarks/PREREG_tier2.md`
(a 20M-parameter 3:1 hybrid against its Gated DeltaNet twin on FineWeb-Edu), and the phase-1
capacity regime (`experiments/h1b_capacity/`). Neither was run in that form, for a reason the
record supports: phase 3's reflection (docs/phase3/REFLECTION.md) found that every phase-2
mechanism was a fixed functional form inside a learned gate's hypothesis class, so a GPU ladder
on it would have scaled a null. The programme restarted from first principles instead, and the
capacity question was answered where it matters, inside a trained Gated DeltaNet layer (rung 1b):
the sparse code does separate from dense fast weights at a 16-dimensional key width, at every
seed, and so do a wider dense key and product keys. The tier-2 pre-registration and the h1b
driver are kept as the record of what was planned; they are not queued.

**What remains open, stated so that it can be tested rather than hoped** (EXPERIMENT_PLAN.md,
"C, synthesis"). Three things would change the conclusions, each a real project with modest odds
and none a fly-specific result in prospect: a language model that has learned far retrieval
(longer training, a larger model, or an auxiliary recall objective) becoming capacity-limited,
where an address could then matter; a hardware-level sparse kernel that makes k-sparse access
cheaper than dense at equal recall, which is where the cost claim would have to be made; and a
setting with few, selectively written facts in very long contexts, which is the mushroom body's
own regime and not a small language model's. For the history states we have no further test to
propose: rung 1 was the regime their design named, and a single store with the same memory did
better. For any future grid we would adopt the paired verdict rule recorded before rung 1's
verdict (the paired-bootstrap interval excluding 0, a minimum effect fixed in advance, the
hypothesis arm ahead at all but at most one seed pair), which does not penalise a hypothesis for an
erratic control's own spread; it is not applied to any past grid (it would have passed rung 1b at
every P, and it would not change rung 1, whose intervals include 0 at both rates).

**What we would not do.** We would not spend further compute on the connectome as a
weight mask, a wiring diagram for a recurrent layer, or a fixed-topology attractor: H5,
H1d and H2 each falsified one form of that idea under matched controls, and axiom A2 in
`docs/aif/AXIOMS.md` says why (once neuron identity is discarded, only statistics can
survive, and those are matched by the shuffle). We would not reintroduce real wiring
unless a test preserves neuron identity at the level of input channels, as H1d did. We
would not run per-token expected-free-energy rollouts, unbounded precisions, or predictive
coding as the learning rule at scale: the literature already shows where those fail, and
our own design panel caught the same failure modes in our first draft. We would not run a
pre-registered language-model test (rung 2) on the history states or on the Kenyon-cell address
as they stand: the rules fixed before the rung verdicts reserved it for a supported mechanism.
We would not report a single-seed, single-learning-rate improvement as a finding, however
large, and the exploratory prototypes in this repository are labelled so that no one can. And
we would not call any model in this repository "a fly brain": the connectome enters as
counts and motifs, and the results above say that is the only way it can enter.

## 7. Reproducibility and the version this report describes

Every pre-registered experiment has a `PREREG.md` frozen by hash on its first run
(`.prereg.sha256`, committed; a later edit makes every later run refuse to start), a `run.py`
that computes its own verdict, a `summary.json` and `SUMMARY.md` under `experiments/`, and a row
per run in `runs_table.csv` for the GPU rungs, where `scripts/export_runs.py` also checked that
every run of a grid executed identical code at one pre-registration hash. The repository is
published as one commit per release; the development history is kept by the author, and
docs/history/README.md records, from that history, when each pre-registration was frozen
relative to its results and the code-identity checks made before publication. RESULTS.md is
rebuilt from the summaries by `scripts/build_results.py`. The CPU experiments run with
`experiments/<name>/run.py`; the GPU rungs and prototypes with the exact commands in
docs/phase3/GPU_RUNBOOK.md, on one 16 GB GPU, with the data fetched by the scripts in `scripts/`
and checked against `data/SHA256SUMS` and `data/manifests/`. The unit tests
(`python -m pytest tests -q`, 129 tests) cover the exactness of the chunked memory step, the
Gated DeltaNet reduction, the sparse kernel, the task generators, the probes and the harness.
Scale and the other limits of every claim are in LIMITATIONS.md; the version is in CITATION.cff
and CHANGELOG.md.
