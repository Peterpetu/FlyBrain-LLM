# LIMITATIONS

Every assumption we supplied, and everything the connectome omits. Read this before
reading RESULTS.md.

## What the data are

* **Source.** MaleCNS v1.0 (Berg et al., Cell 2026), via the `garyb9/fly-drone` CC-BY-4.0
  mirror. We verified the mirror against the official Janelia release: the official
  body-annotation table (SHA-256 `2177e246…` as recorded in the mirror manifest) was
  downloaded from `storage.googleapis.com/flyem-male-cns/v1.0/` and matched; 98.7 % of
  cell-type strings agree exactly (the remainder are neurons the official table leaves
  untyped and the mirror labels by superclass), superclass agrees for 100 %.
* **Edges.** Only connections with >= 3 synaptic contacts are present (10,520,431 edges
  carrying 101,422,396 contacts). Weak connections are missing by construction.
* **Zero-weight edges.** 332,316 of the mirror's 10.5 M edges (3.2 %) carry a stored
  weight of 0 although no edge has weight 1 or 2 (consistent with the >= 3-contact
  inclusion rule). We treat them as present edges with unrecorded weight: they are in
  every structural mask and are initialised at the 3-contact minimum. Circuit statistics
  that count edges use the structural pattern; contact sums ignore them.
* **"Weights" are anatomical synaptic-contact counts.** They are not measured synaptic
  strengths. Every place we use them (initialisation of magnitudes, block normalisation of
  the ring matrix, PN->KC expansion weights) they stand in for unmeasured physiology. The
  mirror's `w_norm` scaling is arbitrary.
* **Signs are predictions.** Excitatory / inhibitory is inferred from predicted
  neurotransmitter (acetylcholine -> excitatory; GABA and glutamate -> inhibitory;
  octopamine, dopamine, serotonin, histamine, unclear -> treated as excitatory by the
  mirror's flag). Glutamate is inhibitory in the fly CNS via GluCl in most known cases but
  not all. Dopaminergic and other modulatory neurons are wired as ordinary signed edges,
  which is wrong physiologically; in our MB module dopamine is modelled explicitly as a
  gate instead.
* **Density discrepancy.** The task brief quotes "density 0.52 %"; we measure 0.038 %
  (10.5 M edges over 166,700 x 166,699 ordered pairs). All other quoted headline numbers
  (N, E, contacts, 30.8 % inhibitory, out-degree mean 63.1 / median 45 / max 7,749)
  reproduce exactly.
* **No FlyWire cross-check.** The brief asks for circuit claims to hold in FlyWire v783.
  `codex.flywire.ai` was unreachable from the authoring environment. The official MaleCNS
  annotation table carries a `flywireType` column for 85.9 % of neurons, which we used only
  to confirm that the MB/CX types are the same named types; we did not re-extract the
  circuits from FlyWire. This is an open item.
* **Male only, one animal.** All circuit numbers are from one male individual.

## What the connectome omits

Gap junctions (electrical synapses), neuromodulator diffusion and receptor identity,
synaptic strengths and short-term dynamics, plasticity rules, neuron biophysics
(time constants, spike thresholds), glia, developmental history, and behavioural state.
Any "dynamics" in this repository were chosen by us.

## Assumptions supplied by us, per component

### Whole-graph mask (H5)
* Core = top-N neurons by contact-weighted degree in the largest strongly connected
  component. Other selections (random SCC nodes, ROI-based) would give different graphs.
* Dynamics: discrete-time tanh RNN, one step per character. Real neurons are not that.
* Magnitudes initialised from log(1 + contacts), column-normalised; learned thereafter.
* Dense baselines share embedding, MLP and head; only the recurrent layer differs.
* On CPU the masked layer is computed as a dense masked matmul, so "compute-matched"
  refers to analytic FLOPs / dense implementation, stated per table.

### Mushroom-body memory (H1)
* The PN->KC matrix is aggregated from PN *neurons* to PN *types* (61 types on the right
  hemisphere), and the model's 61 learned "PN channels" are assigned to types in an
  arbitrary but fixed order. Only the *overlap pattern* between KCs (which KCs share
  inputs) is therefore biological; which input carries which information is learned.
* When fewer than 2,045 KCs are used, a seeded random subset of real KCs is taken.
* k-WTA at 5 % is a constant. In the fly, APL inhibition is graded, activity-dependent and
  partly local (Amin et al. 2020); 5 % is the commonly cited population sparseness (Lin
  2014; Honegger 2011).
* Plasticity: KC->MBON depression under DAN gating is modelled as a delta-rule step
  (gradient of the squared read-out error). This is a functional analogue, not the
  biophysical rule (which involves cAMP/PKA cascades with distinct timescales).
* "Dopamine = prediction error" is implemented as a sigmoid of the log squared error with
  two learned scalars per compartment. MBON->DAN feedback in the fly is cell-type-specific
  and includes lateral MBON->MBON inhibition, which we do not model.
* Two "compartments" (heads) each have their own expansion (different seeds) and their
  own PN projection; in the fly all compartments read one shared KC population and differ
  only in MBON/DAN identity, and there are 15 of them with distinct DAN valences and
  timescales. We model neither the shared code nor valence.
* The memory is reset at every sequence; no consolidation across sequences.

### Ring attractor (H2)
* Rate model, one Euler step per token, bounded rate tanh(relu(.)), unit spectral radius
  at initialisation, learned per-population-pair gains: all ours.
* Velocity input = tanh of a learned linear function of the token, applied as
  multiplicative modulation of left vs right PEN populations. In the fly, PEN activity
  is modulated by angular velocity signals arriving from outside the circuit.
* PEN shift verification: the ~45 degree PEN offset cannot be isolated because this
  mirror has no per-synapse ROI and EPG->PEN contacts occur both in the PB (input) and
  EB (axo-axonal, at the shifted location). We verified sign structure only.
* PEG, EL and ExR neurons are included in the matrix (PEG) or excluded (EL, ExR, all
  fan-shaped-body neurons); the goal-comparison circuit (PFL3, FC2, hDelta) is not
  modelled at all.

### Cell-type SBM (H3)
* Blocks = cell-type labels; most types have 1-2 neurons per hemisphere, so at the
  hemisphere's own N the SBM is nearly the identity and only becomes a generative prior
  at larger N. Sampled graphs preserve mean degree, not density.

## Experimental limitations

* **Scale.** Phases 1-2 and the phase-3 module ladder ran on 4 CPU cores: models of 10^4 - 10^6
  parameters, sequences of 32-512 tokens, character-level or synthetic data. The phase-3 grids
  and prototypes ran on one 16 GB GPU at 0.7-34M parameters (the last section below). Nothing
  here says anything about downstream LLM benchmarks.
* **Budgets.** 400-600 optimiser steps per run. Rankings at convergence may differ.
* **LR sweeps** were 3-point for hypotheses and 5-point for baselines, as pre-registered.
  Other hyper-parameters (weight decay, warm-up, width) were not tuned for any condition.
* **Seeds.** 3 per cell. With 3 seeds the 20,000-sample percentile bootstrap CI of a paired
  difference is essentially its sample range, so "CI excludes 0" alone is weak evidence (a
  null effect clears it about a quarter of the time); the binding pre-registered criterion
  is the effect in units of the larger seed SD (> 2), and differences inside the seed
  spread are never interpreted. The harness refuses to count a CI from fewer than 3 seeds,
  and every positive 3-seed verdict is labelled pending the 6-fresh-seed replication
  (`--variant replication`, reported side by side) before it may be called supported.
* **Cost matching** is analytic (multiply-adds per token). Wall-clock was measured once, for the
  phase-3 sparse address against dense stores on the GPU (`experiments/p3_k1_sparse_kernel/`): the
  arithmetic advantage did not become a time advantage there.
* **What is committed per run.** The CPU experiments (phases 1-2, the phase-3 module ladder) commit
  each grid's `summary.json` and `SUMMARY.md` (per condition and learning rate: seed mean, SD,
  minimum, maximum, the seeds, the rates swept, divergences) but not the per-run directories, so a
  reader can check every reported statistic and recompute every verdict, but not re-derive the
  statistics from per-seed values without re-running the grid (each takes minutes to an hour on a
  CPU). The GPU rungs commit one row per run (`runs_table.csv`, with the commit, the
  pre-registration hash and every metric); their model weights are not committed.
* **The development history is not published.** The repository is published as one commit per
  release. That pre-registrations were frozen before their results, and that every run of a GPU
  grid executed identical code, is recorded in docs/history/README.md from the history before
  publication; a reader of the published repository has that record and the content hashes, not
  the history itself. Two diagnostics (E0, D1) were committed together with their first results.


## Phase 2: assumptions supplied for FlyAI (active-inference model)

* **Free energy (v0.2).** F = pi_0 NLL + pi_prior/2 ||s - mu||^2 with mu the precision-ratio
  fusion of the core and attractor predictions and pi_prior = pi_1 + pi_3 (grouped); it is a
  Gaussian-latent, categorical-observation free energy with the complexity (KL) terms
  reduced to quadratic penalties toward the top-down means; there is no explicit entropy
  term for q(s) (a MAP / Laplace-style approximation).
* **Inference.** K gradient steps with a step size divided by the summed precisions
  (a Newton step on the quadratic part), initialised by a learned amortised correction;
  gradients of the training loss flow through the unrolled steps (backprop). The brain
  does not do this; predictive coding gives local rules, and we did not use them.
* **Precisions** are grouped (G groups of dimensions, not per dimension), tracked by Gamma
  posteriors with forgetting rho = 0.98, and enter only as bounded ratios (softmax over
  levels with learned offsets). Level-3 precision is read from the attractor amplitude.
  Arousal is one scalar in (0.25, 4) from the running token surprise; it scales the
  sensory precision and the write gate and never multiplies F (the v0 form was found to be
  nearly inert by construction and is documented in docs/aif/DESIGN_PANEL.md).
* **Mushroom-body memory**: fixed random expansion (fan-in 7 by the sqrt rule, not the
  measured 5), 5 % k-WTA, delta write, delayed hetero-associative key, one KC address
  shared by the compartments, log-spaced decay priors; the gate is a sigmoid of the
  variance-normalised surprise (v0.2). Phase 1 showed the measured PN->KC matrix adds
  nothing over random with matched fan-in, so the connectome enters here as numbers only.
* **Attractor bank**: canonical cosine rings (not the extracted matrix) by default; the
  extracted 148-neuron matrix is available via flybrain.modules.ring_attractor but the
  bank used in AIF experiments is idealised. Velocity is a learned tanh of the latent.
* **Core**: random sparse fan-in-32 pattern (not the connectome graph; phase 1 showed the
  graph is inert as a mask), 30.8 % inhibitory units with Dale's law, log-parametrised
  magnitudes, E/I balanced per target unit at initialisation (Rajan-Abbott), spectral
  radius 1 at gain 1, tanh rates. Divisive normalisation was moved OUT of the core: with
  it the dynamics are gain-invariant and criticality cannot be regulated through the gain
  (found empirically here; the normaliser cancels g). Normalisation remains at the
  expansion stage (k-WTA) only.
* **Criticality**: the largest Lyapunov exponent is estimated by one JVP power-iteration
  step on batch element 0 with an exponential average, every 8 tokens in the categorical
  model and once per 64-token chunk in the residual block (outside the checkpointed
  region); it is estimated in every condition and reported per run. The homeostatic gain
  update is a dithered extremum-seeking controller (the applied gain alternates
  base x (1 +/- 0.1) between regulation events, the slope d lambda / d gain is estimated by
  demodulating the single-step Lyapunov increment with the dither sign, and the base gain
  moves against the measured slope toward lambda = -0.02, slightly subcritical, with a
  dead-band of 0.01 and a clamp [0.05, 5]). It acts only in training mode; at evaluation
  the gain is frozen. In AIF-H1 (which ran with the earlier single-step slope estimator)
  the exponent sat at -0.04 to -0.24 while the gain drifted to 1.1-2.2: under token drive
  lambda(g) may peak below the target, in which case the dithered controller settles at
  the peak rather than at the target; whether regulation helps at all is what AIF-H4 tests.
* **Inference step**: Newton-style step eta / (pi_0 c_NLL + pi_prior + 1) with
  c_NLL = 0.5 sigma_max(W_out)^2 (a bound on the categorical curvature with respect to the
  normalised latent; the RMSNorm Jacobian is not included, so the step is conservative
  only up to that factor). Precisions are bounded ratios: the Gamma posterior's b is
  floored and every ratio is clamped to [1e-3, 1e3] wherever it is read (unbounded
  precisions are known to collapse; Zhai 2023). The level-0 precision of the residual
  block tracks the residual after fusion, which is self-referential (a confident fusion
  shrinks its own residual); the bound and the learned fusion offsets are what keep that
  loop from degenerating, and the GPU ingredient ladder is where it is measured.
* **Adaptive K never fired.** The stopping rule "stop when F falls by less than tol |F|"
  compares the drop with a free energy whose bulk is the calibration constant n/2 of the
  prior term, so at tol = 0.02 every token in AIF-H1 used all K_max steps (K_mean = 3.00 in
  every run): `flyai_K3` and `flyai_K3_fixed` were the same computation and the adaptivity
  claim is untested. The later experiments run fixed K, as their PREREGs state.
* **The "compute-matched" wide control of AIF-H1 is over-provisioned**: `flyai_K0_wide`
  (n = 192) has 1.88 vs 1.42 measured MFLOPs/token and 262k vs 135k parameters against
  `flyai_K3`; a loss to it is confounded by parameters, a win over it would have been clean.
  The AIF-H1 PREREG also says "4 rings x 16 units" where the code ran the v0.2 default of 8
  units per ring; the PREREG text is frozen and the discrepancy is recorded here.
* **Attractor amplitude**: threshold-linear units with a soft ceiling and a slow leak so
  the bump amplitude can encode certainty (Kutschireiter 2023); the rings are canonical,
  not the extracted matrix.
* **EFE decoding**: epistemic value is approximated by the free-energy drop that observing
  a candidate would cause (a one-sample proxy for expected information gain), computed for
  the top-k candidates only.
* **Scale**: all AIF experiments are character-level or synthetic, ~0.15-0.2 M parameters,
  300 optimiser steps, 3 seeds, 3-point LR sweeps for every condition (a deviation from
  the >= 5-point baseline rule of phase 1, forced by the ~1 s/step CPU cost).

## Phase 3: assumptions supplied for the state-memory design

* **The candidate states are definitions, not measurements of the fly.** The confirmation count
  (a vector accumulated along the observed value, reset on contradiction), the familiarity-gated
  contradiction mass and the evidence mass with its crosstalk floor are our formalisations of
  "how many times confirmed / how often contradicted / how much written" at an address. The fly
  fact behind them (depression-only, non-negative KC-to-MBON plasticity read by a dedicated
  compartment) licenses the existence of a history state, not these particular forms.
* **The confirmation criterion is relative to a tracked expected error** (1.5x a Gamma-tracked
  mean squared error, forgetting 0.9). It was chosen because an absolute tolerance failed under
  noise in the module smoke tests; the constant 1.5 was not swept and is pre-registered as fixed.
* **E0 has no observation noise**, and at write rates below 1 the control logistic is at its
  ceiling there; E0 ranks the states and rejects dead ones, it does not bound their value under
  noise. E2 does that at module level.
* **D1's proxy for "the fact stored at an address" is the next token** of a key-like word, and
  its null is a within-key permutation. It bounds the regime, it does not measure it; a
  kNN-LM-style probe over a trained model is the planned replacement.
* **The chunked form's gate inputs are an approximation** of the sequential specification when
  the gate senses the store per token; the system-level grids use block-delayed sensing (the gate
  sees the store as of each 8- or 16-token block start), which makes the chunked form exact in one
  round and is part of the model's definition. Every run measures its chunked-vs-sequential error
  (gate: < 1%; measured 1e-7 to 1e-6).
* **The rung-1 and rung-1b streams are self-designed**, with re-binding and outlier rates (rung 1)
  and a memory load (rung 1b) that fix the regime; the rates and loads are pre-registered and
  measured, but no external benchmark instantiates these regimes and none is claimed.
* **Scale** of the module ladder: numpy diagnostics and module-level grids at d_val = 16. The
  system-level rungs ran on the GPU; their limitations follow.

## Phase 3, system level (GPU): what the trained-model results do and do not cover

* **One GPU, small models.** Rungs 1 and 1b: 2 layers, d_model 128, 0.7-1.8M parameters, 6,000
  steps of 64 sequences; five seeds per arm (10-14), no replication grid because neither verdict
  was SUPPORTED. The exploratory prototypes: 34M parameters, 49M training tokens, one or two seeds.
  None of this is at the scale at which language-model claims are made, and no such claim is made.
* **The regime of rung 1b was chosen by a rule, after a pilot.** With 32-dimensional keys the
  dense control held the task's loads (0.98, 0.95 in a one-seed pilot) and the test could not
  discriminate; the pre-registered rule lowered the key width to 16, where the dense store is the
  bottleneck. The result therefore says what the sparse address does when key capacity binds, and
  the pilots and prototypes say that it did not bind at the other widths and tasks tried.
* **Rung 1 tested the states in memory-only models, not in the hybrid the design named.** The
  hybrid form (attention plus memory, contradictions beyond a 512-token window) could not be
  learned by any arm within budget, so the test moved to memory-only models on a dense task
  (EXPERIMENT_PLAN.md, "Learnability"); a hybrid trained long enough to learn the drafted task might
  behave differently. Its second control, the single store with the same memory, has 11% more
  parameters (state-matched, not parameter-matched, as pre-registered); the arms were compared at
  equal steps, not equal time (the state arm's step costs about 1.5 times the free arm's); and the
  ablations of the separate state paths ran at one of the two outlier rates.
* **The verdict rule penalised an erratic control.** The 2-seed-spread rule, frozen before the
  runs, failed rung 1b at two of three settings because the Gated DeltaNet control's own spread was
  large; the paired intervals excluded 0 at every setting. The verdict is reported as frozen; the
  rule's weakness is recorded and a paired rule proposed for future grids (EXPERIMENT_PLAN.md).
  Rung 1 does not depend on it: its intervals include 0 under either rule.
* **The prototypes are exploratory.** Not pre-registered, one or two seeds, a single model size
  and training budget; the ablation diagnostic establishes what *this* model does with its memory
  (far recall is not retrieved from it) and nothing about models trained long enough to learn it.
* **The sparse kernel's wall-clock comparison is PyTorch on one GPU.** Gathers against tensor-core
  products; a hand-written kernel could change the outcome, and the arithmetic counts stand.
* **The history-state and address designs are our formalisations** of fly facts (see above); a
  negative verdict for them is a negative verdict for these forms, not for the biology.
