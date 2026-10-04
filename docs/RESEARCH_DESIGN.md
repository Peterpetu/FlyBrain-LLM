# Research design: axioms, first principles, and what we actually test

*Written before any experiment was run. This document is frozen; see PREREG files per
experiment for thresholds and RESULTS.md for outcomes.*

## 0. The question, stated honestly

Does the complete Drosophila connectome (MaleCNS v1.0, 166,700 neurons, 10.5 M edges at
>= 3 contacts, 101.4 M synaptic contacts) contain *computational principles* that improve a
modern sequence model at matched compute? Which principle, by how much, does it scale, and
what is the null?

Not the question: building a competitive LLM "from a fly". The graph is ~10^8 synapses
(GPT-2-Medium scale, 2019) and cannot be scaled without inventing wiring.

## 1. Axioms, reset from first principles

**A1. A connectome is structure, not computation.** Computation = structure x dynamics x
plasticity x neuromodulatory state. Electron microscopy gives the first factor only. Every
other factor is supplied by us. The science lives in *which* dynamics and plasticity we
supply and whether the structure then does anything a control structure cannot.

**A2. Wiring is only meaningful relative to neuron identity.** A synapse from LC4 to DNp02
carries a specific message because LC4 sees looming and DNp02 drives escape. If neurons are
mapped onto arbitrary hidden units of a language model, identity is gone and the only thing
the wiring can still contribute is *graph statistics* (degree distribution, sparsity, sign
fraction, modularity). Those are capturable by a generative graph model, so the literal
graph is not needed for them.

This single axiom explains all three prior results at once:

| Result | Identity preserved? | Task matches circuit? | Outcome |
|---|---|---|---|
| FlyGM (arXiv 2602.17997): whole-brain graph as RL policy for a fly body | yes (sensory in, motor out) | yes | real > shuffled |
| Oruk Labs: 499-neuron frozen reservoir on speech emotion | no | no | real = scrambled (-0.04 pp, CI -0.16..+0.07) |
| Pilot: 1,024-neuron core as char-RNN mask | no | no | real > shuffle by 0.08 bpc, real < dense RNN |

**Prediction from A2:** "graph as recurrent mask" (H5 below) is a weak prior that at best
reflects degree statistics and will not beat a tuned dense layer at matched compute. We keep
H5 as a *calibration and negative control* because it is cheap and it is the idea everyone
tries first. It is not the main path.

**A3. What transfers is the algorithm evolution embedded in identified circuits.** The
unit of generalisation is the cell type (11,770 types here; 8,069 isomorphic across sexes),
and the object of study is a *circuit motif whose function is experimentally established*.
Two such motifs have the strongest functional evidence in the fly and the clearest gap in
current ML:

* **Mushroom body (MB).** ~50 olfactory PN types -> ~2,000 Kenyon cells (KCs) per hemisphere
  by sparse convergence (5-7 inputs per KC; Caron 2013; Litwin-Kumar 2017 shows this fan-in
  maximises representational dimension), APL feedback inhibition holding ~5% of KCs active
  (Lin 2014), KC->MBON synapses *depressed* on coincidence of KC activity and dopaminergic
  (DAN) input (Hige 2015), and DANs that encode a *prediction error* because MBONs feed back
  onto them (Bennett 2021; Springer & Nawrot 2021; Felsenberg 2017). The alpha'3 compartment
  is a novelty detector: novel inputs drive MBON-alpha'3 strongly, repetition depresses the
  synapse, so the response encodes familiarity (Hattori 2017).
  In ML terms this is a **fast-weight associative memory with (i) a sparse-expansion key
  space, (ii) an erase-type (depression) write, (iii) a surprise-gated learning rate**.
* **Central complex (CX).** EPG neurons hold a heading bump on a ring; PEN neurons shift it
  by angular velocity; Delta7 neurons provide global (cosine-profiled) inhibition
  (Turner-Evans 2017; Hulse 2021). The circuit is a near-optimal *circular Kalman filter*:
  bump amplitude encodes certainty and decays without evidence (Kutschireiter 2023, PNAS).
  PFL3 neurons compare heading with a goal held in FC2/hDelta to produce steering
  (Westeinde 2024; Mussells Pires 2024).
  In ML terms this is a **belief state with explicit precision, updated by learned velocity,
  read out against a goal**: state tracking with uncertainty and native length
  extrapolation.

**A4. Active inference is the right *objective*, not a slogan.** Perception = update
beliefs to minimise prediction error plus complexity (KL from prior); learning = the same on
slower parameters; action = choose outputs that minimise expected error against preferences.
The fly circuits in A3 are, experimentally, the three ingredients: prediction-error-gated
plasticity (MB DANs), a precision-carrying belief state (CX bump amplitude), and
goal-versus-state comparison to drive action (PFL3). We do not import Friston's formalism
wholesale; we use one consequence of it that is exact and testable:

> A fast-weight memory M updated by a gradient step on
> F_t(M) = precision_t * || read(M, k_t) - v_t ||^2  +  lambda * || M - M_prior ||^2
> is a variational free-energy descent (accuracy + complexity). Titans, DeltaNet,
> Gated DeltaNet, linear attention and TTT are all instances of this with particular
> choices of read(), precision, and prior (the Miras framework, Behrouz et al. 2025).

The fly's MB makes *specific* choices inside that design space that ML has not tried
together: keys pass through a fixed sparse expansion with k-WTA (KC layer), precision is a
compartment-specific prediction error computed by MBON->DAN feedback (not a global scalar),
the write is depression from a default drive (erase-dominant), and normalisation is a
single global inhibitory feedback neuron (APL). **Those choices are the hypotheses.**

**A5. 2026 context.** Production hybrids (Qwen3-Next / Qwen3.5, Kimi Linear) interleave
Gated-DeltaNet-style fast-weight layers with a few full-attention layers; Mamba-3 and
TTT-E2E push the same test-time-memorisation direction. The open weaknesses that motivate
them are exactly in-context associative recall at scale (MQAR), state tracking, and length
extrapolation. That is where an MB/CX module would have to earn its place, at matched
compute, against a tuned fast-weight baseline. We never claim state of the art.

## 2. Hypotheses (pre-registered thresholds live in each experiment's PREREG.md)

* **H1 (MB memory, main).** A residual layer implementing sparse expansion -> k-WTA ->
  surprise-gated fast weights beats a compute-matched dense fast-weight baseline (linear
  attention / DeltaNet) on multi-query associative recall (MQAR), and the gain is destroyed
  by replacing the sparse expansion with a dense learned projection. Sub-hypotheses:
  H1a sparse-expansion keys; H1b prediction-error gate; H1c depression-type write;
  H1d **the measured PN->KC matrix vs a random matrix with the same fan-in** (the cleanest
  "does the real wiring matter" test, at the level where identity is preserved).
* **H2 (CX belief state).** A ring attractor with the extracted EPG/PEN/Delta7 structure and
  learned velocity input beats a plain RNN of the same size with the same velocity input on
  state tracking with a >= 4x train/test length gap. Ablation: remove ring topology.
* **H5 (graph as mask, calibration).** Real core graph vs degree-preserving shuffle vs
  Erdos-Renyi vs parameter-matched dense vs compute-matched dense as char-LM recurrent
  masks, across >= 3 sizes. Predicted (A2): real > shuffle by a small margin, dense wins.
* **H3 (cell-type SBM prior)** and **H4 (Dale's law / E-I stability)** are gated on H5:
  only worth their compute if H5's scaling curve shows the graph prior surviving scale.

## 3. Controls that every claim must carry

Degree-preserving shuffle (directed swaps, >= 10 |E| attempts); Erdos-Renyi at matched N, E;
parameter-matched dense baseline; compute-matched dense baseline (FLOPs, stated); >= 3 seeds;
mean +/- SD and paired bootstrap CIs; baselines tuned with an LR sweep at least as wide as
the hypothesis; feature ablations for any positive; every number traced to a run directory,
config, seed and commit.

## 4. What this environment can and cannot run

All experiments in RESULTS.md were run on 4 CPU cores with no GPU. They are therefore small:
synthetic capability benchmarks (MQAR, state tracking) and character-level language
modelling on public corpora. These are the standard *architecture-research* benchmarks
(Zoology/MQAR; parity and modular state tracking; length extrapolation), not downstream LLM
leaderboards. `benchmarks/` ships configs for the GPU-scale runs (50-350 M parameters on
FineWeb-Edu, lm-eval-harness tasks) that would be needed before any downstream claim. We
did not run those and make no claim about them.
