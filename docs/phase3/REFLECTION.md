# Phase 3 — reflection before the redesign

*Written after two pre-registered phases came out null and before any phase-3 code exists.
Purpose: say exactly what was learned, diagnose why both designs failed from first principles,
reset the axioms, and lay out the branches that remain, so that the phase-3 design is chosen
for reasons and not for fashion.*

## 1. The record

| phase | thesis | verdict (matched controls, 3 seeds, pre-registered thresholds) |
|---|---|---|
| 1 / H5 | the whole-brain graph is a useful recurrent weight mask | null: real ≈ degree-preserving shuffle ≈ Erdős–Rényi; dense wins |
| 1 / H1d | the measured PN→KC wiring beats random wiring of equal fan-in | null (−0.1 seed-SD) |
| 1 / H1 | a mushroom-body memory beats dense fast weights at matched compute | negative (−5.6 to −6.3 seed-SD); a learned gate beats the error gate |
| 1 / H2 | the extracted ring-attractor matrix tracks state better than an unstructured layer | strongly negative (−9.1 seed-SD) |
| 2 / AIF-H1 | K free-energy inference steps per token beat amortised inference | null (+0.7 seed-SD vs same size; loses to a wider control) |
| 2 / AIF-H2..H6 | precision gate, arousal, criticality, EFE decoding, Bayesian tracking | stopped before completion for the pivot; first H2 runs at chance |

The code of phase 2 was verified line by line against its specification (docs/aif/VERIFICATION.md),
so these are results about the ideas, not about bugs.

## 2. Why both phases failed, from first principles

**Category error 1: a connectome is a prior for the tasks that shaped it.** The fly connectome
is a compressed description of a sensorimotor controller for one body in one ecological niche:
odours, visual motion, gravity, walking, flight. Its synapse-level structure encodes the
statistics of that world. A prior helps only when the target task shares the statistics the
prior encodes; text shares nothing with olfactory glomeruli. What can transfer are architectural
*invariants* (fan-in, sparsity, E/I ratio, time constants), and phase 1 showed those act only
through their statistics, which a random graph with the same statistics also has. So: **wiring
transfers nothing; statistics transfer as hyper-parameters, not as structure.** This is now
established for this project and is not revisited.

**Category error 2: a fixed biological rule bolted onto an end-to-end trained model.** Every
phase-2 ingredient (error-gated writes, precision-weighted fusion, gain regulation) was a fixed
functional form inserted into a network trained by backpropagation. A learned gate with the
same inputs contains the fixed rule as a special case, and backpropagation finds it. At matched
parameters the biological rule is therefore at best equal and typically worse. An inductive bias
pays only when data are scarce relative to the hypothesis class *and* the bias matches the task;
ours matched the fly's tasks, not ours.

**Category error 3: measuring what the fly brain was never optimised for.** The fly brain's
objectives are survival-relevant competence under brutal constraints: about 1.4×10⁵ neurons,
microwatt power, no offline training set, one-trial learning of rewarded associations, continual
learning in a non-stationary world, robustness to noise and damage, development from a 13,000-gene
genome. Our benchmarks (bits per character, associative recall at 8 pairs) measure none of these.
A design aligned with the fly brain should be expected to win, if anywhere, on *those* axes:
continual adaptation, long-horizon memory, sample efficiency, compute per token. It should not
be expected to beat the transformer's inductive bias (global permutation-invariant mixing trained
by backpropagation at scale) on perplexity per FLOP, and no biological architecture ever has.

**On active inference specifically.** The free-energy principle is a normative description of
what any self-maintaining system does in the limit, not an algorithm. The algorithmic content we
tested, iterative inference on top of amortised inference, has a known negative result: the
amortisation gap closes with training, so the inner loop buys nothing a trained network cannot
already do in one pass (AIF-H1 confirmed it). Precision weighting is a gate, and learned gates
already exist. Expected free energy is an *action* policy, relevant to agents, not to next-token
quality. Active inference gave this project a vocabulary, not an edge. The one piece with
algorithmic bite is *learning as Bayesian filtering of the parameters*: an inferred precision on a
weight is a learning rate, which is exactly what dopamine sets, and exactly what test-time-learning
models lack a principled rule for. That piece survives into phase 3; the rest does not.

## 3. The axiom reset

The following replace docs/aif/AXIOMS.md for phase 3. They are stated as biology and then as the
engineering constraint each one imposes.

**A1. Evolution is the pretraining; the lifetime is the test.** The genome (≈13,000 genes) is a
compressed program that instantiates a competent network; within a lifetime the fly learns by
local, neuromodulator-gated plasticity, and consolidates during sleep. *Mapping:* backpropagation
on a corpus plays the role of evolution (there is nothing unbiological about it at that level);
what transformers lack is the **lifetime learning system**: local fast plasticity, modulated
learning rates, multi-timescale compartments, offline consolidation. That system is where a
fly-aligned design can add something the current best models do not have.

**A2. Fast learning is local and three-factor.** Within a lifetime no global error signal is
transported across the brain. Kenyon-cell → MBON synapses change as a product of presynaptic
activity, a postsynaptic/eligibility term and a dopamine signal that encodes prediction error and
valence. *Constraint:* every within-context update of the model's memory is a local rule with a
modulated rate; the rate is a state variable inferred from prediction errors, not a constant.

**A3. Memory is sparse, high-dimensional, compartmentalised and consolidated.** Random expansion
to ~2,000 cells at 5 % coding level gives near-orthogonal addresses (low interference);
compartments with distinct dopamine neurons forget at distinct rates; labile memory (ARM) and
consolidated memory (LTM, protein synthesis, sleep-dependent) are two stages. *Constraint:* the
memory system has at least two stages, addresses are sparse, timescales are explicit, and there
is an offline consolidation step from fast state into slow parameters.

**A4. The learning rate is a quantity the animal infers.** Dopamine, octopamine and internal state
set how much a surprise changes a synapse; a stable world lowers the rate, a changed world raises
it. *Constraint:* precision-weighted updates (the surviving piece of active inference).

**A5. Continuous variables live on phasors, discrete items on sparse codes.** The central complex
represents heading as a sinusoid over a ring and does vector arithmetic by phase shifts; the
mushroom body represents items as sparse binary codes. *Observation:* rotary position embeddings
are already a phasor code; the fly's contribution here is to *anchor* phasors by Hebbian learning
and to do arithmetic on them, which is a candidate mechanism for long-range structure and for
memory addressing by position.

**A6. Compute is bounded and sparse.** Five percent of the memory is active per event; energy is
microwatts. *Constraint:* per-token memory cost O(k·d) with k ≪ n, not O(n·d).

**A7. Measure the claim.** Benchmarks must test continual learning without forgetting, retention
beyond the context window, one-shot in-context binding, adaptation speed and compute per token,
against compute-matched test-time-learning controls (Gated DeltaNet, Titans, LaCT, TTT), with the
standard zero-shot suite as a no-regression check. Perplexity per FLOP against a transformer is
a regression check, never the claim.

## 4. The branches, assessed from these axioms

| branch | what it is | verdict for phase 3 |
|---|---|---|
| active inference, deeper | inference as filtering; EFE policies | keep only: inferred precision as learning rate; EFE for agentic decoding tasks. Not an architecture. |
| memory-based / test-time learning | fast weights updated at test time (TTT, Titans, LaCT, DeltaNet) | **the main branch**: it is what the mushroom body is; the 2025-26 frontier is open; the fly adds sparse addressing, three-factor modulated rates, compartments, consolidation |
| reservoir computing | random expansion + trained readout | a *component*: the fly's expansion is a reservoir used as an address space, not a backbone; reservoirs as backbones are not competitive for language |
| contrastive / local learning for the backbone | forward-forward, predictive coding, EP | reject for slow weights (backprop = evolution; the measured gap is large); local rules are used where they are natural, in the fast weights |
| spiking networks | event-driven neurons | reject for the model (many fly neurons are graded; the quality gap is large; the energy claim needs hardware); note as a future substrate |
| hyperdimensional / Hopfield | sparse codes with binding/bundling; attention as Hopfield retrieval | a *component*: capacity theory for sparse codes and binding for structure inside the memory |
| consolidation ("sleep") | replay from fast memory into slow weights | the most biological and least explored piece; makes the memory branch a *continual-learning* claim, which is an open frontier |
| genomic bottleneck | compressed weights, hypernetworks | park: intriguing, no evidence, not needed for the claim |

## 5. Candidate thesis (to be tested against the independent design panel)

A language model whose **lifetime learning system** is built on the fly's memory principles
(sparse hashed addresses, local three-factor plasticity with inferred learning rates, compartments
across timescales, sleep-like consolidation into slow weights) will beat the best test-time-learning
models on long-horizon retention and continual adaptation at matched compute, while matching them
on the standard suites. The killer experiment is the capacity / re-binding regime at 20M
parameters on one 16 GB GPU against a Gated-DeltaNet or Titans twin, with a 2 seed-SD threshold
and a six-seed replication.

Honest prior: perhaps one chance in four that a fly-derived component beats a well-tuned learned
control in its own regime; essentially zero that anything here is state of the art on perplexity
per FLOP at scale. The panel's job is to find a design with better odds or to show that this one
is the best available.
