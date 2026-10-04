# FlyAI design, v0.2 — decisions after the literature sweep and the design panel

*Synthesis of docs/aif/AXIOMS.md, docs/aif/MODEL_SPEC.md (v0/v0.1), docs/aif/LITERATURE.md
(the literature sweep) and docs/aif/DESIGN_PANEL.md (three independent revision proposals scored by
two judges). Written before any AIF experiment ran; the experiments' PREREG files reference
this document.*

## 1. What the panel established (verified against the code)

1. **The residual block's free energy is exactly quadratic in the fast latent** (Gaussian
   level-0 term with a linear map, quadratic prior terms), so K gradient steps with
   double-backprop approximate a posterior that has an elementwise closed form. Iteration
   has content only where the likelihood is non-quadratic: the categorical token head of
   the CPU model. The block therefore uses the closed form; the K-loop stays in the
   categorical model for AIF-H1 and as a pre-registered ablation arm.
2. **Arousal as a global multiplier of F was nearly inert**: the preconditioned step
   divides by gamma * P + 1, so gamma cancels for P >> 1, and the adaptive-K test is
   scale-invariant. AIF-H3 as first written would have compared the model with itself.
3. **The write gate had the wrong sign**: beta = sigmoid(a log pi + b) with a > 0 *lowers*
   writes when the memory is contradicted. The hypothesis says the opposite.
4. **Dead state**: the level-2 precision tracker measured a target, not an error, and
   entered no free-energy term.
5. **GPU memory blow-up**: a full copy of the fast-weight state per token is kept for
   backprop (about 130 GB per layer at T = 2048, B = 32, 2048 KCs).
6. **Phase-1 nulls are settled**: the whole-graph mask and the measured PN->KC matrix carry
   nothing beyond their degree statistics; the connectome enters FlyAI as *numbers and
   motifs*, never as wiring. Reworded in LIMITATIONS.md.

## 2. Adopted changes (v0.2)

| # | change | where | control / ablation |
|---|---|---|---|
| 1 | closed-form precision-weighted fusion: s* = sum_l w_l mu_l, w_l = softmax_l(log pi_l + c_l) per group (G = n_heads groups); amort and dense g0 deleted (parameter-matches the GDN twin) | flyai/block.py | learned GRU-style gate at equal parameters; K = 2 loop arm |
| 2 | arousal re-routed: gamma_t (from running surprise, bounded (0.25, 4)) scales the sensory precision pi_0 and the write gate beta; no longer multiplies F | model.py, block.py, memory.py | gamma = 1; pi_0-only; beta-only; learned input-driven scalar |
| 3 | write gate = variance-normalised surprise sigmoid(a log(pi_slow \|\|eps\|\|^2 / n) + b), a > 0 | memory.py | raw surprise (Titans-style); learned; const; no memory |
| 4 | categorical head: pi_0 = learned temperature (no Gamma tracking of one-hot errors); NLL curvature bound in the Newton denominator; level-2 tracker deleted | model.py | -- |
| 5 | precisions as bounded ratios (softmax over levels), grouped per head (G groups); slow rho = 0.98 for fusion, fast rho = 0.9 for the gate | model.py, block.py, primitives.py | scalar-per-level arm |
| 6 | hetero-associative delayed key: write (code(s_{t-1}), u(s_t)); read with code(s_{t-1}) | memory.py | -- |
| 7 | shared KC address across compartments with log-spaced decay priors tau_h in {4,16,64,256} | memory.py | separate per-head codes |
| 8 | level-3 precision from ring amplitude with a 2-d von Mises drive (natural parameters) | attractor.py, model.py, block.py | Gamma-tracker arm |
| 9 | novelty MBON (Bloom-filter familiarity of the address) as an available surprise / epistemic signal | primitives.py, memory.py | -- |
| 10 | optimiser hygiene: no weight decay on log-parametrised quantities, gains, biases, norms; Rajan-Abbott column re-balance after every optimiser step; Lyapunov update every 8 tokens in the categorical model and once per 64-token chunk in the residual block (outside the checkpointed region); subcritical target -0.02 | primitives.py, experiments | -- |
| 11 | per-chunk activation checkpointing in the residual block (chunk 64) | block.py | -- |
| 12 | connectome content in the AIF critical path = numbers only: fan-in by the sqrt rule (7 at 61-64 PN channels), coding level 5 % (sweep 15 % as an arm), 8-column canonical rings, 30.8 % inhibitory | configs | -- |
| 13 | statistical protocol: any positive 3-seed verdict is replicated at the winning LR with 6 fresh seeds (hypothesis + strongest control) before it is reported as supported | RESULTS builder, PREREGs | -- |

## 3. Rejected or deferred (with the reason)

* Keeping the K-step loop in the GPU block: EBT-class training cost for a linear solve.
* Replacing the Dale core by a diagonal linear (LRU/Mamba) core as the default: abandons AX6
  before AIF-H4 reports. Kept as the pre-registered fallback arm for the GPU pilot.
* Memory keys from the block input instead of the latent as default: turns the store into
  an input-addressed GDN clone; kept only for the chunk-parallel arm.
* Fan-shaped-body vector ALU, compartment-local divisive normalisation inside the core,
  anatomical k-WTA fractions per KC subtype, MBON->DAN loop matrices: no language evidence,
  or contradicted by the dead-ends list, or predicted null by their own priors. CPU-only
  arms at most, after the amplitude-calibration gate passes.
* Token-level EFE with rollouts: unaffordable; EFE stays at the action level (ask / answer).
* LaCT-style chunk-delayed memory writes and the chunk-parallel block variant: adopted as
  GPU-pilot arms with an equivalence test (C = 1 reproduces the sequential module), not as
  defaults; per-chunk checkpointing solves the memory problem exactly in the meantime.
* Throughput expectations of 40-60 % of GDN: unsupported; planned at 15-30 %, measured.

## 4. What the experiments now test (CPU, then 50M pilot, then 125M)

AIF-H1 iterative inference on the categorical head (K vs K0 vs K0-wide, with the pi_0 fix);
AIF-H2 the write-gate ladder under re-binding; AIF-H3 the arousal ladder on topic switches;
AIF-H4 Lyapunov regulation vs fixed gains on long-range retention; AIF-H5 EFE vs greedy vs
entropy-threshold on ask-to-learn; AIF-H6 the Bayesian wind tunnel (KL to the exact HMM
posterior predictive, plus the precision-tracking correlation). The 50M pilot ladder is in
docs/aif/GPU_RUNBOOK.md; an ingredient goes to 125M only if its removal costs > 2 seed-SD.
