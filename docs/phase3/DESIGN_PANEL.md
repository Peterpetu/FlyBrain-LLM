# Phase-3 design panel: three independent designs, two judges, one red team

*Three designers (biology-first, mathematics-first, state-of-the-art-first) received the sixteen briefs and the record of phases 1-2 and proposed one complete design each; two judges scored them; a red team attacked the leader. Verbatim outputs, lightly formatted. The synthesis is docs/phase3/DESIGN.md.*

## Design [biology-first]: ATTEST — Attested Fast Weights (phase-3 model: FlyATTEST-1). A drop-in second, non-negative "how much have I been taught here" state on a modern gated-delta memory, derived from the mushroom body's novelty compartment.

### Axioms

- A1 (architecture). The mushroom body is not a network but a BANK OF PARALLEL READOUTS OF ONE SHARED SPARSE CODE: ~2,000 Kenyon cells, ~5% active, read by ~21 MBON types across 15 compartments, each with its own dopaminergic teaching neuron. The compartments are the heads; the KC code is shared. (Aso et al. 2014, eLife 3:e04577; Li et al. 2020, eLife 9:e62576.) MaleCNS v1.0 as extracted in this repo: 2,019/2,045 KCs, 48/49 MBONs, 158 PAM + 8 PPL1 DANs, 1 APL per hemisphere.
- A2 (the axiom this design bets on). NOT ALL READOUTS OF THAT CODE ARE VALUE READOUTS. At least one compartment, MBON-alpha'3, reports NOVELTY/FAMILIARITY — 'how much has been taught at this address' — rather than valence, it habituates over minutes, and its output is used as a control signal. (Hattori et al. 2017, Cell 169:956-969.) A memory that reads only its own content is not the fly's architecture; a memory that reads its content AND its own accumulated teaching history is.
- A3 (why the fly can compute it). Dopamine-driven plasticity at KC->MBON only DEPRESSES and the weights are non-negative, so a compartment's weight vector is a MONOTONE ACCUMULATOR of the teaching it has received, not a signed quantity that can return to zero. Accumulated teaching is therefore physically readable, and 'never taught' is physically distinguishable from 'taught contradictorily'. (Hige et al. 2015, Neuron 88:985-998; Lipshutz et al. 2023, PLOS Comput Biol 19:e1010864 — normative derivation as online LDA; citation as recorded in this repo's digest, verify.)
- A4 (the write). Plasticity is three-factor and PRESYNAPTIC x NEUROMODULATOR: the postsynaptic MBON's own firing is essentially irrelevant to the weight change. The write gate is a function of (key activity) x (teaching signal), never of the read value. (Hige et al. 2015.)
- A5 (the teaching signal is manufactured inside the layer). MBON->DAN and MBON->MBON feedback close a local loop, so the teaching signal for each compartment is produced by the other compartments' READOUTS, not transported from outside. MaleCNS measured here: 339/300 MBON->DAN edges carrying 2,861/2,605 contacts. (Eschbach et al. 2020, Nat Neurosci 23:544-555; Bennett et al. 2021, Nat Commun.) Engineering corollary: the write gate is a small LEARNED critic reading the memory's own outputs — not a hand-specified formula.
- A6 (heterogeneity is architectural). Compartments differ deliberately in decay time, sign, capacity and flexibility, from minutes to days; the heterogeneity is built, not learned. (Aso & Rubin 2016, eLife 5:e16135.)
- A7 (forgetting is driven). Forgetting is an active, state-dependent, dopamine-gated process (Rac1, DAMB), up with locomotion and down with rest/sleep — not passive decay. (Shuai et al. 2010, Cell 140:579-589; Berry et al. 2012, Neuron 74:530-542; Berry et al. 2015, Cell 161:1656-1667.)
- A8 (durability needs spacing and an offline phase). LTM requires ~10 SPACED sessions with ~15-min rests; the same trials massed give only ARM. Post-learning sleep reactivates the specific dopaminergic neurons engaged at acquisition. (Tully et al. 1994, Cell 79:35-47; Dag et al. 2019, eLife 8:e42786.)
- A9 (sparsity is enforced by graded, LOCAL inhibition, and the normaliser is information). One graded non-spiking GABAergic APL neuron contacts ~100% of KCs (measured here: 1.000 L / 0.9995 R) with a ~50 um space constant, i.e. several local pools, not one global winner-take-all. (Lin et al. 2014, Nat Neurosci 17:559-568; Amin et al. 2020, eLife 9:e56954.) Corollary from efficient coding: whenever you divide by a running statistic you must also transmit that statistic. (Fairhall et al. 2001, Nature 412:787-792.)
- A10 (neuromodulation acts on plasticity and timescale, never on gain). In a normalised network a global multiplicative gain is EXACTLY cancelled (RMSNorm(g x) = RMSNorm(x)); the same scalar on a decay gate moves the memory horizon 16x. Dopamine is a per-compartment plasticity controller; octopaminergic arousal (a pure ~2x sensory gain) is the least transferable channel. (Measured in this repo; Maimon et al. 2010, Nat Neurosci 13:393-399.)
- A11 (the fly supplies no sequence model). Working memory is ONE target for ~4 s; serial order comes from a suppression hierarchy and ~3 sticky internal states; the only chaining is depth-2 second-order conditioning. There is no order memory, no recursion, no syntax. (Neuser et al. 2008, Nature 453:1244-1247; Seeds et al. 2014, eLife 3:e02951; Calhoun et al. 2019, Nat Neurosci 22:2040.) Constraint: the sequence backbone must come from elsewhere and be trained by backpropagation, which plays the role of evolution.
- A12 (the connectome is a distribution, not a blueprint). Within ONE animal, left and right copies of the same cell type share only 52% of type->type edges (68% for strong ones); PN->KC convergence is apparently random. Adjacency transfers nothing; a handful of scalars and one architectural motif transfer. (Caron et al. 2013, Nature 497:113-117; measured bilateral Jaccard 0.523-0.680 in this repo.) Constraint: phase 3 reads ZERO adjacency matrices. This also immunises the design against the repo's known data defects (all monoaminergic edges stored with weight 0; DPM mis-predicted as dopaminergic).

### Thesis

THE ONE PRINCIPLE: the mushroom body reads its own teaching history, not just its own content. Because KC->MBON weights are non-negative and dopamine only depresses (A3), each compartment's weight vector is a MONOTONE ACCUMULATOR of the teaching delivered at each address, and the circuit devotes a whole compartment (MBON-alpha'3) to reading that accumulation as novelty/familiarity and using it as a control signal (A2). The bet is that this second, non-negative, per-address state — "how much have I been taught here, and how much did it conflict?" — is the missing input to every modern test-time-learning memory, and that it is what should gate plasticity, forgetting, compute and consolidation.

WHY IT COULD MATTER FOR LANGUAGE. Every 2024-2026 fast-weight layer — DeltaNet (Schlag et al., ICML 2021, arXiv:2102.11174), Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025, arXiv:2412.06464), Titans (arXiv:2501.00663), ATLAS (arXiv:2505.23735), MesaNet, KDA/Kimi Linear, LaCT, PGDN — keeps ONE signed matrix S and computes its write gate from the token and the instantaneous error e_t = v_t - S^T k_t. That gate is blind in a specific, provable way.

PROPOSITION (separation). Let a layer's state be a single matrix updated as S_t = A_t (*) S_{t-1} + k_t g_t^T, with g_t any function of the token and of S_{t-1}^T k_t. Take history H_A: address a never written. History H_B: address a written with +v, then with -v at the same gate. Both leave S^T a = 0, so every read, every gate value and every statistic derived from the state at a is identical forever after. Any evidence channel that accumulates |written residual| at a returns 0 under H_A and ~2 beta |v| under H_B. QED. (Caveat stated honestly: the separation is at that address; a model with more state could encode it elsewhere — the claim is that no shipped architecture does, and that this is the cheapest sufficient statistic.)

MEASURED FOR THIS NOTE (numpy, sparse 32-of-2048 address, 96 keys, 96 distractor writes, gated delta store). Three address classes — never-written, written-once-consistently, written-then-contradicted. By |error|, which is ALL a Gated DeltaNet or Titans gate can see: novel 12.98, clean 5.82, contradicted 6.19 — clean and contradicted are indistinguishable, and novel-vs-contradicted AUROC is 0.000, i.e. error orders them exactly BACKWARDS relative to how hard you should write. By a one-scalar-per-address evidence channel with the crosstalk floor removed: novel 6.48, clean 43.51, contradicted 101.78 — monotone, with novel-vs-contradicted AUROC 1.000 and clean-vs-contradicted AUROC 0.955. No monotone function of the instantaneous error can produce that ordering.

This is why the write gate should not be tuned harder: it should be given an input it currently cannot compute. The three downstream uses are exactly the fly's: precision-weighted writing (do not overwrite a heavily attested address on one noisy observation; do write hard at a novel one), active forgetting (clear high-conflict addresses faster — A7), and recall-gated, spacing-gated consolidation (A8; the slow write needs a confidence, and this is it).

### Architecture

====================================================================
0. SCOPE AND SHAPE OF THE BET
====================================================================
ATTEST is a MINIMAL, NESTED, PROOF-BACKED modification of the current best sequence-memory layer, not a new backbone. The primary variant (ATTEST-D) is Gated DeltaNet plus one non-negative accumulator per head and three extra scalars into a learned gate: +3% state, +2% memory FLOPs, <0.1% parameters. The sparse-address variant (ATTEST-S) is a separate, later hypothesis. The backbone is a conventional hybrid decoder trained by backpropagation (A11/A1). No connectome adjacency is read anywhere (A12).

====================================================================
1. BACKBONE (fly content: none, by design)
====================================================================
Decoder-only, pre-norm, d_model = d, L layers, SwiGLU MLP, tied embeddings, RoPE.
Layer schedule 3:1 — three ATTEST layers to one sliding-window attention layer (window 512),
matching Qwen3-Next (full_attention_interval 4) and Kimi Linear. No pure-recurrent model is SOTA
anywhere, so the claim is "a better 75% layer", never "attention replaced".
Rung-2 config: d=512, 8 layers (6 ATTEST + 2 SWA), ~22M non-embedding.
Rung-3 config: d=768, 12 layers (9 ATTEST + 3 SWA), H=4 heads, d_k=d_v=128, ~110M total.

====================================================================
2. THE ATTEST LAYER — STATE
====================================================================
Per head h (H heads per layer), two states, both per-sequence and both reset at sequence start:

  CONTENT   S_h  in R^{d_k x d_v}           signed; the ordinary gated-delta store
  EVIDENCE  Z_h  in R_{>=0}^{d_k x d_z}     NON-NEGATIVE, monotone; d_z << d_v (default 4; d_z=1 is the cheap ablation)
  scalar    sigma_h in R_{>=0}              running L1 mass of Z_h, kept incrementally, for the crosstalk floor

State cost: d_k d_v + d_k d_z + 1 = 16,384 + 512 + 1 values per head at d_k=d_v=128, d_z=4,
i.e. +3.1% over Gated DeltaNet. This is the entire state-side price of the thesis.

====================================================================
3. PER-TOKEN COMPUTATION
====================================================================
Inputs x_t in R^d. Per head, with short causal conv (kernel 4) on q,k,v as in fla:

(3.1) Projections, L2-normalised keys/queries (as in Gated DeltaNet / PGDN):
        q_t = norm(W_q x_t),  k_t = norm(W_k x_t),  v_t = W_v x_t

(3.2) CONTENT READ and error (the delta rule's own quantities):
        y_t   = S_{h,t-1}^T q_t                                  (the layer's read)
        vhat_t= S_{h,t-1}^T k_t
        e_t   = v_t - vhat_t                                     (instantaneous error)

(3.3) EVIDENCE READ, with the APL crosstalk floor removed (A9). This is the new quantity:
        z_t     = Z_{h,t-1}^T k_t          in R_{>=0}^{d_z}
        floor_t = rho * sigma_{h,t-1}      rho = E[<k,k'>] for the key distribution;
                                           for unit-norm d_k-dim keys rho = 1/sqrt(d_k);
                                           for a kappa-of-n sparse code rho = kappa/n.
                                           (rho is LEARNED as a scalar, initialised at that value.)
        a_t     = relu( ||z_t||_1 - floor_t )                    ATTESTED MASS at this address
        nov_t   = exp( -a_t / mu_t ),  mu_t a running mean of a   NOVELTY   (Hattori 2017)
        cfl_t   = a_t / ( ||vhat_t||_1 + a_t + eps )              CONFLICT: taught mass that is
                                                                  not present in the content read.
                                                                  Novel => a~0 => cfl~0.
                                                                  Clean  => content ~ taught => cfl small.
                                                                  Contradicted => a large, content
                                                                  small => cfl -> 1.
   (nov_t, cfl_t) is the two-dimensional signal Proposition-separation says a single-matrix store
   cannot compute. Floor subtraction is NOT optional: without it the signal is swamped by
   crosstalk mass (measured for this note: separation collapses to ~1.3x; with it, 15.7x).

(3.4) THE DAN NET — a small LEARNED critic (A5). Crucially the CONTRIBUTION IS THE INPUT SET,
      NOT THE FORM. (A hand-specified evidence gate was tested for this note and collapsed to
      chance: beta = sigmoid(...-0.9 log(1+z)) drove recall to 0.044 because the gate latched
      shut. Hand-specified gates are the phase-1 H1b failure and are forbidden by construction.)

        u_t = [ log1p(a_t), nov_t, cfl_t, log1p(||e_t||^2 / m_t), log1p(sigma_h), phi_t, s_t ]
        (m_t = running mean of ||e||^2; phi_t = log of the layer's input-norm EMA, the Fairhall
         rule A9: transmit the normaliser; s_t = slow document-clock state, section 3.8)

        h_t         = GELU(W_1 u_t + b_1)            W_1 in R^{16 x 7}   (~200 params/head)
        beta_t      = softplus(w_beta^T h_t + b_beta)          WRITE gate      in (0, 2)
        gamma_t     = softplus(w_gam^T h_t  + b_gam)           ERASE gate      in (0, 2), decoupled
                                                               (GDN-2-style separation; A3's two
                                                               receptors, one transmitter)
        lambda_t    = softplus(w_lam^T h_t  + b_lam)           FORGET drive (A7)
        alpha_{h,t} = exp( -exp(a_h) * lambda_t )              per-head decay; exp(a_h) log-spaced
                                                               over tau_h in {2^3,2^5,2^7,2^9,2^11,2^13}
                                                               tokens across heads (A6), FIXED
                                                               ladder, learned offset only.
      beta_t is allowed in (0,2) so eigenvalues may be negative — required for state tracking
      (Merrill et al., ICML 2024, arXiv:2404.08819; Grazzi et al., ICLR 2025 — arXiv id uncertain).

(3.5) CONTENT WRITE — unchanged from Gated DeltaNet except the gates are decoupled and now
      depend on (nov, cfl). This is the ONLY place the thesis touches the content dynamics:

        S_{h,t} = alpha_{h,t} * ( S_{h,t-1} ( I - gamma_t k_t k_t^T ) ) + beta_t k_t e_t^T

      With gamma_t = beta_t this is exactly Gated DeltaNet. With u_t's first three (evidence)
      components masked to zero, the whole layer is exactly Gated DeltaNet with a learned gate.
      THAT IS THE PRIMARY CONTROL: the same code, three input features zeroed.

(3.6) EVIDENCE WRITE — three-factor, presynaptic x neuromodulator, monotone, never erasing (A3, A4).
      The postsynaptic read does not appear:

        Z_{h,t}    = alpha^Z_h * Z_{h,t-1} + beta_t * k_t ( P |e_t| )^T ,   P in R^{d_z x d_v} learned, row-stochastic
        sigma_{h,t}= alpha^Z_h * sigma_{h,t-1} + beta_t * || P |e_t| ||_1
      alpha^Z_h is a SEPARATE, slower decay than alpha_{h,t} (tau^Z ~ 8 tau_h), fixed ladder,
      learned offset. Non-negativity is enforced structurally (|e_t| >= 0, P >= 0 via softplus,
      alpha^Z > 0) — no clamping, so gradients are clean.
      Note sigma is maintained in O(1), so the floor costs nothing.

(3.7) LAYER OUTPUT — the evidence is a FEATURE, not only a gate (A9, Fairhall):
        o_t = W_o [ y_1 ; ... ; y_H ; log1p(a_1..a_H) ; nov_{1..H} ; cfl_{1..H} ; phi_t ]
        x_t^{out} = x_t + o_t
      (Cost: W_o gains 3H columns. The "evidence as feature" and "evidence as gate" paths are
      separately ablatable and separately pre-registered.)

(3.8) SLOW STATE s_t (document clock, A6/A10): s_t = (1-1/T_s) s_{t-1} + (1/T_s) g(mean_h nov, mean_h cfl, ||e||),
      T_s = 128 tokens, s_t in R^4. It conditions the DAN net's affine parameters (FiLM on the
      GATE, never on activations — A10). Ablatable.

====================================================================
4. OPTIONAL MODULES (separate hypotheses, each independently ablatable, NOT on the critical path)
====================================================================
(4a) ADAPTIVE COMPUTE (A2, Hattori 2017). If mean_h nov_t > theta_1 or mean_h cfl_t > theta_2,
     re-enter the block once (weight-tied). theta learned with a straight-through threshold;
     budget enforced by a Lagrangian on the expected re-entry rate. Pre-registered separately
     because learned halting gates have a poor record; fixed-prior depth supervision is the fallback.

(4b) ATTEST-S, SPARSE ADDRESSING (A9, A12-compatible). Replace the dense key by a learned
     compression -> divisive normalisation -> block-sparse expansion -> adaptive-threshold k-WTA:
        adaptive gain:  g_t = rho_a g_{t-1} + (1-rho_a)|x_t| ; xt~ = x_t/(eps+g_t); phi_t = log mean g_t
        compression:    u_t = W_pn xt~ in R^{d_pn}, d_pn = 128
        div. norm:      uh_{t,i} = sign(u)|u|^nu / (s^nu + ((1/d_pn) sum_j c_j |u_j|)^nu), nu learned, init 1.5
                        (the fly's exponent n ~ 1.5, Olsen, Bhandawat & Wilson 2010, Neuron 66:287,
                         is a PREDICTION to test, not a constant to copy)
        expansion:      B blocks of b rows (n = B b = 4096); router picks top-m=4 blocks;
                        within them z = E uh with learned fan-in K per row
        APL threshold:  per-BLOCK (local, not global — Amin et al. 2020) running-quantile threshold
                        theta updated by Robbins-Monro: theta <- theta + eta(f - 1[z>theta]);
                        k_t = normalise(relu(z - theta)); hard top-kappa at inference.
        K and the coding level f are DERIVED by maximising the participation ratio
        (tr C)^2/tr(C^2) of the code on real token data, not copied from K=6, f=5%
        (Litwin-Kumar et al. 2017, Neuron 93:1153-1164). Expect K ~ 15-30 at d_pn=128.
     Why separate: this is Product-Key Memory / Memory-Layer territory (Lample et al., NeurIPS 2019,
     arXiv:1907.05242; Berges et al., arXiv:2412.09764), it is NOT what the thesis rests on, and
     it carries the only serious wall-clock risk. The thesis works on dense keys.
     What it buys, measured in this repo (docs/phase3/MEMORY_CAPACITY.md): at matched active rows
     per token, recall at 256 stored pairs is 0.82 (sparse 100-of-2048) vs 0.17 (dense d_key=100);
     at 512 pairs 0.70 vs 0.07. Lazy decay (a cumulative log-decay accumulator plus per-row
     last-touch timestamps) keeps per-token cost at O(kappa d_v), exactly.

(4c) SPACING- AND RECALL-GATED CONSOLIDATION (A8). A slow pair (Sbar_h, Zbar_h), tau ~ 2^14.
     Eligibility at time t for the current address: ELIG = 1[cfl_t < tau_c] * 1[Delta_t in [D_min, D_max]],
     where Delta_t = t - (last write at this address, from the row timestamps). I.e. the slow
     store is written only when the FAST store already reads correctly (recall-gating; Lindsey &
     Litwin-Kumar, eLife 2024;13:e90793 — predicts learnable timescale LINEAR rather than
     logarithmic in repetitions) AND the recurrence was SPACED (Tully et al. 1994).
     Offline pass: sleep pressure P_t = sum_t beta_t ||e_t||; when P_t > P*, replay the top-J
     addresses BY STORED GATE VALUE (selective replay, Dag et al. 2019), with lambda_t := 0
     (forgetting suppressed during sleep — A7, Berry et al. 2015), then reset P.
     Read: y_t = S^T q + zeta (Sbar)^T q, zeta learned.
     Honest flag: this repo's own naive-replay toy was negative (retention 0.06 vs 0.02 beyond
     the fast horizon, docs/phase3/CONSOLIDATION_TOY.md). It is included because the two missing
     ingredients it identified — SELECTION and a CONFIDENCE GATE — are exactly what the evidence
     channel supplies. It stays off the critical path and gets its own pre-registration.

====================================================================
5. WHAT IS LEARNED HOW
====================================================================
BY GRADIENT (backpropagation = evolution, A1; gradients flow through the in-sequence writes
exactly as in DeltaNet/GDN):
  W_q, W_k, W_v, W_o, short convs; the DAN net (W_1, w_beta, w_gam, w_lam, biases);
  the projection P; the decay offsets a_h and alpha^Z offsets; the floor coefficient rho;
  zeta, theta, the slow-state map g; and in ATTEST-S: W_pn, c, nu, s, the router, E, and
  the coding-level target f. All backbone parameters.

BY LOCAL RULE AT TEST TIME (no gradient, O(d_k d_v) per token, runs identically at inference):
  S_h (gated delta, eq. 3.5); Z_h and sigma_h (monotone three-factor, eq. 3.6);
  the running statistics m_t, mu_t, g_t, s_t; in ATTEST-S the APL thresholds theta (Robbins-Monro);
  in 4c the slow stores Sbar/Zbar and the sleep-pressure integrator.

FIXED BY DESIGN (architectural heterogeneity, A6 — not learned, and that is the point):
  the log-spaced decay ladder tau_h in {8,32,128,512,2048,8192} across heads;
  tau^Z_h = 8 tau_h (evidence outlives content — the fly's novelty channel habituates over
  minutes while compartment memories decay in minutes-to-days);
  the non-negativity and monotonicity of Z; the 3:1 layer schedule; one shared key per head's
  two states.

====================================================================
6. COST, EXACTLY
====================================================================
Per head per token, d_k=d_v=128, d_z=4:
  Gated DeltaNet: read d_k d_v = 16,384 MACs; write ~2 d_k d_v = 32,768. Total 49,152.
  ATTEST adds:    evidence read d_k d_z = 512; evidence write 512; DAN net ~200. Total +1,224 = +2.5%.
  State:          +d_k d_z + 1 = +513 values = +3.1%.
  Parameters:     +W_1,P,w_* ~ 0.8k/head; at H=4, L=9 that is ~29k on a 110M model = +0.03%.
This ratio — a proof-backed new signal for +3% state and +2.5% FLOPs — is the single most
important property of the design, because it makes the matched-compute and matched-state
comparisons nearly free and removes every excuse for an unmatched control (the phase-2 failure,
where the "wide" control had 1.94x the parameters and 1.32x the FLOPs).

====================================================================
7. CHUNKED FORM (trainability)
====================================================================
The content recurrence is unchanged, so the existing chunkwise UT-transform kernel applies
verbatim (Yang et al., arXiv:2406.06484; fla chunk_gated_delta_rule, chunk 64). The evidence
recurrence Z_t = alpha^Z Z_{t-1} + beta_t k_t (P|e_t|)^T is PURELY LINEAR — no (I - tril(.))^{-1},
no erase term — so it chunks exactly like gated linear attention: cheaper than the content path
it rides on. The only new sequential coupling is that beta_t depends on z_t = Z_{t-1}^T k_t, which
is resolved the same way GDN resolves its own gate: compute gates from the chunk-start state with
an intra-chunk correction, or (at chunk 32-64) scan intra-chunk. Reference implementation:
subclass fla's GatedDeltaNet, add one linear recurrence and one 7->16->4 MLP. That is the whole
engineering delta for ATTEST-D, and it is why one engineer can build it.

====================================================================
8. TWO MECHANISMS THIS DESIGN DELIBERATELY DOES NOT CONTAIN
====================================================================
Both were candidate theses, both were falsified numerically in ~20 minutes before this design was
written, and both are reported so nobody re-runs them:
 (i) OPPONENT CHANNELS BY COORDINATE SIGN (M+ = relu(r), M- = relu(-r), different decays), the
     naive reading of Felsenberg et al. 2018 (Cell 175:709-722). PROVED to collapse exactly to the
     gated delta rule when the decays are tied (max|(M+ - M-) - M_gdn| = 6.7e-16, verified), and
     measured to LOSE badly when untied (reversion accuracy 0.38 vs 1.00) because asymmetric decay
     on a coordinate-wise +/- split biases every stored value positive. The fly's opposing
     compartments hold different MEMORIES with fixed opposite readout signs, not the positive and
     negative coordinates of one value vector; the conflation is fatal.
 (ii) CONFIRMATION-ROUTED TWO-TRACE MEMORY (slow trace on agreement, labile trace on contradiction).
     Measured to lose to the tied single-trace control on every metric (rebind 0.56 vs 0.98,
     reversion 0.76 vs 1.00, untouched 0.37 vs 0.52), replicating this repo's own smoke-test
     finding of an intrinsic trade-off. Hand-specified ROUTING loses for the same reason
     hand-specified GATING lost in phase-1 H1b.
The surviving mechanism is the one that adds an input rather than a rule.

### What is new

1. THE EVIDENCE STATE ITSELF. A second, non-negative, monotone, per-address accumulator of delivered teaching, read at the same address as the content and at ~2.5% of its cost. To my knowledge no sequence model has one. The adjacent work is adjacent in a way that can be stated precisely and is not the same object: PGDN (2026) accumulates a per-CHANNEL second moment of the key as an Adagrad-style preconditioner (a property of the key distribution, not of what was taught at an address); MesaNet keeps a key COVARIANCE and solves a ridge regression exactly (again key statistics, not teaching history); GDN-2 (2026) decouples per-channel erase and write gates but still holds one signed matrix; Titans and ATLAS gate on the instantaneous gradient norm, which the measurement below shows is the wrong signal; LaCT renormalises fast weights homeostatically. None can distinguish "never taught here" from "taught contradictorily here".

2. A PROOF OF SEPARATION, NOT AN INTUITION. Any layer whose state is one matrix updated by S_t = A_t (*) S_{t-1} + k_t g_t^T with g_t a function of the token and of S_{t-1}^T k_t admits two histories with identical state, identical reads and identical gates at an address forever after, which the evidence channel separates. This covers DeltaNet, Gated DeltaNet, KDA, GDN-2, Titans, ATLAS, MesaNet and every learned-gate variant of them. This matters methodologically more than technically: phases 1 and 2 both proposed fixed RULES, which lie inside a learned gate's hypothesis class, so a learned gate contained them as a special case and beat them (H1b: -2.6 seed-SD). ATTEST proposes a new INPUT that is provably outside that class. The generalisable lesson — DO NOT PROPOSE A GATE, PROPOSE A STATE — is itself a contribution of this design.

3. MEASURED, NOT ASSERTED (for this note, sparse 32-of-2048 address, 96 keys, 96 interfering writes). Three address classes and their signals: by |error| (all a GDN/Titans gate sees) novel 12.98 / clean 5.82 / contradicted 6.19; by crosstalk-corrected evidence novel 6.48 / clean 43.51 / contradicted 101.78. Novel-vs-contradicted AUROC: 0.000 by error (ordered exactly backwards), 1.000 by evidence. Clean-vs-contradicted AUROC 0.955 by evidence, ~0.5 by error. The gap is not marginal.

4. THE CROSSTALK FLOOR AS AN ARCHITECTURAL REQUIREMENT. A monotone accumulator's read is dominated by interference mass, so the raw signal is nearly useless (measured 1.3x separation). Subtracting rho*sigma, with sigma the accumulator's own L1 mass kept in O(1), restores it (15.7x). This is the APL/antennal-lobe normalisation (A9) doing real work, plus the Fairhall rule (transmit the normaliser) — and it is a cheap, specific, non-obvious implementation requirement that would sink a naive attempt.

5. A NESTED, EXACT CONTROL. The baseline is the same code with three input features masked to zero and gamma_t tied to beta_t — provably Gated DeltaNet, with identical parameter count, identical kernels, identical tuning budget. There is no reimplementation asymmetry to argue about, which is precisely the criticism that weakened phase 2 (its "wide" control had 1.94x the parameters).

6. THE FIRST ARCHITECTURE THAT CAN ACTUALLY IMPLEMENT RECALL-GATED CONSOLIDATION IN AN LM. Lindsey & Litwin-Kumar (eLife 2024;13:e90793) prove that gating the slow write on the FAST store reading correctly changes the learnable timescale from logarithmic to LINEAR in repetitions. That gate needs a confidence signal. No LM has one; ATTEST's cfl_t is one. Combined with the spacing gate (Tully et al. 1994) this is a concrete, biologically-derived, unimplemented test-time-training schedule.

7. TWO SELF-FALSIFIED MECHANISMS REPORTED AS RESULTS (see architecture section 8), including the exact-equivalence proof that coordinate-sign opponent channels with tied decays ARE the gated delta rule to machine precision. That is a genuine contribution to anyone reading Felsenberg 2018 and reaching for the obvious implementation.

NOT CLAIMED AS NEW: sparse hashed memory (Product-Key Memory, arXiv:1907.05242; Memory Layers, arXiv:2412.09764); data-dependent rotation (RoPE, CoPE, Selective RoPE); multi-timescale memory (Nested Learning/HOPE); the delta rule (Schlag et al. 2021); homeostatic fast-weight renormalisation (LaCT); negative eigenvalues for state tracking; divisive normalisation (it is RMSNorm's family).

### Regime of the claimed advantage

THE DEFINED REGIME: in-context associative memory over streams that contain CONTRADICTION AND REPETITION — where the same address is written more than once, sometimes inconsistently, among distractors — at 20M-125M parameters, at MATCHED per-token FLOPs AND MATCHED recurrent-state bytes, reported as a Pareto frontier over the number of stored associations. This is the regime in which the separation proposition bites: if every address is written at most once, novel and contradicted never both occur and the evidence channel is provably worthless.

PRIMARY BENCHMARK (new, pre-registered and released with the code):
  MQAR-N — MQAR (Arora et al., "Zoology", ICLR 2024, arXiv:2312.04927) extended with (i) observation
  noise on values, (ii) distractor writes on unseen keys, (iii) silent re-binding of a fraction of
  keys mid-sequence, and (iv) queries STRATIFIED by address class (never-written / consistently
  written / contradicted). Run at P in {64, 128, 256, 512} stored pairs — never at 8, which is five
  times below the dense control's own capacity ceiling and is why phase-1 H1 could not have
  detected anything.

SECONDARY BENCHMARKS (all standard, all run by us on our own twins, never quoted across papers):
  RULER S-NIAH-1/2/3 at 1k-32k with training at 2k (Hsieh et al., arXiv:2404.06654) — MULTI-needle
    especially, which collapses 4-8x earlier in context than single-needle for every constant-state
    model and is the closest public proxy for interference.
  SWDE / FDA / SQuAD-completion recall suite (Arora et al., "Based", ICML 2024, arXiv:2402.18668).
  BABILong qa1-qa5 at 4k-128k (Kuratov et al., NeurIPS 2024 D&B, arXiv:2406.10149) — optional.
  lm-eval zero-shot (ARC-e, ARC-c, HellaSwag, PIQA, WinoGrande, LAMBADA) as a NO-REGRESSION CHECK
    ONLY. MMLU is excluded: at this scale it is at chance.

STRONGEST COMPETITORS, named separately for the layer and for the gate:
  FOR THE LAYER: Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025, arXiv:2412.06464) — the
    production answer, deployed at 3:1 in Qwen3-Next and (as KDA) Kimi Linear. Instantiated as
    OUR OWN nested control (evidence features masked, gamma tied to beta), plus an independent
    fla-kernel GDN at its standard config as an external-validity check.
  FOR THE GATE: Titans (Behrouz et al., arXiv:2501.00663), whose entire contribution is a
    surprise-gated write with learned momentum and decay; and PGDN (2026), whose accumulated
    second-moment preconditioner is the 2026 answer to "normalise the surprise signal". These are
    the papers ATTEST must beat to claim the gate contribution, and the separation proposition
    applies to both.
  BOUNDS: a sliding-window-attention-only twin and a full-attention twin, as floor and ceiling.
  ALSO REQUIRED IF ATTEST-S IS PURSUED: a product-key Memory-Layer twin (arXiv:2412.09764) at
    matched memory parameters, since sparse hashed memory is that literature's, not the fly's.

THE CLAIM, STATED SO IT CAN BE CHECKED: "At matched per-token FLOPs and matched recurrent-state
bytes, at 125M parameters on 6-8B tokens, a gated-delta memory augmented with a per-address
evidence channel strictly dominates its own nested Gated-DeltaNet control on stratified recall
under contradiction (MQAR-N) and on multi-needle S-NIAH, at >=2 of 3 state sizes, by >2 seed-SD
with bootstrap CIs excluding zero over >=3 seeds, WITHOUT regressing zero-shot perplexity or
lm-eval beyond 1 seed-SD."

EXPLICITLY NOT CLAIMED: perplexity per FLOP against a well-tuned transformer or Gated DeltaNet on
generic text (no biological architecture has ever won this, and at ~400M the spread across seven
architectures is 0.7 accuracy points while the spread on 32K retrieval is 0-91 points — perplexity
cannot support an architecture claim at this scale); energy or joules per token (no accepted
benchmark, no neuromorphic hardware); exact long-range copying (provably impossible for fixed-state
models, Jelassi et al., ICML 2024, arXiv:2402.01032); and absolute leaderboard SOTA against
released small models trained on 2T-36T tokens, which is a data claim and unreachable here.

### Killer experiment

EXPERIMENT K0 — "CAN A LEARNED GATE WITH THE EVIDENCE CHANNEL BEAT THE SAME LEARNED GATE WITHOUT IT?"
Module level, 4 CPU cores, ~6 CPU-hours total. No language model. This is the cheapest possible
test of the central thesis and it is designed so that a null is as informative as a win.

TASK: a noisy, non-stationary, contradiction-rich associative stream, batched over 64 independent
streams. P key slots (P in {64, 128, 256}), value dimension 16, T = 4,096 events. Each event is a
write (key, value + sigma*noise) or a query. Three generators run concurrently: (i) first-time
writes on fresh keys, (ii) consistent repeats, (iii) silent re-bindings at rate 0.004/key/step,
plus distractor writes on keys that are never queried. Queries are STRATIFIED into
never-written / consistently-written / contradicted, and the primary metric is computed on the
pooled natural mix so that no arm can be favoured by stratum weighting.

ARMS — all five are THE SAME MODULE with THE SAME learned gate MLP, differing ONLY in which inputs
the gate can see and in one tied scalar. Identical parameter counts to within the gate's input
layer; identical state except where noted; identical LR sweep {3e-3, 1e-2, 3e-2}; 5 seeds.
  A1  ATTEST      gate sees [log1p(a), nov, cfl, log1p(||e||^2/m), log1p(sigma), phi, s]
  A2  MASKED      THE PRIMARY CONTROL. identical code, evidence features zeroed, gamma tied to
                  beta => provably Gated DeltaNet with a learned gate. Same params (dead weights),
                  same kernels, same sweep.
  A3  TITANS-LIKE gate sees only the gradient-norm surprise ||e||^2 with learned momentum and decay
  A4  BEST FIXED  beta swept over {0.2, 0.3, 0.5, 0.7, 1.0}, best per setting (a strong, tuning-
                  advantaged control)
  A5  STATE-MATCHED MASKED   A2 with d_v raised so that total state bytes equal A1's
                  (this is the control that removes "you just added state")

PRIMARY METRIC: recall accuracy on the pooled query mix at the end of the stream.
SECONDARY (pre-registered, reported either way): (a) accuracy restricted to contradicted addresses;
(b) AUROC of the evidence read for the three-way address classification; (c) calibration — AUROC of
each arm's best available confidence scalar against read correctness.

PRE-REGISTERED THRESHOLDS:
  SUPPORTED iff  A1 - A2 > 2 x (larger seed-SD) with a 20,000-sample paired-bootstrap 95% CI
                 excluding 0, at >= 2 of the 3 P settings AND at >= 2 of the 3 noise levels
                 sigma in {0.3, 0.6, 1.0};
            AND  A1 - A5 > 0 with CI excluding 0 at >= 2 of 3 P settings (the effect is not just
                 extra state);
            AND  A1 does not lose to A2 on a CONTRADICTION-FREE version of the same stream by more
                 than 1 seed-SD (no regression where the mechanism should be inert);
            AND  the evidence read achieves three-way AUROC > 0.8.
  FALSIFIED iff  the A1 - A2 CI includes 0 at >= 2 of 3 P settings, OR A1 loses to A5.
  CORRECTNESS GATE (not a hypothesis — a verification requirement, run first): with the evidence
                 features masked, gamma tied to beta and decay disabled, the module must reproduce
                 a reference DeltaNet's plain-MQAR accuracy to within 0.5 seed-SD. If it does not,
                 the implementation is wrong and no result is reported. (Phase 2 was verified
                 line-by-line against its spec; this keeps that standard.)

WHY THIS IS THE RIGHT FALSIFIER, AND WHAT IT ALREADY SURVIVED. The separation proposition
guarantees the SIGNAL exists and cannot be computed by A2; a measurement for this note put it at AUROC 1.000
(novel vs contradicted) and 0.955 (clean vs contradicted) against 0.000 and ~0.5 for |error|. What
K0 tests is the only thing still open: whether a gate that can SEE that signal converts it into
recall. That is exactly where phases 1 and 2 died, so it is where the money should be spent first.
A related caution is already banked: a HAND-SPECIFIED evidence gate was measured for this note and
collapsed to chance (recall 0.044 vs 0.82 for the best fixed rate) because the gate latched shut —
so every arm in K0 uses a learned gate, and any future hand-specified functional form is barred.

COST: 5 arms x 3 P x 3 sigma x 5 seeds x 3 LRs = 675 runs of ~30 s = ~6 CPU-hours on 4 cores;
~3 days of engineering on top of the existing flyprinc module harness. If K0 is falsified, the
thesis is dead for ~6 CPU-hours and 3 days, and the negative result — "a provably-available signal
that a learned gate cannot exploit" — is itself worth reporting.

### Why phases 1-2 failed and how this differs

WHAT WENT WRONG, DIAGNOSED AS THREE DISTINCT ERRORS.

(1) PHASE 1 TREATED A CONNECTOME AS A PRIOR. The whole-graph mask was indistinguishable from a degree-preserving shuffle and from Erdos-Renyi at N = 256/512/1024 (-0.6/-0.2/-0.4 seed-SD); real PN->KC wiring equalled shuffled wiring (-0.1 seed-SD); the extracted 148-neuron ring tracked state far worse than an unstructured layer (0.24 vs 0.67, -9.1 seed-SD). These are the PREDICTED results, not failures: within one animal, left and right copies of the same cell type share only 52-68% of type-level edges, and PN->KC convergence is apparently random (Caron et al. 2013). A connectome is a sample from a distribution, and it encodes the statistics of odours and optic flow, which text does not share.
HOW ATTEST DIFFERS: it reads ZERO adjacency. It takes one architectural MOTIF (many readouts of one shared code, at least one of which reports teaching history rather than value) and three derived scalars, and it derives fan-in and coding level from an objective rather than copying K=6 and f=5%. There is no wiring experiment anywhere in the plan.

(2) PHASES 1 AND 2 PROPOSED FIXED RULES, WHICH LIE INSIDE THE LEARNED HYPOTHESIS CLASS. Every ingredient — the error-magnitude write gate, precision-weighted fusion, Lyapunov gain regulation, K-step free-energy descent — was a fixed functional form inserted into a backprop-trained model. A learned gate with the same inputs CONTAINS the fixed rule as a special case, so backpropagation finds it and the biology is at best redundant. H1b measured exactly that: the learned gate beat the error gate by 2.6 seed-SD. AIF-H1 measured the same thing for iterative inference: K=3 beat K=0 by 0.026 bits/char (0.7 seed-SD, below the 2-SD bar) and lost to a wider control, because the amortisation gap closes with training.
HOW ATTEST DIFFERS — THIS IS THE CENTRAL METHODOLOGICAL CHANGE: it does not propose a gate, it proposes a STATE. The gate stays fully learned; what changes is that it is given an input the control provably cannot compute. The separation proposition makes this airtight rather than hopeful: for any single-matrix store with any learned gate, two histories exist that produce identical state, identical reads and identical gate values at an address forever, and the evidence channel separates them. A learned gate cannot learn information that is not in its inputs. The generalisable rule extracted from the two nulls is: DO NOT PROPOSE A RULE THAT A LEARNED GATE CONTAINS; PROPOSE A STATE THAT A LEARNED GATE CANNOT RECONSTRUCT. I applied that rule to this design ruthlessly — and it killed two of my own candidate mechanisms before this document was written (see architecture section 8), including one I could prove reduces to Gated DeltaNet to machine precision.

(3) PHASES 1 AND 2 MEASURED IN REGIMES WHERE NOTHING WAS DETECTABLE, WITH UNMATCHED CONTROLS. H1 tested associative memory at 8 key-value pairs against a store whose own capacity ceiling is ~65 — five times below the point where any capacity difference can appear. AIF-H1 measured bits/char on TinyShakespeare at 135k parameters on 4 CPU cores, where the seed spread is 0.005-0.045 bits/char and the literature's own effects only emerge above ~360M parameters; and its "wide" control had 1.94x the parameters and 1.32x the FLOPs, so the comparison was unfair against the hypothesis and uninterpretable either way.
HOW ATTEST DIFFERS: (a) the primary metric is recall under contradiction, chosen because the separation proposition says the mechanism is NECESSARY there by construction and INERT where addresses are written once — so both a win and a null are informative; (b) P is swept across and beyond the control's capacity ceiling, never at 8; (c) the primary control is the SAME CODE with three input features masked and one scalar tied, so parameter count, kernels, tuning budget and implementation quality are identical by construction — there is no "wide control" argument available to anyone; (d) a state-matched arm is mandatory, so "you just added state" is pre-answered; (e) a correctness gate (the masked model must reproduce DeltaNet to 0.5 seed-SD) runs before any hypothesis, preserving the phase-2 discipline of formal verification while removing the phase-2 problem that a verified model can still be a verified implementation of a bad idea.

(4) ON ACTIVE INFERENCE SPECIFICALLY. Phase 2 imported a vocabulary, not an algorithm. The one piece with algorithmic bite — the learning rate as an inferred precision — survives into ATTEST, but in a weakened and honest form: this repo's own measurement (docs/phase3/ADAPTIVE_RATE.md) shows the precision-gated rate matches a change-point Kalman oracle without knowing the regime, yet beats a TUNED fixed rate by only a few points, and this session reproduced that (precision gate 0.822 vs best fixed 0.825 at sigma=0.3, with better MSE). So precision enters as one of seven gate inputs, never as the thesis. Expected free energy, arousal as a global gain, Lyapunov gain regulation and per-token iterative inference are all dropped: a global multiplicative gain upstream of a norm is EXACTLY cancelled (RMSNorm(gx) = RMSNorm(x)), which is a theorem and not a tuning failure, and it retro-explains why phase 2's arousal was inert.

(5) WHAT IS CARRIED FORWARD UNCHANGED. Pre-registration with hash locks before any run; matched controls; >=3 (now >=5) seeds; the 2 seed-SD plus bootstrap-CI rule; mechanical verdicts computed from summary.json; formal verification against a written spec; and the prohibition on calling any model "a fly brain", claiming SOTA without matched controls, or treating anatomical contact counts as synaptic strengths.

### Cost

ENGINEERING: one engineer, ~7 weeks to the end of rung 3, on top of the existing flyprinc module harness and prereg/bootstrap infrastructure (which already exist and are reused).
  rung 0 (module + K0 prereg + correctness gate)                 3 days
  rung 1 (ablation and sweep grid, results builder blocks)       4 days
  rung 1.5 (fla subclass, Triton/torch kernel, throughput bench) 3 days
  rung 2 (training harness, MQAR-N in-model, S-NIAH, eval)       5 days
  rung 3 (scale-up, frontier harness, 6-seed replication)        8 days
  write-up, LIMITATIONS, release of MQAR-N                       5 days
The rung-0/1 work is pure PyTorch gather/scatter on existing code; the rung-1.5 work is the only
part needing a custom kernel, and ATTEST-D can ship without one (it is one extra linear recurrence
plus a 7->16->4 MLP on top of fla's GatedDeltaNet).

COMPUTE:
  CPU (4 cores, available now): ~6 hours (rung 0) + ~30 hours (rung 1) = ~36 CPU-hours.
  GPU (1x16GB, available): rung 1.5 ~8 h; rung 2 ~80 h (22M params x 2B tokens = 2.6e17 FLOPs
    per run, ~3 h/run at ~25 TFLOP/s effective bf16, x 5 arms x 3 seeds x ~2 configs);
    rung 3 at ~110M x 7B tokens = 4.6e18 FLOPs per run ~ 50 h/run, so a 3-arm x 3-seed x 1-state-size
    grid = 9 runs ~ 450 GPU-hours ~ 19 days continuous, and the full 2-3 state-size frontier is
    900-1,350 GPU-hours ~ 5-8 weeks continuous on one card.
  Total on owned hardware: ~36 CPU-hours and ~550-1,450 GPU-hours, i.e. 4-9 weeks of wall clock
    with the card running continuously.
  RENTED ALTERNATIVE for rung 3 only: ~450-900 A100-hours at ~$1-2/h = ~$500-1,800, or roughly
    3-6 days on a single 8xH100 node at ~$25/h = ~$1,800-3,600. Recommended if the frontier is to
    include more than one state size, because the one-card path makes rung 3 the schedule bottleneck.
  Out of scope and stated as such: rung 4 (350M-1B on 50-100B tokens) needs ~5,000-20,000
    A100-hours (~$8k-40k) and is not funded or planned.

DECISION POINTS THAT CAP THE SPEND: rungs 0, 1 and 1.5 together cost ~36 CPU-hours, ~8 GPU-hours and
~10 engineer-days, and any one of them can kill the programme before a single language-model token
is trained. That asymmetry — the falsifier costs ~1.5% of the total budget — is the main reason to
prefer this design over a larger one.

### Ladder

1. RUNG 0 — CPU (4 cores), ~6 CPU-hours, ~3 days engineering. Experiment K0 exactly as pre-registered above: the evidence channel as an input to a learned gate, against the same module with those inputs masked (provably Gated DeltaNet), a Titans-like surprise gate, a tuning-advantaged best-fixed gate, and a state-matched masked control. GO/NO-GO: A1-A2 > 2 seed-SD with CI excluding 0 at >=2 of 3 P settings AND >=2 of 3 noise levels; A1 > A5 (not just extra state); no regression on the contradiction-free stream beyond 1 seed-SD; three-way evidence AUROC > 0.8; and the DeltaNet correctness gate passes first. NO-GO => the thesis is dead, write it up as a negative result with the separation proof and the measured AUROCs, and do not proceed. This rung must be paid for before any GPU time.
2. RUNG 1 — CPU (4 cores), ~30 CPU-hours, ~4 days. Decompose the win. (a) Which downstream use of the evidence signal carries it: write gate only / decay gate only / output feature only / all three, as a 2x2x2 ablation. (b) The d_z sweep {1, 2, 4, 8, 16} and the evidence-decay ladder sweep tau^Z/tau in {1, 2, 8, 32} — the sharpest tuning risk, since too-fast evidence is just recency and too-slow makes everything look attested. (c) The crosstalk-floor ablation (learned rho vs fixed vs none), which this session's numbers say is the difference between a 15.7x and a 1.3x signal. (d) The capacity curve P in {32,64,128,256,512} at matched state and matched FLOPs, to check whether the numpy result (0.82 vs 0.17 at P=256) survives learned keys and learned gates. GO/NO-GO: at least two of the three downstream uses survive independently at >2 seed-SD, and at least one hyperparameter setting is robust across all three P values (a win that exists only at one d_z is a tuning artefact, not a mechanism). NO-GO => report the module-level result as the finding and stop; do not escalate a fragile effect to the GPU.
3. RUNG 1.5 — GPU (1x16GB), 1 day, no training. THROUGHPUT AND MEMORY GATE, deliberately before any language-model run. Benchmark the ATTEST-D layer against fla's chunk_gated_delta_rule at the rung-2 and rung-3 shapes (B=16, T=2048, H=4, d_k=d_v=128, d_z=4) for forward, forward+backward and single-token decode. GO/NO-GO: >=0.9x GDN tokens/s in training and >=0.95x at decode (ATTEST-D adds one linear recurrence and a 7->16->4 MLP, so anything worse than this indicates an implementation problem, not a design problem). Separately: if ATTEST-S is to be pursued at all, its block-sparse gather/scatter must reach >=0.5x GDN tokens/s at the same shapes; if it does not, ATTEST-S is dropped and only ATTEST-D proceeds. The thesis does not depend on sparse addressing, which is the deliberate de-risking that distinguishes this plan from phase 2's.
4. RUNG 2 — GPU (1x16GB), ~80 GPU-hours (~5 days continuous), ~1 week engineering. First language model: ~22M non-embedding (d=512, 8 layers, 6 ATTEST-D + 2 SWA at 3:1), 2B FineWeb-Edu tokens, context 2k, 3 seeds. Arms: ATTEST-D, the nested masked control, the state-matched masked control, an independent fla GDN, and an SWA-only floor. Evaluations: MQAR-N in-model, RULER S-NIAH-1/2/3 at 1k-8k (trained at 2k, so 4x extrapolation), SWDE/FDA/SQuAD, lm-eval zero-shot. GO/NO-GO: no perplexity or lm-eval regression beyond 1 seed-SD, AND a win on MQAR-N or S-NIAH-2/3 beyond 2 seed-SD with CI excluding 0. NO-GO => the honest report is 'the mechanism works where contradiction is frequent; natural text at 2k context does not contain enough of it', which is a real and publishable boundary result and the single most likely outcome of the whole programme.
5. RUNG 3 — GPU (1x16GB for ~3 weeks, or ~450 rented A100-hours), ~2 weeks engineering including the evaluation harness. THE CLAIM. ~110M parameters (d=768, 12 layers, 9 ATTEST-D + 3 SWA, H=4), 6-8B FineWeb-Edu tokens, 3 seeds, 2-3 matched state sizes, full pre-registered Pareto frontier of recall against recurrent-state bytes. Competitors trained by us on identical data in identical order: nested masked control, state-matched masked control, fla GDN, Titans-MAG, and (only if rung 1 supported ATTEST-S) a product-key Memory-Layer twin. Full benchmark set plus BABILong qa1-qa5. Learning rates fixed per arm by a 300M-token pilot, never swept inside the grid. GO/NO-GO for publishing a SOTA claim: strict dominance over the nested control at >=2 of 3 state sizes by >2 seed-SD with bootstrap CIs excluding zero, no lm-eval regression beyond 1 seed-SD, and a 6-seed replication of the single best cell. Anything less is reported as a frontier plot without a SOTA claim.
6. RUNG 4 — OUT OF SCOPE, NAMED FOR HONESTY. 350M-1B parameters on 50-100B tokens, which is where reviewers will say the architecture question actually lives (Mixture-of-Recursions pays only above 360M; the recall-vs-state literature runs at 400M-1.4B). This needs ~5,000-20,000 A100-hours and is not reachable by this project. State that limit in the paper rather than implying the claim scales.
7. OPTIONAL RUNG C — only after rung 2 passes, never before. The consolidation hypothesis (architecture section 4c): recall-gated plus spacing-gated writes into a slow store with an offline replay pass, against uniform replay, recency replay, no consolidation, and the Ibrahim et al. (COLM 2024, arXiv:2403.08763) incumbent of LR re-warm plus ~5% replay, all at matched total capacity and matched total FLOPs. GO/NO-GO: retention of items older than the fast horizon beats the equal-state no-consolidation control beyond 2 seed-SD. This repo's own naive-replay toy was negative (0.06 vs 0.02), so the prior here is poor and it stays off the critical path.

### Risks

- MODAL FAILURE — the learned gate may not need the signal. The evidence channel provably EXISTS and is provably unavailable to the control (AUROC 1.000 vs 0.000 measured for this note), but that does not mean backpropagation can convert it into recall. A learned gate on the instantaneous error is a strong control and has already beaten a fly-derived gate once in this repo (H1b, -2.6 seed-SD). This is exactly what rung 0 buys for 6 CPU-hours, and it is where I would put most of the probability mass of failure. Estimated P(rung 0 passes) ~ 0.55.
- TRANSFER FAILURE — contradiction may be rare in natural text. The regime is defined by addresses being written more than once, inconsistently. MQAR-N constructs that; FineWeb-Edu at 2k context may not contain enough of it for the mechanism to pay, in which case rung 2 returns a null even though rung 0 and 1 passed. Mitigation: pre-register the boundary as the finding, and measure the contradiction rate in the corpus so the null is interpretable rather than mysterious. Estimated P(rung 0 passes AND rung 2 passes) ~ 0.25.
- HYPERPARAMETER FRAGILITY — the evidence decay constant tau^Z is the sharpest knob in the design. Too fast and the evidence read is just a recency signal the model already has from its own decay; too slow and every address looks attested and the signal saturates. If the win exists only at one setting of tau^Z and d_z, it is a tuning artefact. Rung 1's robustness requirement (one setting must work across all three P values) is designed to catch this, and it may well kill the design.
- FLOOR ESTIMATION UNDER LEARNED KEYS. The crosstalk floor rho*sigma assumes near-isotropic addresses; language-model keys are strongly anisotropic and correlated. With a badly estimated floor the separation collapses (measured: 15.7x with the floor, 1.3x without). Learning rho helps but may not be enough. The known analogue is the reported hard failure of fixed sparse hashing on un-whitened LM states at ~100k keys. Mitigation: whitening/decorrelation before the memory, and a floor ablation that is pre-registered as a possible cause of a null.
- NOVELTY CONTESTED AT REVIEW. 'You added a second state to the memory' is a small delta, and 2026 work is adjacent: PGDN's accumulated per-channel second moment, MesaNet's key covariance, GDN-2's decoupled erase/write gates, Titans' surprise. The separation proposition is the defence and it must be stated against each of them individually — none accumulates teaching history at an ADDRESS. If a 2026 paper published after May 2026 already does this, the contribution collapses; the literature must be re-checked before any run, because this session's web-search budget was exhausted and several citations below are flagged uncertain.
- TWO OF THIS DESIGN'S OWN CANDIDATE MECHANISMS WERE ALREADY FALSIFIED IN 20 MINUTES (coordinate-sign opponent channels; confirmation-routed two-trace memory). That is good discipline, but it is also the base rate: fly-derived mechanisms in this project have a poor record, and the surviving one may fall the same way at rung 0. Anyone reading this design should weight that base rate, not the enthusiasm of the write-up.
- EXTRA STATE IS A REAL COST AND THE MATCHED-STATE CONTROL MAY WIN. Z costs d_k*d_z per head (+3.1% at d_z=4). Arm A5 gives that budget to the baseline as wider values. Recall capability is bounded by recurrent state size, so if the baseline converts those bytes into recall more efficiently than the evidence channel converts them into gate quality, the design loses on the axis it chose. This control is mandatory and is expected to be the tightest.
- SCALE — every result here is below the scale at which the field settles architecture questions. At ~400M the spread across seven architectures on zero-shot commonsense is 0.7 points; recall benchmarks discriminate, which is why they are the primary metric, but a reviewer may still demand 1B+, which this project cannot reach (rung 4). The claim must be stated with its scale attached and must never be extrapolated.
- CONSOLIDATION IS LIKELY TO FAIL AGAIN. This repo's own toy measured 0.06 vs 0.02 retention beyond the fast horizon. The evidence channel supplies the two missing ingredients it identified (selection and a confidence gate), but 'add sleep' has already been shown not to be free. It is deliberately off the critical path; if it is put back on, expect a null.
- ADAPTIVE COMPUTE (4a) HAS A POOR TRACK RECORD. Learned halting gates underperform fixed-prior depth supervision, CALM-style confidence exit lost to a static baseline on every summarisation dataset tested, and this repo's own adaptive-K rule never fired (K_mean = 3.000 in every run). Keep it as an ablation with a Lagrangian budget, never as a headline.
- ENGINEERING — the gate depends on z_t = Z_{t-1}^T k_t, which is an intra-chunk dependency. Resolving it from the chunk-start state introduces an approximation whose error grows with chunk size; resolving it exactly costs an intra-chunk scan. Rung 1.5 must measure both the throughput and the approximation error, and the approximation must be an explicit, reported design choice rather than a silent one.
- WHAT THIS DESIGN CANNOT FAIL AT, BECAUSE IT DOES NOT ATTEMPT IT: nothing here reads the connectome adjacency, so the repo's known data defects (every monoaminergic edge stored with weight 0; DPM mis-predicted as dopaminergic; contact counts treated as strengths) cannot contaminate a result. That is deliberate, and it is the main structural lesson of phase 1 being taken seriously.
- HONEST PROBABILITIES. P(rung 0 passes) ~ 0.55. P(survives to a >2 seed-SD recall win in a trained LM at rung 2) ~ 0.25. P(a defensible pre-registered SOTA claim in the named regime at rung 3 against a self-trained Gated-DeltaNet twin) ~ 0.12-0.15. P(any perplexity-per-FLOP win at scale) < 0.02. P(the programme yields a publishable result EITHER WAY, given a proof-backed separation, a nested exact control and a pre-registered boundary) ~ 0.8 — which is the property phases 1 and 2 lacked, because their nulls were predicted by outside theory and so carried little information.

## Design [mathematics-first]: SIEVE — a gated-cascade, sparse-address test-time memory (model: SIEVE-H, a 3:1 hybrid decoder)

### Axioms

- A1 (established). The mushroom body's compartments share ONE address code. ~2,000 Kenyon cells per hemisphere form a single sparse population (this repo's MaleCNS extraction: 2,019 L / 2,045 R), read by ~21-34 MBON types across ~15 compartments (Aso et al., eLife 3:e04577, 2014; Li et al., eLife 9:e62576, 2020). Compartments differ in time constant, sign, capacity and teaching signal — NOT in address. Phase 2 violated this by giving each compartment its own expansion.
- A2 (established). The coding level is actively REGULATED, not wired. One graded, non-spiking GABAergic neuron (APL, 1 per hemisphere, contacting ~100% of KCs; measured here 1.000 L / 0.9995 R) holds the active fraction near 5-10%; blocking it roughly doubles activity, re-correlates odours and destroys fine discrimination (Lin et al., Nat Neurosci 17:559-568, 2014; Amin et al., eLife 9:e56954, 2020). Sparsity is a servo loop with a set-point, not a constant.
- A3 (established). The write is three-factor and PRESYNAPTIC x NEUROMODULATOR. The postsynaptic MBON's own firing is essentially irrelevant to the weight change (Hige et al., Neuron 88:985-998, 2015). In vector form the compartment's update is the delta rule; with decay it is the gated delta rule. The fly and the 2025 state of the art write the same equation.
- A4 (established). Retention is ARCHITECTURAL and heterogeneous. Compartment decay constants are set by cell type and span minutes to days; the heterogeneity is fixed, not learned (Aso & Rubin, eLife 5:e16135, 2016). ~15 compartments is a designed spectrum of forgetting rates over one shared address.
- A5 (established). Durable storage requires temporal SPACING of repeated experience, not repetition count. 10 sessions with 15-min rests give 7-day protein-synthesis-dependent memory; the same 10 sessions massed give none (Tully, Preat, Boynton & Del Vecchio, Cell 79:35-47, 1994). The same law holds in humans with optimal gap ~10-20% of the retention interval (Cepeda et al., Psychol Bull 132:354-380, 2006). Consolidation is gated on the TIME STRUCTURE of recurrence.
- A6 (promising). Promotion to a durable store is gated by the fast store's own SUCCESS, not by error. Recall-gated consolidation — write to the slow synapse when the fast synapse already reads out correctly — is the normative account of the fly/cerebellar two-stage system and makes the learnable timescale grow LINEARLY rather than logarithmically with repetitions (Lindsey & Litwin-Kumar, eLife 13:e90793, 2024). This is the opposite sign to Titans' surprise gate.
- A7 (established). Forgetting is an ACTIVE, state-dependent, dopaminergically driven process — set by tonic PPL1 MP1/MV1 activity, raised by locomotion, lowered by rest and sleep (Shuai et al., Cell 140:579-589, 2010; Berry et al., Neuron 74:530-542, 2012; Berry et al., Cell 161:1656-1667, 2015) — and it is suppressed during the offline consolidation window (Donlea et al., Science 332:1571-1576, 2011; Dag et al., eLife 8:e42786, 2019).
- A8 (established, mathematical + biological). Neuromodulation acts on PLASTICITY and TIMESCALE, never as a gain upstream of a normaliser. RMSNorm(g*x) = RMSNorm(x) exactly (verified in this repo to 3e-7 for g in [0.25,4]), and a global scalar on the loss cancels under any preconditioned optimiser. Biologically, ~94% of dopaminergic output inside the mushroom body lands presynaptically on KC axons (7,212 DAN->KC vs 486 DAN->MBON edges, this repo's extraction), i.e. on the write, not on the activation.
- A9 (established). The specific wiring is a SAMPLE from a developmental distribution; only its statistics are reproducible. Within one animal, left and right copies of the same cell type share only ~52% of type->type edges (68% for the strongest), and PN->KC convergence is random to first order (Caron, Ruta, Abbott & Axel, Nature 497:113-117, 2013; Eichler et al., Nature 548:175-182, 2017). Connectome numbers enter this design ONLY as hyperparameter priors that are then swept, never as matrices.
- A10 (established). The circuit's learned output bandwidth is tens of scalars per event (48-49 MBONs per side, measured here). It is a memory and control module, never a language backbone; and routing an LM through anatomical ports cost -0.63 nat/token in this repo, 6/6 seeds. Backpropagation plays the role of evolution and trains everything on the slow timescale; the local rule plays the role of lifetime learning and runs only at test time.

### Thesis

THE ONE PRINCIPLE: the fly does not have a memory with a forgetting rate. It has a CASCADE OF GATED FILTERS OVER ONE SHARED SPARSE ADDRESS, in which a write is admitted to a slower stage only if it passes a test performed by the faster stage (does the fast stage already read this association back correctly?) and by the address's own history (did this address recur after a gap?). Everything else — the sparse Kenyon-cell code, the delta-rule write, the ~15 compartments, the graded APL normaliser — is machinery in service of that one operation: SELECTIVE PROMOTION UNDER A SHARED ADDRESS.

WHY THIS IS THE RIGHT MATHEMATICAL OBJECT. Every test-time memory in the 2024-2026 literature (DeltaNet, Gated DeltaNet, KDA, GDN-2, Titans, ATLAS, TTT, LaCT, MesaNet) writes EVERYTHING to EVERYTHING and modulates only the RATE. That collapses three quantities into one number, the state dimension d_k:
  (i) CAPACITY — how many associations can be held without interference;
  (ii) COMPUTE — how many rows are touched per token;
  (iii) RETENTION — how long a given association survives, and which associations survive longest.
The fly separates all three. Capacity is the address size n (~2,000 KCs). Compute is the coding level k = f*n (~5-10%, ~100 cells). Retention is the compartment ladder H with fixed per-compartment decays, and WHICH item lands on which rung is decided by a test, not by the item's key geometry.

THE STATEMENTS THAT ARE PROVABLE BEFORE ANYTHING IS MEASURED.
(P-A, capacity/compute, a theorem.) A linear fast-weight state S in R^{d_k x d_v} written by the delta rule satisfies rank(S) <= d_k, so it can represent at most d_k linearly independent key->value associations exactly; read+write cost is Theta(d_k*d_v). A k-of-n sparse-address state M in R^{n x d_v} has rank up to n while a read/write gathers only k rows, cost Theta(k*d_v). Exact capacity per unit of per-token compute is therefore n/k rather than 1. This repo has already measured the consequence at matched compute in pure numpy (docs/phase3/MEMORY_CAPACITY.md): at P = 256 stored pairs, dense delta d_key=100 gives 0.17 recall, sparse delta 100-of-2048 gives 0.82; at P = 512, 0.07 vs 0.70.
(P-B, retention spectrum.) Reading compartment h at delay D after its last write gives alpha_h^D * v. A learned read mixture over H compartments gives R(D) = sum_h w_h alpha_h^D. Because t^{-gamma} = (1/Gamma(gamma)) * integral_0^inf lambda^{gamma-1} e^{-lambda t} dlambda, a geometric quadrature lambda_h = c^{-h} with w_h proportional to lambda_h^gamma approximates a power law on [1, c^H] with a log-periodic ripple of amplitude 2|Gamma(gamma + 2*pi*i/ln c)|/Gamma(gamma) — well under 1% at c = 4, gamma ~ 1 (arithmetic done here; verify the constant before publishing). H = 1 is exponential and deviates from any power law by Omega(1) beyond one e-fold.
(P-C, selective promotion — THE CLAIM.) Proposition P-B alone is NOT enough to separate this design from the field, and I want that on the record: per-channel decay (KDA, GDN-2, Mamba-2's per-channel Delta) already yields a mixture of exponentials, sum_i k_i^2 alpha_i^D, so it too can be log-tailed. What per-channel decay CANNOT do is decide, after the fact, WHICH associations occupy the slow modes: the mixture weights are the key's own components, fixed at write time, and every write that touches a slow channel is retained there, interference included. In the cascade, rung h+1 receives a gated subset of the write stream. If the stream is a mixture of recurring "signal" at rate p and one-off "chaff" at rate 1-p, and the promotion test passes signal with probability q_s and chaff with q_n << q_s, then the slow rung's interference load is P*(p*q_s + (1-p)*q_n) instead of P. Since delta-rule recall degrades in the load/capacity ratio, the slow rung's usable horizon scales with the SELECTIVITY q_s/q_n of the test, a quantity that has no counterpart in any current LM memory. Recall-gating makes q_s/q_n grow with the number of repetitions, which is why Lindsey & Litwin-Kumar (eLife 2024) get a learnable timescale linear rather than logarithmic in repetitions.

WHY IT COULD MATTER FOR LANGUAGE. Text is overwhelmingly chaff: most tokens are never referred to again, while a small set of entities, definitions, names and task instructions recur at log-distributed gaps. A constant-state memory that writes everything wastes its slow modes on the chaff. The fly solves exactly this problem — an insect's day is almost entirely one-off experience punctuated by a few associations worth keeping for a week — and it solves it with a test, not with more state. That is the one place where the fly's optimisation pressure and a language model's actual failure mode (long-horizon recall from a constant state) genuinely coincide.

### Architecture

=====================================================================
SIEVE: a gated-cascade, sparse-address test-time memory layer.
SIEVE-H: the language model — a pre-norm decoder with a 3:1 schedule
of SIEVE layers to sliding-window softmax-attention layers.
This is a memory layer derived from a reading of one insect circuit.
It is not a model of a brain and must never be described as one.
=====================================================================

--------------------------------------------------------------------
0. STATE INVENTORY (per SIEVE layer, per sequence)
--------------------------------------------------------------------
  M_h in R^{n x d_v},  h = 1..H      compartment value matrices (the memory)
  M_fam in R^{n x 1}                 familiarity/novelty compartment
  ell in R^n (int)                   per-address last-touch timestamp
  s in R^n (fp16)                    per-address log-decay bookkeeping (see sec. 7)
  a in R^p                           per-channel adaptation state (front end)
  theta in R                         APL threshold (scalar servo)
Everything in this list is written by LOCAL RULES at test time and is
never a gradient-trained parameter. Everything else is trained by
backpropagation. Nothing is initialised from a connectome matrix.

Reference sizes, 125M config: d = 768, p = 128, n = 16384 (m = 128),
k = 32, d_v = 64, H = 7, chunk C = 64.

--------------------------------------------------------------------
1. FRONT END (the normalisation stage). Learned by gradient.
--------------------------------------------------------------------
From the residual stream x_t in R^d:

  u_t   = W_pn x_t,                     W_pn in R^{p x d}          (learned compression)
  a_t   = rho*a_{t-1} + (1-rho)*|u_t|,  rho = exp(-1/tau_a), tau_a = 64   (Weber-Fechner)
  utl_t = u_t / (eps + a_t)                                        (per-channel adaptive gain)
  z_t   = utl_t / (sigma + ( (1/p) * sum_j |utl_{t,j}|^nu )^{1/nu} )      (divisive normalisation)
  g_t   = log(eps + a_t)   in R^p                                  (the normaliser's OWN state)

nu is a learned scalar per layer, initialised at 1.5 (the antennal-lobe
exponent, Olsen, Bhandawat & Wilson, Neuron 66:287-299, 2010); sigma learned.
g_t is projected to R^8 and fed forward into every gate. This is Fairhall's
rule (Fairhall et al., Nature 412:787-792, 2001): whenever you divide by a
running statistic you must also TRANSMIT that statistic, or you destroy the
information about which distribution you are in. Phase 2's arousal unit used
surprise as a GAIN, which axiom A8 shows is exactly inert; here the analogous
signal is a FEATURE and a gate input only.

--------------------------------------------------------------------
2. ADDRESS (expansion + APL servo). Sub-keys learned; code local.
--------------------------------------------------------------------
Product-key addressing (Lample et al., NeurIPS 2019, arXiv:1907.05242) gives
an exactly-searchable n = m^2 address at O(m*p) cost. Split z_t = [z1; z2]
with z1, z2 in R^{p/2}; learned sub-key tables C_1, C_2 in R^{m x p/2}:

  s1 = C_1 z1,   s2 = C_2 z2,   score(i,j) = s1_i + s2_j
  S_t = top-k indices of score   (exact, from top-k of each factor)

APL SERVO (one scalar unit; this is the fly's only global inhibitory neuron):

  theta_t = theta_{t-1} + eta_theta * ( fhat_t - f* ),   f* = k/n, eta_theta = 0.01
  fhat_t  = fraction of scores above theta_{t-1} (estimated from the top-k tail)
  kappa_{t,i} = relu(score_i - theta_t) for i in S_t, else 0
  kk_t        = kappa_t / (||kappa_t||_2 + eps)          [the k-sparse unit key]

theta carries a stop-gradient. Top-k fixes the gather shape (kernel-friendly);
theta fixes the MAGNITUDES, so the code's overlap statistics are invariant to
any monotone rescaling of the score distribution. Without theta, a shift in
||z|| changes the effective coding level and hence the expected code overlap
k^2/n; this is the mechanism whose absence produced the HARD_FAIL of fixed
fly-LSH at 100k adversarially similar LM keys recorded in this repo's sweep.

Optional train-time regulariser (the ONE legitimate role for contrastive
learning here — shaping the address, not the objective):
  L_sep = lambda_sep * E_{t != t'} [ ( <kk_t, kk_{t'}> - k/n )_+ ^2 ]

--------------------------------------------------------------------
3. THE CASCADE (the thesis). Compartments share kk_t.
--------------------------------------------------------------------
Values v_t = W_v x_t in R^{d_v}.  FIXED decay ladder (not learned):
  alpha_h = exp(-1/tau_h),  tau_h = tau_0 * c^h,  tau_0 = 4, c = 4, H = 7
  => tau in {4, 16, 64, 256, 1024, 4096, 16384}   (12 octaves; phase 2 used 6)

For h = 1..H, in order:

  r_{h,t}    = M_h(t-1)^T kk_t                                    [read, before write]

  WRITE RATE (three-factor, presynaptic x neuromodulator; A3):
  beta_{h,t} = 2*sigmoid( w_h^T [ v_t - r_{h,t} ; r_{h,t} ; W_g x_t ; g'_t ; nu_t ] + b_h )
               in (0, 2)     [range (0,2) admits negative eigenvalues, the free
                              fix for TC^0 expressivity: Grazzi et al., ICLR 2025]

  PROMOTION GATE (the new object):
  gamma_{1,t} = 1
  gamma_{h,t} = sigmoid( a_h * cos( r_{h-1,t}, v_t ) + b_h )          [RECALL gate, A6]
              * exp( - ( ln Delta_t - ln Delta*_h )^2 / (2 s_h^2) )   [SPACING gate, A5]
  Delta_t = t - mean_{i in S_t} ell_i ;  then ell_i <- t for i in S_t

  UPDATE (gated delta rule with active forgetting, A7):
  alphat_{h,t} = alpha_h ^ ( 1 + c_h * act_t ),  act_t = EMA of sum_h beta_{h,t}
  M_h(t) = alphat_{h,t} * M_h(t-1)
           + gamma_{h,t} * beta_{h,t} * kk_t * ( v_t - alphat_{h,t} * r_{h,t} )^T

Read-out (learned per-compartment, per-channel mixture, plus an output gate
in the style of Qiu et al., NeurIPS 2025, arXiv:2505.06708):
  Omega_t = sigmoid( W_om x_t ) in R^{H x d_v}
  o_t     = sum_h Omega_{h,t} (elementwise*) r_{h,t}
  y_t     = W_o ( o_t (elementwise*) sigmoid(W_gate x_t) )

The recall gate is the SIGN FLIP that makes this design falsifiable:
Titans/ATLAS write to memory when SURPRISE is high (the model was wrong);
SIEVE promotes to the slow rung when the fast rung was RIGHT. Both cannot
be correct; the killer experiment pits them against each other directly.

--------------------------------------------------------------------
4. FAMILIARITY COMPARTMENT (free; MBON-alpha'3 analogue)
--------------------------------------------------------------------
  M_fam <- alpha_f M_fam + beta_f kk_t ( 1 - M_fam^T kk_t )
  nu_t  = 1 - clip( M_fam(t-1)^T kk_t , 0, 1 )                 [novelty, O(k)]
nu_t is a gate input everywhere, and (ablation arm) an adaptive-compute
signal: skip the memory update when nu_t < threshold. Cost: n extra scalars.
Phase 2 exposed a novelty MBON and never used it as a control signal.

--------------------------------------------------------------------
5. OFFLINE CONSOLIDATION (second hypothesis, not in the critical path)
--------------------------------------------------------------------
At document boundaries, accumulate plasticity pressure P_t = sum ||Delta M||.
When P_t crosses a threshold: sample addresses with probability proportional
to the stored gamma*beta (the modulatory tag, per Dag et al. 2019 — replay
what the gate marked, NOT uniformly), replay them into rung H with forgetting
suppressed (c_h -> 0), reset P. The repo's own negative result
(docs/phase3/CONSOLIDATION_TOY.md: 0.06 vs 0.02 retention beyond the fast
horizon) is what tells us naive uniform replay is worthless; selection by
stored gate value, multi-pass, is the specific fix being tested.

--------------------------------------------------------------------
6. THE HYBRID LANGUAGE MODEL (SIEVE-H)
--------------------------------------------------------------------
Pre-norm decoder. Layer schedule 3:1 — three SIEVE layers, then one
sliding-window softmax-attention layer (window 512, 4 sink tokens), as in
Qwen3-Next (48 layers, full_attention_interval 4) and Kimi Linear (27 layers,
3:1 KDA:MLA). Every layer has a causal depthwise conv of kernel 4 on q/k/v.
SwiGLU FFN at 2.67d. RoPE in the attention layers only.
  20M calibration: 8 layers, d = 256, p = 64, n = 4096, k = 16, d_v = 32, H = 5.
  60M rung-2:      16 layers, d = 512, p = 96,  n = 8192, k = 32, d_v = 48, H = 6.
  125M rung-3:     24 layers, d = 768, p = 128, n = 16384, k = 32, d_v = 64, H = 7.
No pure-recurrent claim is made at any point. The claim is about a better
75%-layer, which is the only honest form of the claim in 2026.

--------------------------------------------------------------------
7. WHAT IS LEARNED HOW (the axiom A10 split)
--------------------------------------------------------------------
BY GRADIENT (backprop = evolution): W_pn, sigma, nu, C_1, C_2, W_v, all gate
  weights (w_h, b_h, a_h, s_h, Delta*_h, c_h), W_om, W_o, W_gate, the whole
  backbone. ~270k params per SIEVE layer vs ~6d^2 = 3.5M for a GDN layer at
  d = 768: SIEVE is parameter-CHEAP, so matched-parameter arms put the slack
  in the (identical) FFN and match FLOPs, params and state bytes as three
  separately reported quantities with three separate matched arms.
BY LOCAL RULE AT TEST TIME (no gradient, no backward pass): M_h, M_fam,
  theta, a, ell, act. These are the "lifetime" memory.
FIXED (never learned): the decay ladder {alpha_h}, the coding-level set-point
  f*, the compartment count H, the chunk size C. These are the connectome
  STATISTICS entering as hyperparameter priors, each swept (H in {1,3,5,7},
  f in {0.002 .. 0.05}, c in {2,4,8}) rather than copied.

--------------------------------------------------------------------
8. IMPLEMENTABILITY: the one trick that makes this a weekend kernel
--------------------------------------------------------------------
A naive per-token sparse gather/scatter gets terrible tensor-core utilisation
and would make "FLOP-matched" a lie. The fix:
  WITHIN A CHUNK OF C TOKENS, TAKE THE UNION U OF THE TOUCHED ADDRESSES,
  |U| <= C*k. GATHER M_h[U], RESCALE EACH ROW BY alpha_h^(t - ell_i), DENSIFY
  THE CHUNK'S KEYS INTO R^{C x |U|}, RUN THE STANDARD DENSE CHUNKED GATED
  DELTA-RULE KERNEL ON THE SUBMATRIX, SCATTER BACK.
With n = 16384, C = 64, k = 32: |U| <= 2048 = n/8. The inner kernel is then
exactly flash-linear-attention's chunk_gated_delta_rule, unmodified. Decay of
untouched rows is handled by per-row last-touch timestamps (ell, already kept
for the spacing gate) with a per-chunk renormalisation to avoid underflow of
alpha^Delta. Available on the development machine: torch 2.14.0+cu130,
triton 3.8.0. Install flash-linear-attention for the control arms.
Total engineering: one gather/scatter wrapper, one servo, one gate stack.
Estimated 8-12 weeks for one engineer including the evaluation harness.

--------------------------------------------------------------------
9. PER-TOKEN COST (125M config, one SIEVE layer)
--------------------------------------------------------------------
  address:   2*m*(p/2) = 2*128*64      = 16.4k MAC
  memory:    3*H*k*d_v = 3*7*32*64     = 43.0k MAC
  gates:     ~ H*(3*d_v + 16)          =  1.5k MAC
  projections: (d*p) + (d*d_v) + (d_v*d) + (d*d_v) = 0.098M+0.049M+0.049M+0.049M = 246k MAC
  TOTAL ~ 307k MAC/token/layer, vs ~1.57M for a Gated DeltaNet layer at d=768.
  State: n*d_v*H*2 bytes = 16384*64*7*2 = 14.7 MB/layer/sequence (fp16). This
  is LARGE and must be reported honestly; it is the price of capacity n.
  Matched-state arms therefore shrink n, and are pre-registered as expected-null
  (the Arora et al. state-size bound forbids a win there).

### What is new

Six things, ranked by how much of the claim they carry. Each is separately ablatable and separately pre-registered.

1. SELECTIVE PROMOTION BETWEEN MEMORY STAGES (carries the claim). A write is admitted to a slower stage only if the faster stage already reads the association back correctly. This is recall-gating (Lindsey & Litwin-Kumar, eLife 13:e90793, 2024) lifted into a language model's test-time memory. Its sign is OPPOSITE to every existing write gate: Titans (arXiv:2501.00663) and ATLAS (arXiv:2505.23735) write when surprise is HIGH; SIEVE promotes when the fast stage is RIGHT. Both cannot be correct, and the killer experiment is a head-to-head. To my knowledge (verify) no LM memory performs any test on a write before routing it to a longer-lived store.

2. SPACING-GATED CONSOLIDATION (second-most distinctive). Promotion is additionally gated on the TEMPORAL SEPARATION of repeated touches of the same address, via per-address last-touch timestamps and a log-Gaussian kernel centred at Delta*. This is Tully et al. (Cell 79:35-47, 1994) — 10 spaced sessions give 7-day memory, 10 massed sessions give none — and it independently matches the human spacing law (Cepeda et al., Psychol Bull 2006). Nobody in the test-time-learning literature gates consolidation on temporal separation; the whole field gates on magnitude. Cost: one int32 vector of length n.

3. THE THREE-WAY DECOUPLING ITSELF. Capacity = n, compute = k, retention = (H, {alpha_h}), as three independently settable numbers. Every current memory (DeltaNet, Gated DeltaNet, KDA, GDN-2, Titans, Mamba-2, TTT, LaCT, MesaNet) sets all three to one number. Memory Layers at Scale (arXiv:2412.09764) and PKM (arXiv:1907.05242) decouple capacity from compute but write by SGD at train time and have no retention axis; HOPE / Nested Learning has a multi-frequency continuum but a dense address and no promotion test. The combination appears unoccupied — flag as UNCERTAIN, requires a literature check on "multi-timescale memory layer" before publication.

4. THE APL SERVO AS A NECESSARY CONDITION, NOT A DECORATION. A scalar threshold controller holding the coding level at a set-point is what makes a fixed learned address survive distribution shift, because code overlap k^2/n is invariant under monotone rescaling of the score distribution only if the coding level is held. This converts a vague "biologically inspired inhibition" into a falsifiable prediction (the address fails without it) with a cheap ablation.

5. THE CHUNK-UNION DENSIFICATION. Restricting a chunk to the union of its touched addresses (|U| <= C*k = n/8 at the reference sizes) lets a sparse-address delta memory reuse an unmodified dense chunked kernel. This is the engineering result that makes the whole design trainable, and it removes the single largest risk flagged in the digest (FLOP-matching is not wall-clock matching for sparse gathers).

6. WHAT IS DELIBERATELY NOT NEW, AND SAID SO. The delta rule is DeltaNet/Gated DeltaNet and also the fly's KC->MBON rule — this design ADOPTS it rather than competing with it, which is the honest correction of phase 1's error (the phase-1 MB memory lost 0.981 to 0.998 because it was a weaker special case of its own baseline). Sparse top-k memory is PKM/Memory Layers. Divisive normalisation is RMSNorm's family. Negative eigenvalues (beta in (0,2)) are a free one-line fix from Grazzi et al. (ICLR 2025) and earn no credit. Homeostatic renormalisation is LaCT. Hybridisation at 3:1 is Qwen3-Next / Kimi Linear.

BRANCHES THE OWNER NAMED, AND WHAT HAPPENED TO THEM. Active inference: exactly one piece survives — an inferred, residual-normalised write rate — and it enters as an ABLATION ARM tested only out of distribution, because a learned gate with the same inputs contains any fixed rule as a special case and phase 1 already measured that (-2.6 seed-SD). Free-energy branding is dropped entirely; expected-free-energy decoding is dropped (no central-complex correlate, and a measured negative elsewhere). Memory-based modelling: this IS the design. Reservoir computing: rejected — the Dambre bound (Sci Rep 2:514, 2012) caps total capacity at the state width, and this repo already measured a 211k-neuron frozen reservoir LM at 3.644 bpc, tied with a bigram at 3.572. Contrastive learning: admitted in ONE narrow role, as a separation regulariser on the address code, never as the training objective (InfoNCE is capped at log(#negatives) nats). Spiking: rejected — a LIF neuron is provably a 1-D diagonal SSM plus a threshold and a reset, so spiking contributes a leak this design already has plus a 1-bit quantiser that costs accuracy; and most of the relevant fly neurons (photoreceptors, LMCs, many AL local neurons, APL itself) are graded and non-spiking, so a spiking design would be LESS faithful, not more.

### Regime of the claimed advantage

THE DEFINED REGIME. Retention and re-retrieval of ONE-SHOT, IN-CONTEXT associations under a high ratio of one-off "chaff" writes to recurring "signal" writes, as a function of write-to-query DISTANCE, at matched per-token FLOPs and matched parameters, inside a 3:1 hybrid decoder at 60M-125M parameters trained on 3-8B FineWeb-Edu tokens at 2k-4k context. The claimed quantity is the SHAPE of the recall-versus-distance curve — its area over D in [2^8, 2^14] and its log-log slope — not perplexity, not lm-eval average, not tokens per joule.

PRIMARY BENCHMARKS (pre-registered, in this order).
  1. DELAYED-RECALL MQAR: MQAR (Arora et al., Zoology, ICLR 2024, arXiv:2312.04927) with P in {64, 128, 256, 512} key-value pairs and query-write distance D in {2^6 ... 2^14}, with a controllable chaff rate. Never at P = 8; phase-1 H1 ran at 8 pairs against a d_key = 65 store, five times below the point where any capacity effect could appear, and could not have detected the thing it was testing.
  2. RULER S-NIAH-1/2/3 (Hsieh et al., arXiv:2404.06654) at 2x-8x the training length — length extrapolation, not in-distribution.
  3. BABILong qa1-qa5 (Kuratov et al., NeurIPS 2024 D&B, arXiv:2406.10149) at 4k-64k.
SECONDARY (real text): SWDE / FDA / SQuAD zero-shot information extraction, the Based suite (Arora et al., ICML 2024, arXiv:2402.18668). Calibration target from the literature at ~400M: Gated DeltaNet scores SWDE 29.5 / FDA 8.3 / SQuAD 31.3 while a forgetting Transformer scores 52.5 / 64.3 / 30.1 — i.e. this is the axis where constant-state models actually lose, and therefore the axis where an improvement is visible.
NO-REGRESSION CHECKS ONLY (never a claim): lm-eval zero-shot average over ARC-e, ARC-c, HellaSwag, PIQA, WinoGrande, OpenBookQA must be within 0.5 points of the twin; LAMBADA and WikiText perplexity within 2%. MMLU is excluded (below chance at this scale).

STRONGEST COMPETITOR, IN ORDER.
  1. GATED DELTANET (Yang, Kautz & Hatamizadeh, ICLR 2025, arXiv:2412.06464), run by us in the IDENTICAL hybrid, identical tokenizer, identical data order, identical optimiser, at (a) matched per-token FLOPs, (b) matched parameters, (c) matched recurrent-state bytes as three separate arms. Its numbers must come from our runs, never from its paper.
  2. KDA / Kimi Linear per-channel decay — the arm most likely to close the retention gap, because per-channel decay already yields a mixture of exponentials sum_i k_i^2 alpha_i^D. This is the honest strongest competitor for Proposition P-B and the reason the claim is staked on SELECTION (P-C) rather than on multi-timescale retention alone.
  3. TITANS (Behrouz et al., arXiv:2501.00663) — the sign-flipped control: surprise-gated writing to a second, deeper memory. A direct head-to-head on the same substrate.
  4. MEMORY LAYERS AT SCALE (arXiv:2412.09764) / PKM — sparse address, no ladder, no test-time write. Isolates what the cascade adds over a sparse memory layer.
  5. MAMBA-2 (arXiv:2405.21060) and a sliding-window-attention-only hybrid as floor/ceiling references.

WHAT IS EXPLICITLY NOT CLAIMED. No perplexity-per-FLOP claim against a well-tuned transformer or Gated DeltaNet — no biologically derived architecture has ever won that, and at 2k-4k context the attention FLOP term is only ~30% of the parameter term, so there is barely any headroom. No absolute lm-eval or MMLU number against published model cards (Llama-3.2-1B saw ~9T tokens; we will see 8B). No energy or joules-per-token claim — there is no accepted benchmark and no neuromorphic hardware here. No pure-recurrent claim. No claim at matched state bytes (pre-registered as expected-null; running it anyway is what makes the matched-FLOP result credible).

### Killer experiment

THE CHAFF-STREAM RETENTION TEST. Module-level, no language model, 4 CPU cores, roughly 20-25 CPU-hours total. It tests Proposition P-C directly: does a TEST on the write buy long-horizon recall that a RATE on the write cannot?

SETUP. A synthetic event stream of T = 20,000 events over an address vocabulary of 4,096 items. Each event is a write. With probability p it is a SIGNAL write, drawn from a set of 64 recurring keys with a fixed value per key, recurring at log-uniformly spaced gaps; with probability 1-p it is CHAFF: a fresh key never seen again, with a random value. Interleaved query events ask for the current value of a signal key at a write-to-query distance D drawn log-uniformly from [1, 2^14]. Primary statistic: AUC of recall accuracy over D in [2^8, 2^14]. Secondary: fitted log-log slope of recall vs D, and short-delay recall (D < 2^6) as a do-no-harm check. Chaff rates p in {0.02, 0.05, 0.20}; 5 seeds; every learnable parameter in every arm trained by backprop through the module on the recall loss, so the learned controls are as strong as they can possibly be.

ARMS (all FLOP-matched; the whole grid is then re-run state-matched).
  A1 SIEVE-full      sparse address k=32 of n=4096, H=5 ladder tau in {8,64,512,4096,32768},
                     recall gate x spacing gate.
  A2 SIEVE-noselect  IDENTICAL, but gamma_{h,t} = 1 for all h: every write goes to every rung,
                     with a learned per-rung rate. <-- THE MATCHED CONTROL.
  A3 SIEVE-surprise  IDENTICAL, but gamma_{h,t} = sigmoid(a*||v_t - r_{h-1,t}||^2 + b): promote
                     when the fast rung was WRONG (Titans' sign). <-- THE SIGN-FLIP CONTROL.
  A4 SIEVE-flat      H=5 compartments, all alpha equal and learned. Isolates the ladder.
  A5 SIEVE-H1        one compartment, learned decay, same sparse address, 5x value width.
  A6 GDN twin        gated delta rule, dense keys, d_k = k, scalar decay.
  A7 KDA twin        gated delta rule, dense keys, PER-CHANNEL decay. <-- strongest competitor
                     for the retention half of the claim.

PRE-REGISTERED THRESHOLD (written before any run, with the analysis script committed first).
SUPPORTED if and only if, on the FLOP-matched grid:
  (i) AUC(A1) - AUC(A2) > 2 x the larger seed-SD, with a 20,000-sample paired bootstrap 95% CI
      excluding 0, in at least 2 of the 3 chaff rates; AND
  (ii) the same holds for AUC(A1) - AUC(A3), i.e. recall-gating beats surprise-gating; AND
  (iii) A1 does not lose to A2 at short delay (D < 2^6) by more than 1 seed-SD.
This is the project's standing 2-seed-SD + CI rule, unchanged.

FALSIFIED — AND THE PROJECT STOPS — if A1 is within 1 seed-SD of A2 at every chaff rate. That
outcome says selection buys nothing, the design collapses to "a sparse multi-timescale memory",
which is incremental relative to Memory Layers plus HOPE and does not deserve a GPU. It should
then be reported as the third honest null.

TWO FURTHER OUTCOMES, BOTH INFORMATIVE. If A1 beats A2 but A7 (per-channel decay) matches A1,
the retention story is not the discriminator and the surviving claim narrows to capacity per FLOP
(P-A), which is real but much smaller. If A1 beats A2 only at the highest chaff rate, the effect
is genuine but needs a regime argument about text, which diagnostic D1 below supplies or refutes.

GATING DIAGNOSTIC, RUN IN PARALLEL (D1, ~4 CPU-hours). On FineWeb-Edu, measure the distribution
of distances at which a repeated rare token or entity actually reduces next-token loss, using a
kNN-LM-style probe (Khandelwal et al., ICLR 2020, arXiv:1911.00172) over a small pretrained model.
If that distribution has no mass beyond 2^10 tokens, the retention claim has no purchase on text
regardless of how the killer experiment comes out, and the project should narrow to the pure
capacity claim before spending any GPU time. This diagnostic is cheap and should be run FIRST.

### Why phases 1-2 failed and how this differs

PHASE 1 AND PHASE 2 DID NOT FAIL FOR BIOLOGICAL REASONS. Each failed for a reason that was a theorem or a methodology error, identifiable before any run. Naming them precisely is what makes phase 3 different, because each one becomes a design constraint.

FAILURE 1 — TESTING STRUCTURE WHERE ONLY STATISTICS ARE REPRODUCIBLE. Phase 1 used the adjacency matrix as a weight mask. But within a single animal, left and right copies of the same cell type share only ~52% of type-to-type edges (68% for the strongest), and PN->KC convergence is random to first order (Caron et al., Nature 497:113-117, 2013). Roughly half the graph is a developmental sample. 'Real graph = degree-preserving shuffle' was therefore the PREDICTED result, not a disappointment. PHASE 3: no connectome matrix appears anywhere in the model. Connectome numbers enter only as hyperparameter priors (H ~ 7 from ~15 compartments; f* ~ 0.05 coding level; the compartment decay spectrum) and every one is swept rather than copied.

FAILURE 2 — ITERATING ON A QUADRATIC OBJECTIVE. Phase 2's K-step free-energy descent minimised a Gaussian-linear energy whose minimiser is the closed form s* = sum_l pi_l mu_l / sum_l pi_l. The Hessian does not depend on the state, so K gradient steps are exactly equivalent to one affine map. The +0.7 seed-SD result is what the algebra required, and the repo's own DESIGN.md section 1.1 recorded the problem before any run. PHASE 3: there is no inference loop anywhere. The only per-token iteration is the H-step cascade, and it is not descent on a fixed objective — each rung's gate reads the PREVIOUS rung's output, so the computation is genuinely sequential and non-collapsible. Standing rule: before any iterative module is built, compute d^2F/ds^2 and delete the module if it does not depend on s.

FAILURE 3 — A SCALAR GAIN UPSTREAM OF A NORMALISER. Phase 2's arousal multiplied activations and the free energy. RMSNorm(g*x) = RMSNorm(x) exactly (verified in this repo to 3e-7 for g in [0.25,4]), and a constant factor on the loss cancels under any preconditioned optimiser. The unit was inert by theorem. PHASE 3: every modulatory quantity acts on a WRITE RATE (beta), a PROMOTION GATE (gamma), a DECAY (alpha-tilde) or a THRESHOLD (theta) — never on an activation upstream of a norm and never on the loss. The one place a running statistic enters the forward path is as a FEATURE (g_t = log a_t, Fairhall's rule), which is precisely the correct form and precisely what phase 2 got backwards.

FAILURE 4 — A FIXED-FORM GATE AGAINST A LEARNED GATE, IN DISTRIBUTION. Phase 1's H1b: the learned gate beat sigmoid(a log(pi ||eps||^2/n) + b) by 2.6 seed-SD. This is structurally unwinnable — the learned hypothesis class contains the fixed rule. PHASE 3: the control arm A2 has a LEARNED per-rung rate with the same inputs, so the comparison is not gate-form against gate-form; it is presence-versus-absence of a ROUTING TEST that no rate can express, whatever its form. Where a fixed form is still tested (the inferred-precision write rate, the surviving piece of active inference), it is tested only out of distribution, where the learned gate has no training signal for the regime it now faces.

FAILURE 5 — A WEAKER SPECIAL CASE OF THE BASELINE. Phase 1's MB memory lost 0.981 to Gated DeltaNet's 0.998 because, read as an equation, it WAS a gated delta rule with a scalar surprise gate and no per-key delta correction. PHASE 3: the delta rule is adopted, not re-derived. The fly and Gated DeltaNet write the same equation; the design competes on the routing of writes across stages, which is a different object.

FAILURE 6 — MEASURING BELOW THE DETECTION FLOOR, WITH UNMATCHED CONTROLS. 135k parameters on TinyShakespeare, 400-600 optimiser steps, 4 CPU cores; the 'wide' control had 1.94x the parameters and 1.32x the FLOPs; MQAR was run at 8 pairs against a store of capacity 65, five times below where an effect could exist. PHASE 3: the primary endpoints are synthetic retention and capacity curves whose effect sizes are large by construction; controls are matched on FLOPs, parameters and state bytes as three SEPARATE arms with the matched-state arm pre-registered as expected-null; MQAR is run at P in {64,...,512}; perplexity is a no-regression check only; and the scale ladder puts the decisive language-model test at 60-125M rather than 135k.

FAILURE 7 — GATES THAT NEVER FIRED, UNDETECTED. Phase 2's adaptive-K rule never fired (K_mean = 3.000 in every single run) and this was only noticed afterwards. PHASE 3: the distribution of gamma and beta is a REPORTED DIAGNOSTIC at every rung, and a degenerate gate distribution is a stop condition at rung 1, before any GPU spend.

THE CHANGE IN KIND. Phases 1 and 2 imported a mechanism from biology and hoped it would help. Phase 3 identifies one mathematical property the biology has and current memories do not — write routing decided by a test rather than a rate — states the load arithmetic that makes it matter before anything is run, builds the cheapest possible experiment whose failure kills it in twenty CPU-hours, and commits in advance to reporting the null.

### Cost

ENGINEERING: one engineer, 8-12 weeks full time. Breakdown: SIEVE module in PyTorch with the chunk-union gather/scatter wrapper around flash-linear-attention's chunk_gated_delta_rule (2-3 weeks, the main technical risk); APL servo, promotion gates, timestamp bookkeeping and the per-row decay renormalisation (1 week); module-level evaluation harness and the chaff-stream generator (1 week); control arms via `pip install flash-linear-attention` plus a PKM baseline (1 week); LM training and evaluation harness — delayed-MQAR, RULER S-NIAH, BABILong, the Based SWDE/FDA/SQuAD suite, lm-eval (2-3 weeks); pre-registration documents, analysis scripts and the results builder, reusing the repo's existing scripts/build_results.py machinery (1 week). Environment is ready: torch 2.14.0+cu130, triton 3.8.0, numpy 2.4.6 already installed.

COMPUTE, BY RUNG:
  Rung 0 (CPU, 4 cores): ~25 CPU-hours. 210 module runs at ~5 min each, plus the three diagnostics. Effectively free. Wall clock ~2 days of machine time inside a 1-2 week implementation window.
  Rung 1 (1 GPU): ~50 GPU-hours, ~1 week wall clock.
  Rung 2 (one 16 GB GPU): 24 runs x ~5 h = ~120 GPU-hours, plus ~20 GPU-hours of pilot LR selection and evaluation. ~5-7 days wall clock. Arithmetic: 60M params x 3B tokens = 6*6e7*3e9 = 1.08e18 training FLOPs per run; at ~60 TFLOP/s effective bf16 on a 4090/L4/A10-class card that is ~5 h.
  Rung 3 (one 16 GB GPU): 125M x 8B tokens = 6e18 FLOPs per run ~ 28 h; 4 arms x 3 seeds = 12 runs = ~340 GPU-hours, plus ~100 GPU-hours for the ablation ladder at reduced token count and ~60 GPU-hours of long-context evaluation. Total ~500 GPU-hours, ~3-5 weeks wall clock on one card.
  Rung 4 (NOT on this hardware): ~2,000-5,000 GPU-hours on 8xA100/H100; requires external compute and is gated on rung 3.

TOTAL TO A PUBLISHABLE RESULT OR A CLEAN NULL: ~25 CPU-hours + ~670-700 GPU-hours on a single 16 GB card, i.e. roughly 5-7 weeks of continuous GPU time, or about USD 300-600 rented at USD 0.40-0.80/hour for a 4090/L4/A10-class instance. Plus 8-12 weeks of one engineer's time, the dominant cost.

KILL POINTS AND THEIR COST. The whole programme can be terminated for ~25 CPU-hours at rung 0 if the killer experiment refutes selective promotion, or for ~4 CPU-hours if diagnostic D1 shows text has no recall-useful mass beyond 2^10 tokens. That asymmetry — a 25-CPU-hour falsifier guarding a 700-GPU-hour programme — is the single most important budgetary property of this design and the direct correction of phase 2, which spent its compute before its theory.

### Ladder

1. RUNG 0 — CPU CALIBRATION (4 cores, ~25 CPU-hours, ~1-2 weeks incl. implementation). (a) Diagnostic D1: the FineWeb-Edu recall-distance probe (run first, ~4 CPU-hours). (b) The killer experiment: 7 arms x 3 chaff rates x 5 seeds x 2 matchings = 210 module-level runs, ~135k params each. (c) Diagnostic D2: the APL-servo shift test — inject a mid-stream rescaling and rotation of the input statistics, measure post-shift recall with the servo on and off. (d) Diagnostic D3: wall-clock benchmark of the chunk-union densified kernel against flash-linear-attention's chunk_gated_delta_rule at matched FLOPs. GO/NO-GO: killer experiment SUPPORTED per its threshold, AND D1 shows recall-useful mass beyond 2^10 tokens, AND D2 shows the servo recovers >=80% of pre-shift recall where the fixed threshold recovers <50%, AND D3 shows <2x wall-clock penalty per FLOP. Any failure -> stop or narrow the claim; do not touch the GPU.
2. RUNG 1 — SMALL SYSTEM TEST (1 GPU, ~50 GPU-hours, ~1 week). The SIEVE layer inside a 2-4 layer, 5-20M-parameter toy LM trained end to end. Delayed-recall MQAR at P in {64,128,256,512} x D in {2^6..2^11}, against FLOP-matched Gated DeltaNet, KDA, DeltaNet, Mamba-2 and a PKM-style sparse-memory-without-ladder, 5 seeds. This is where phase 1's H1 failure mode (the memory is present but the trainer never learns to use it) is caught. GO/NO-GO: dominance at >=2 of 4 P values by >2 seed-SD with CI excluding 0, AND no regression on ordinary MQAR at P=8, AND the SIEVE arm's gates measurably fire (report gamma distributions; phase 2's adaptive-K rule never fired, K_mean = 3.000 in every run — that failure must be detectable here).
3. RUNG 2 — 60M LANGUAGE MODEL (one 16 GB GPU, ~120 GPU-hours, ~5-7 days). 16 layers, d=512, 3:1 SIEVE:SWA, 3B FineWeb-Edu tokens, ctx 2048, ~1.1e18 FLOPs/run ~ 5 h/run. Grid: 4 arms (SIEVE, Gated DeltaNet, KDA, PKM-no-ladder) x 2 matched state sizes x 3 seeds = 24 runs. Learning rate fixed per arm by a single 300M-token pilot, never swept inside the grid. Primary: delayed-recall MQAR and RULER S-NIAH at 2x/4x the training length. Secondary: SWDE/FDA/SQuAD. No-regression: lm-eval zero-shot average within 0.5 points, LAMBADA ppl within 2%. GO/NO-GO: primary dominance at BOTH state sizes by >2 seed-SD with CI excluding 0, and the no-regression checks satisfied. A win at matched state bytes would be a red flag (the state-size bound forbids it) and should trigger an audit for a leak, not a celebration.
4. RUNG 3 — 125M LANGUAGE MODEL AND THE PAPER (one 16 GB GPU, ~400-600 GPU-hours, ~3-5 weeks). 24 layers, d=768, 8B tokens, ctx 4096. Add BABILong qa1-qa5 at 4k-64k, full RULER, and the continual/streaming arm: continually pre-train on a domain-sorted stream and compare retained perplexity on held-out old domains against the incumbent recipe (LR re-warm + re-decay + ~5% replay; Ibrahim et al., TMLR/COLM 2024, arXiv:2403.08763), claiming parity at <=1% stored replay using the offline consolidation pass instead. Also run the ablation ladder: H in {1,3,5,7}; gamma on/off; spacing gate on/off; recall vs surprise gate; APL servo on/off; active vs constant forgetting; novelty-gated adaptive compute on/off; learned vs fixed address. GO/NO-GO: a frontier plot in which SIEVE dominates the Gated DeltaNet and KDA twins on recall-vs-distance at >=2 of 3 state sizes, with every ablation reported including the ones that come out null. This rung produces either a paper with a regime-bounded SOTA claim, or a clean fourth negative result.
5. RUNG 4 — SCALE CHECK (NOT on the available hardware; ~2,000-5,000 GPU-hours, needs external compute). 350M-1.3B parameters on 30-100B tokens at 4k-8k context, 3:1 hybrid, SIEVE vs Gated DeltaNet twin only. GO/NO-GO GATE SET IN ADVANCE: proceed only if the effect size in seed-SD units does NOT shrink between the 60M and 125M rungs. A shrinking effect from 60M to 125M is the signature of a small-scale artefact and must end the programme rather than be scaled through. Do not seek this compute before rung 3 is complete and written up.

### Risks

- R1 (HIGHEST). Per-channel decay may already do it. KDA / GDN-2 / Mamba-2 read a key at delay D as sum_i k_i^2 alpha_i^D — already a mixture of exponentials, already potentially log-tailed. If the advantage is really about multi-timescale retention rather than about SELECTION, the strongest competitor closes the gap and the design is incremental. Mitigation: the claim is staked on selection (P-C), not on the ladder (P-B); arm A7 in the killer experiment tests this at rung 0 for a few CPU-hours; and the thesis is written so this outcome is a clean, early, cheap refutation rather than a late surprise.
- R2. The learned gate objection, again. Phase-1 H1b measured a learned input-dependent gate beating a hand-specified error gate by 2.6 seed-SD. A learned gate with the same inputs contains the recall gate and the spacing gate as special cases, so in distribution backprop will find them and the arms will tie. Mitigation: arm A2 (SIEVE-noselect) has a LEARNED per-rung rate with the same inputs, so it is the strong control by construction; and the regime is chosen so that the discriminating structure (which writes recur, at what gaps) varies at test time in ways the training stream did not contain. If A1 ties A2 in distribution and only wins out of distribution, that is a narrower but still honest claim and must be reported as such, not as a general win.
- R3. Wall-clock is not FLOPs. Sparse gather/scatter can be 5-20x slower than a dense matmul of the same nominal FLOP count. Mitigation: the chunk-union densification (|U| <= C*k = n/8) reduces the inner loop to an unmodified dense kernel; diagnostic D3 at rung 0 is a hard gate at 2x; every result reports tokens/s and joules-free wall-clock alongside FLOPs, and a >2x penalty forces larger address blocks before anything else proceeds.
- R4. State bytes. The reference configuration holds 14.7 MB of memory state per SIEVE layer per sequence — enormous next to a Gated DeltaNet layer's ~0.5 MB. This is the honest price of capacity n, it constrains batch size on a 16 GB card, and it means any matched-state-bytes comparison is expected to be null (the recall-versus-state-size bound). Mitigation: pre-register the matched-state arm as expected-null and run it anyway; report state bytes as a first-class axis on every plot; shrink n at rung 2 if memory pressure caps the batch below 8.
- R5. The attention layers may absorb the effect. In a 3:1 hybrid at 2k context the 25% sliding-window attention layers can do all the recall the benchmark needs, masking any memory-layer improvement. Mitigation: run an attention-free ablation at rung 1-2, and make the primary long-context evaluations (S-NIAH at 4x-8x training length, BABILong at 16k-64k) sit well outside the attention window.
- R6. Scale. At 60-125M on 3-8B tokens the field's own spread across seven architectures is under 1 point of lm-eval average, and Mixture-of-Recursions only starts paying above 360M. There is a real chance the effect exists and is invisible here. Mitigation: primary endpoints are synthetic retention curves where effect sizes are large by construction; perplexity is no-regression only; rung 4 exists precisely to answer the scale question and is gated on the effect NOT shrinking from 60M to 125M.
- R7. Novelty collision. 'Multi-timescale memory layer' is close to Nested Learning / HOPE's continuum memory system and to mixture-of-memories designs; 'sparse top-k memory' is PKM, PEER, UltraMem and Memory Layers at Scale; 'input-dependent decay' is KDA. The unoccupied part is narrow: a promotion TEST between stages over a shared sparse address. Mitigation: a dedicated literature check on arXiv and OpenReview for 'recall-gated', 'consolidation gate', 'promotion between memory stages', 'spacing' plus 'test-time memory' BEFORE rung 2 spends GPU time; if it is taken, the design's remaining claim is the spacing gate and the APL servo, which is smaller and should be re-scoped rather than oversold.
- R8. Anisotropic LM keys break fixed sparse addressing. The one preregistered attempt recorded in this repo's sweep hard-failed at 100k adversarially similar keys. Mitigation: the front end (Weber-Fechner gain, divisive normalisation) and the APL servo are both mandatory and both ablated; measured code overlap E[<k_t,k_t'>] and dead-address fraction are reported as standing diagnostics, with the expected overlap k^2/n as the reference line.
- R9. The consolidation pass may again buy nothing. This repo already measured naive replay at 0.06 vs 0.02 retention beyond the fast horizon. Selection by stored gate value, multi-pass replay and a slow store larger than the stream are the specific fixes, and none is guaranteed. Mitigation: consolidation is deliberately the SECOND hypothesis and is kept out of the critical path; the core claim stands or falls on the online promotion gate alone.
- R10. Researcher degrees of freedom. Seven arms, three chaff rates, two matchings, eight ablations and five benchmarks is a large enough grid to find something. Mitigation: PREREG.md with the primary statistic, the arms, the seeds, the threshold and the analysis script committed before any cross-condition result is looked at, exactly as phases 1 and 2 did; the 2-seed-SD plus bootstrap-CI rule unchanged; a replication grid on held-out seeds; and every ablation reported whether it helps or not.
- R11. Honest overall probability. P(the rung-0 killer experiment is SUPPORTED) ~ 0.65, since selective filtering of a chaff-dominated stream is close to forced by the load arithmetic and the main risk is arm A7 matching it. P(the effect survives to rung 2 as a >2 seed-SD win in a trained 60M LM against a Gated DeltaNet twin) ~ 0.30-0.35. P(a defensible, reviewable, regime-bounded SOTA claim at rung 3) ~ 0.15. P(any general language-model SOTA claim) ~ 0.00-0.02; that is not the target and should never be stated as one. The expected outcome of this programme, weighted, is a well-characterised negative result with two or three transferable positive components (the servo, the capacity-per-FLOP curve, the promotion gate), which is a worthwhile scientific return on ~600 GPU-hours.

## Design [sota-first]: CLAW — Compartmentalised Latent Associative Writes: a sparse-addressed, multi-compartment test-time memory layer for a hybrid LM

### Axioms

- A1. THE CONNECTOME IS A SAMPLE, NOT A BLUEPRINT. Within one animal, left and right copies of the same cell type share only ~52% of type->type edges (68% for the strongest; measured in this repo). Caron et al. (Nature 497:113, 2013) established PN->KC convergence is apparently random. ENGINEERING CONSTRAINT: Phase 3 loads ZERO adjacency matrices. The connectome supplies (a) distributional scalars, (b) a mechanism inventory (which functional parts exist and what each one does), (c) a validation check on derived optima. Nothing else. This also sidesteps the repo's own data defect that every monoaminergic edge in data/malecns/graph.bin has weight 0 and W_dan_kc/W_dan_mbon export as all-zero arrays.
- A2. THE FLY COMPUTES IN GRADED SIGNALS, NOT SPIKES, AT EVERY PLACE THAT MATTERS HERE. APL -- the single inhibitory neuron that sets the sparse code -- is non-spiking and graded (Lin et al., Nat Neurosci 17:559, 2014; Amin et al., eLife 9:e56954, 2020); photoreceptors, LMCs and many AL local interneurons emit no action potentials; every quantitative MB model that reproduces behaviour (Gkanias 2022; Bennett 2021; Jiang & Litwin-Kumar 2021) is a rate model. CONSTRAINT: no spikes anywhere. Sparsity is a THRESHOLD on a graded variable, not an event. A spiking design would be LESS fly-faithful, not more.
- A3. SPARSITY EXISTS SO THAT A SINGLE LOCAL WRITE TOUCHES O(k) SYNAPSES, AND ITS LEVEL HAS AN INTERIOR OPTIMUM SET BY INPUT SNR AND TASK SMOOTHNESS -- it is not 'as sparse as possible' and 5% is a convention, not a constant (Turner et al. 2008; Honegger et al. 2011; Babadi & Sompolinsky, Neuron 2014; Xie et al., eLife 12:e82914, 2023). CONSTRAINT: coding level f and fan-in are first-class swept hyperparameters chosen by maximising the participation ratio (tr C)^2/tr(C^2) on real token data, never copied from K=6/f=0.05; and every read and every write must cost O(k), not O(n).
- A4. NORMALISE BEFORE EVERY THRESHOLDED PROJECTION, AND AGAIN AS SCALAR FEEDBACK AFTER IT -- and the inhibition is LOCAL, not global. The antennal lobe divides by pooled activity (exponent n~1.5, compressing a 57x ORN range to 6.6x; Olsen, Bhandawat & Wilson, Neuron 66:287, 2010); APL has a ~50 um space constant and acts as at least four independent local normalisers (Amin 2020). CONSTRAINT: per-BLOCK feedback inhibition computed by an explicit unit, not a pointwise LayerNorm and not one global k-WTA. This is also the documented fix for the dead-latent pathology of top-k SAEs (up to 90% dead latents; Gao et al., arXiv:2406.04093).
- A5. WHENEVER YOU DIVIDE BY A RUNNING STATISTIC, YOU MUST ALSO TRANSMIT THAT STATISTIC. Rapid gain rescaling creates an ambiguity the fly resolves by sending the slow gain variable itself as a signal (Fairhall, Lewen, Bialek & de Ruyter van Steveninck, Nature 412:787, 2001). CONSTRAINT: adaptation state, log population drive and novelty are FEATURES fed forward, never only gains. This is the precise diagnosis of Phase 2's inert arousal unit.
- A6. NEUROMODULATION ACTS ON PLASTICITY AND TIMESCALE, NEVER ON ACTIVATION MAGNITUDE. RMSNorm(g*x) = RMSNorm(x) exactly (verified in this repo to 3e-7 for g in [0.25,4]), and a global scalar on the loss cancels under any preconditioned optimiser. The same bounded scalar applied to a decay gate moves the memory horizon 16x. CONSTRAINT: every modulatory scalar's only legal targets are the write gate beta, the decay gate alpha, the eligibility time constant lambda, the cross-compartment read mix, and the output temperature. This is a theorem, not a tuning rule.
- A7. THE TEACHING SIGNAL IS LOCAL AND PER-COMPARTMENT, MANUFACTURED BY THE CIRCUIT'S OWN READOUT. Dopamine is spent ~86% on ~15-20 independently gated MB compartments (this repo: 316/392 predicted DANs are PAM, 158 PAM + 8 PPL1 per side), and MBON->DAN feedback closes a local error loop inside the layer (339/300 edges, 2,861/2,605 contacts; Eschbach 2020; Bennett 2021). Flies show no reliable Kamin blocking, and Gkanias et al. (eLife 11:e75611, 2022) reproduce 92 experiments with a purely LOCAL rule that PREDICTS no blocking. CONSTRAINT: per-compartment residual errors with no shared error budget; no global scalar RPE; the gate for compartment h takes compartment h's own readout as an input.
- A8. THE WRITE IS THREE-FACTOR AND PRESYNAPTIC x NEUROMODULATOR, NOT HEBBIAN -- the postsynaptic MBON's own firing is essentially irrelevant to the weight change (Hige et al. 2015). In vector form this is exactly the delta rule, i.e. DeltaNet (Schlag et al., ICML 2021; Yang et al., arXiv:2406.06484). CONSTRAINT: the gate is a function of (address activity) x (teaching signal), never of the read value; and because the delta rule IS the biology, any new design must CONTAIN Gated DeltaNet as an exact special case rather than being a degraded version of it. Phase 1 lost 0.981 to 0.998 precisely because it built a weaker special case of its own baseline.
- A9. REVISION IS NON-DESTRUCTIVE. The circuit never erases a trace; extinction writes an OPPOSING trace into a parallel compartment and the readout is a signed sum, which is why extinguished memory spontaneously recovers (Felsenberg et al., Nature 544:240, 2017; Cell 2018). Each compartment's weights are non-negative and its DAN can only depress. CONSTRAINT: paired non-negative channels read as a difference; rebinding writes to the OTHER channel.
- A10. THE WRITE KERNEL IS BIPHASIC IN TIME AND SIGNED BY ORDER. The same synapse depresses for forward (odour->dopamine) and potentiates for backward pairing, through Dop1R1-Gs-cAMP and Dop1R2-Gq-Ca2+ with opposite second messengers and a zero-crossing near simultaneity (Tanimoto et al., Nature 430:983, 2004; Handler et al., Cell 178:60, 2019; Cohn et al. 2015). CONSTRAINT: a short learned causal kernel over the address stream, initialised biphasic. This has NO counterpart in DeltaNet, Gated DeltaNet, KDA, GDN-2, Titans, ATLAS, LaCT, MesaNet or PGDN.
- A11. COMPARTMENTS ARE DELIBERATELY HETEROGENEOUS IN SIGN, DECAY, CAPACITY AND FLEXIBILITY, AND THE HETEROGENEITY IS ARCHITECTURAL, NOT LEARNED (Aso & Rubin, eLife 5:e16135, 2016; Aso et al., eLife 3:e04577, 2014). Memory phases span minutes to a week, i.e. 10^3-10^5 in time constant. CONSTRAINT: a fixed log-spaced decay ladder over >= 10 octaves with sign and write capacity UNTIED from decay -- not the 64x span (4/16/64/256) Phase 2 used.
- A12. FORGETTING IS A DRIVEN, STATE-DEPENDENT PROCESS, NOT PASSIVE DECAY, and it is switched off during sleep (Shuai et al., Cell 140:579, 2010; Berry et al., Neuron 74:530, 2012; Cell 161:1656, 2015; Sabandal et al., Nature 591:426, 2021). CONSTRAINT: the decay gate is an explicit function of recent write volume, and is clamped to 1 during the offline consolidation pass.
- A13. DURABLE STORAGE REQUIRES SPACED RECURRENCE PLUS AN OFFLINE PHASE WHOSE REPLAY SET IS SELECTED BY THE MODULATORY TAG APPLIED AT ENCODING. 10 sessions x 15 min rest gives 7-day LTM; the same trials massed give none (Tully et al., Cell 79:35, 1994). Post-learning sleep reactivates specifically the DANs engaged at acquisition (Dag et al., eLife 8:e42786, 2019), and recall-gated consolidation makes the learnable timescale grow LINEARLY rather than logarithmically with repetitions (Lindsey & Litwin-Kumar, eLife 13:e90793, 2024). CONSTRAINT: consolidation is triggered by accumulated write magnitude, gated by spacing AND by successful fast recall, and replays proportional to the stored write tag -- never uniformly.
- A14. THE CIRCUIT HAS AN UNSUPERVISED TEACHING SIGNAL -- NOVELTY -- COMPUTED BY READING THE MEMORY ITSELF, and it habituates over minutes (MBON-alpha'3; Hattori et al., Cell 169:956, 2017). CONSTRAINT: a familiarity compartment whose read gates the other compartments' writes and allocates compute, at O(k) cost and with no reference to the loss.
- A15. THE MUSHROOM BODY'S LEARNED OUTPUT BANDWIDTH IS TENS OF SCALARS PER EVENT (48-49 MBONs per side in this repo's extraction). CONSTRAINT: it is the MEMORY AND CONTROL MODULE of a model whose backbone is something else. It is never a backbone, never a sequence mixer on its own, and never an anatomical I/O port (measured here at -0.63 nat/token for ALPN-in/MBON-out vs random ports, 6/6 seeds).
- A16. BACKPROPAGATION PLAYS THE ROLE OF EVOLUTION; LOCAL RULES PLAY THE ROLE OF LIFETIME LEARNING. Nobody has trained a backprop-free LM above ~100M parameters; iPC loses to BP on the causal objective (142.5 vs 113.3 ppl, arXiv:2212.00720). CONSTRAINT: slow weights AND all plasticity-rule parameters are trained by backprop; only the compartment STATES are written locally at test time. Removing backprop is not on the table.

### Thesis

THE ONE PRINCIPLE: **the mushroom body decouples memory capacity from per-event compute by writing one-shot, error-corrected, non-destructive traces into a bank of heterogeneous compartments that all read a single SPARSE address.** A Kenyon-cell code of ~100 active of ~2,000 cells means a write touches ~5% of the substrate, so capacity scales with the number of addresses (n) while cost scales with the number of active ones (k). Every compartment (MBON/DAN pair) reads the same address stream, so adding a compartment is nearly free; each has its own sign, decay and write rate; and contradiction is handled by writing an opposing trace in a parallel compartment rather than by overwriting.

WHY THIS COULD MATTER FOR LANGUAGE. Every competitive 2026 sequence-mixing memory -- DeltaNet, Gated DeltaNet (Qwen3-Next), KDA (Kimi Linear), GDN-2, Titans, ATLAS, LaCT, MesaNet, PGDN -- is a single DENSE matrix state S in R^{d_k x d_v} read by o = S^T q. Such a state is a rank-d_k object: it holds about d_k mutually-resolvable associations, and d_k is ALSO its per-token cost. Capacity and compute are the same number, and every stored item interferes with every new write in proportion to <k_i, k_j>, which no model in that family attacks because all of them use dense L2-normalised keys. The mushroom body breaks that identity. This repository already measured the size of the break, in numpy, before any design existed (docs/phase3/MEMORY_CAPACITY.md): at equal active rows per token, a sparse 100-of-2048 delta memory recalls 0.82 at 256 stored pairs and 0.70 at 512, where a dense delta memory with d_key=100 recalls 0.17 and 0.07. That is a 4-10x factor from one structural change, and its mechanism is a theorem (expected overlap of two k-of-n codes is k^2/n ~ 4.9 units against a signal of k=100), not a hope.

The arithmetic of the concrete layer below makes the trade exact: at d=768, n_a=2048, k=64, H=8, d_v=64 the layer moves 196,608 state floats per token per layer -- IDENTICAL to a Gated DeltaNet layer at d_k=d_v=128 with 6 heads -- while addressing 16x as many rows, at 4.6x FEWER MACs and 5.6x fewer mixer parameters. The cost is 10.7x the resident state memory. That is the whole bet, stated as an economics claim rather than a capability claim: **sparse addressing converts resident memory into recall more efficiently than widening a dense recurrent state, at constant per-token compute and constant per-token bandwidth.**

The five other fly mechanisms in the design (biphasic write kernel, opposing non-negative channel pairs, within-layer residual allocation across compartments, an architectural decay ladder, a novelty compartment that gates writes) are not the thesis. They are separately-ablatable riders, each of which has no counterpart in the 2026 fast-weight family and each of which is individually falsifiable in the same cheap experiment.

### Architecture

CLAW = Compartmentalised Latent Associative Writes. A drop-in token-mixer layer inside a standard hybrid decoder. Everything below is PyTorch-implementable by one engineer; no Triton is required for the calibration and 60M runs.

================================================================
0. BACKBONE (deliberately conventional -- axiom A15/A16)
================================================================
Pre-RMSNorm decoder, 3:1 schedule as in Qwen3-Next (48 layers, full_attention_interval 4) and Kimi Linear (27 layers, 3:1 KDA:MLA). Block l:
  x <- x + Mixer_l(RMSNorm(x));  x <- x + SwiGLU(RMSNorm(x))
Mixer_l = CLAW for l mod 4 != 0, sliding-window attention (window 512, RoPE) for l mod 4 == 0.
Configs: CPU-calibration d=256, L=4 (1 attn), ~5M params. GPU-main d=768, L=12 (9 CLAW + 3 SWA), d_ff=2048, vocab 32k, tied embeddings, ~110-125M params. All slow weights trained by backprop (A16). The memory states M are NOT parameters; they are initialised to zero at every sequence start and written only by the local rule.

================================================================
1. GLOMERULAR STAGE: compress, adapt, normalise, and TRANSMIT THE NORMALISER (A4, A5)
================================================================
Per token t, from the normed residual x_t in R^d:
  (1.1)  u_t = W_pn x_t,                     W_pn in R^{d_pn x d},  d_pn = 64
  (1.2)  a_t^adapt = rho_a a_{t-1}^adapt + (1 - rho_a)|u_t|,   rho_a = exp(-1/tau_a), tau_a = 64 tokens (FIXED; the Weber-Fechner arm, ORN tau ~ 250 ms)
  (1.3)  u_hat_{t,i} = u_{t,i} / (eps + a^adapt_{t,i})
  (1.4)  p_{t,i} = R * sign(u_hat) |u_hat_{t,i}|^n / (sigma^n + (kappa * sum_j |u_hat_{t,j}|)^n),   n LEARNED per layer, init 1.5 (Olsen et al. 2010); sigma, kappa, R learned scalars
  (1.5)  s_t = log(eps + sum_j |u_hat_{t,j}|)          <-- the normaliser's own state, FED FORWARD as a feature (A5). This single scalar is the corrected form of Phase 2's inert arousal unit.

================================================================
2. ADDRESS STAGE: hierarchical block routing + APL feedback inhibition (A3, A4)
================================================================
n_a addresses in B contiguous blocks of size m = n_a/B. Default n_a=2048, B=32, m=64.
  (2.1)  block scores  c_t = W_B p_t,   W_B in R^{B x d_pn}.  Select the top-J blocks, J=4 (straight-through; ties broken by index).
  (2.2)  within-selected-block scores  z_{t,i} = <A_i, p_t> for i in the J selected blocks only.  A in R^{n_a x d_pn} is DENSE and LEARNED.
         -- NOTE, and this is a place where the evidence overrules the anatomy: the repo's own H1a found a dense LEARNED projection matched the fixed sparse random expansion (+0.0413, CI includes 0) while hard k-WTA was worth +0.0666 (+1.1 seed-SD), and BioHash (Ryali, Hopfield, Grinberg & Krotov, ICML 2020) shows learned sparse projections beat the data-independent fly hash. The value is in the NORMALISATION, not in the random matrix. So: keep the competitive normalisation, learn the projection, and get the cost saving from block routing instead of from a fixed fan-in.
  (2.3)  APL: one graded inhibitory unit PER BLOCK (A4), two feedback iterations, local reach:
             g^{(b)} = (1/m) sum_{i in b} relu(z_i);   z_i <- relu(z_i - lambda_b g^{(b)} - theta_b),  lambda_b, theta_b learned per block
         theta_b is additionally driven toward a running (1-f) quantile of z within block b by an online quantile update (target coding level f, swept in [0.01, 0.20]).
  (2.4)  hard top-k_b within each selected block (k_b = ceil(f*m)), gradient by straight-through on the surviving relu'd values; total active k = J*k_b = 64 (global f = 3.1%).
  (2.5)  a_t = the resulting k-sparse non-negative vector, L2-normalised: ||a_t||_2 = 1.
COST: (2.1) B*d_pn = 2,048 MACs; (2.2) J*m*d_pn = 16,384 MACs. Total address cost 18,432 MACs -- 28x cheaper than scoring all n_a densely, two small dense matmuls, and the selected rows are CONTIGUOUS so the later gathers are coalesced. This is the single most important engineering decision in the design: it is what makes the sparse memory GPU-viable rather than FLOP-viable-but-slow.

================================================================
3. BIPHASIC ELIGIBILITY (A10) -- new, and the cheapest genuinely-novel mechanism here
================================================================
Keep a ring buffer of the last L_elig address index-sets and weights (L_elig=4; k*L integers + floats, ~256 entries). For compartment h:
  (3.1)  e_t^{(h)} = sum_{tau=0}^{L-1} kappa_{h,tau} a_{t-tau},   then renormalise ||e_t^{(h)}||_2 = 1
  kappa_h in R^L is LEARNED, initialised biphasic with a zero-crossing at tau ~ 1-2:
         kappa_{h,tau} = c1 exp(-tau/tau_1) - c2 exp(-tau/tau_2),  tau_1 = 0.8, tau_2 = 3.0, c1,c2 > 0
  One gather of k*L rows serves ALL H compartments; only the L weights differ per compartment. Reduces exactly to the instantaneous key at L=1, kappa=[1].
  INTERPRETATION: this gives the memory a native notion of "which address PRECEDED this teaching signal", signed by order. No model in the DeltaNet/GDN/KDA/GDN-2/Titans/ATLAS/LaCT/MesaNet/PGDN family has any temporal kernel on the key.

================================================================
4. THE COMPARTMENT BANK (A3, A7, A8, A9, A11)
================================================================
H compartments, organised as H/2 OPPOSING PAIRS. States M^{h+}, M^{h-} in R_{>=0}^{n_a x d_v} (non-negativity enforced by clamp after each write; A9). H=8, d_v=64.

READ (gather of k*L rows, per pair h):
  (4.1)  r_t^{(h)} = (e_t^{(h)})^T ( M_{t-1}^{h+} - M_{t-1}^{h-} )   in R^{d_v}

DECAY -- LAZY, so it is O(k) and not O(n_a). Keep per-row last-touch timestamps T[i]:
  (4.2)  on touching row i at time t:  M[i] <- alpha_h^{(t - T[i])} * nu_t * M[i];  T[i] <- t
  where alpha_h = exp(-1/tau_h) is the compartment's FIXED base decay from a log-spaced ladder over >= 10 octaves, tau_h in {2^2, 2^4, 2^6, 2^8, 2^10, 2^12, 2^14} tokens (A11 -- not Phase 2's 64x span), and nu_t is the ACTIVITY-DEPENDENT forgetting multiplier (A12):
  (4.3)  nu_t = sigmoid(-c_1 * act_t + c_0),  act_t = rho_w act_{t-1} + (1-rho_w) sum_h beta_t^{(h)} ||eps_t^{(h)}||,  nu_t := 1 during the consolidation pass.

WRITE -- delta rule, three-factor, presynaptic x neuromodulator (A8). Target v_t = W_v x_t in R^{d_v}.
WITHIN-LAYER RESIDUAL ALLOCATION (A7; the MBON->DAN teaching graph as boosting). Compartments are visited in G sequential groups (G=2; within a group they are parallel). eps_t^{(0)} = v_t. For group g, for each pair h in g:
  (4.4)  beta_t^{(h)} = sigmoid( w_h^T [ log||eps_t^{(g-1)}||^2 , log pi_t^{(h)} , s_t , nov_t , log||e_t^{(h)}||_1 ] + b_h )     <-- a LEARNED gate whose INPUTS are the biological quantities. Phase 1 H1b showed a learned gate beats a fixed sigmoid(a log(pi||eps||^2/n)+b) by 2.6 seed-SD; the fly does not use a fixed form either (its gate is a ~20-DAN-type recurrent critic receiving MBON feedback). The fly's contribution is (i) one gate per compartment, (ii) the compartment's own readout as an input, (iii) the gate being far smaller than the module it controls.
  (4.5)  delta_t^{(h)} = eps_t^{(g-1)} - (e_t^{(h)})^T (decayed M^{h+} - decayed M^{h-})          (post-decay, pre-write residual)
  (4.6)  NON-DESTRUCTIVE SPLIT (A9). Route the positive and negative parts of the correction to the two channels:
           M_t^{h+} = decayed M^{h+} + beta_t^{(h)} e_t^{(h)} relu( delta_t^{(h)} )^T
           M_t^{h-} = decayed M^{h-} + beta_t^{(h)} e_t^{(h)} relu(-delta_t^{(h)} )^T
         Rebinding a key therefore ADDS an opposing trace instead of erasing the old one. Because the two channels have different effective loads and the decay is per-row and lazy, the old binding re-emerges as the newer opposing trace decays -- spontaneous recovery, which a single gated matrix cannot produce.
  (4.7)  eps_t^{(g)} = eps_t^{(g-1)} - sum_{h in g} (e_t^{(h)})^T (M_t^{h+} - M_t^{h-})

NOVELTY COMPARTMENT (A14): M_fam in R^{n_a x 1}, written with v = 1 and beta = 1 always, tau_fam = 2^12:
  (4.8)  nov_t = 1 - clip( a_t^T M_fam , 0, 1 )
  Used as (i) an input to every beta gate, (ii) a chunk-boundary / consolidation trigger, (iii) optional compute allocation (extra layer pass on novel tokens). O(k) cost, no reference to the loss.

PRECISION TRACKER (the ONLY surviving piece of active inference). Per compartment, a Gamma posterior on the residual precision:
  (4.9)  a_t^{(h)} = rho_pi a_{t-1}^{(h)} + d_v/2 ;   b_t^{(h)} = rho_pi b_{t-1}^{(h)} + ||delta_t^{(h)}||^2/2 ;   pi_t^{(h)} = a_t^{(h)}/b_t^{(h)}
  log pi enters the gates (4.4) and the output gate (4.10) as a FEATURE, never as a multiplier upstream of a norm (A6). Marginalising the Gamma gives a Student-t likelihood, so the write becomes an IRLS step rather than a fixed-rate SGD step -- the one place where iteration/weighting is mathematically non-redundant. Per the repo's own ADAPTIVE_RATE.md this buys tuning-freedom under non-stationarity (0.81 vs 0.78-0.80 for the best fixed rate, matching a change-point Kalman ORACLE), NOT accuracy, so it is claimed ONLY out-of-distribution.

OUTPUT (precision-conditioned gate; the Gated-Attention mechanism of Qiu et al., NeurIPS 2025, arXiv:2505.06708, with the gate made a function of inferred uncertainty rather than of the query):
  (4.10) g_t^{(h)} = sigmoid( w_g^T [ log pi_t^{(h)}, nov_t, s_t ] + b_g )
  (4.11) o_t = W_o [ g_t^{(1)} r_t^{(1)} ; ... ; g_t^{(H/2)} r_t^{(H/2)} ; nov_t ; s_t ; act_t ; log pi_bar ]

================================================================
5. EXACT REDUCTION TO GATED DELTANET -- the structural insurance against Phase 1's failure
================================================================
Set n_a = d_k, remove the top-k (a_t = L2-normalised dense key), H = 2 collapsed to one signed matrix (drop the non-negativity clamp), L_elig = 1 with kappa = [1], G = 1, nu_t = 1, per-channel alpha:
   M_t = (I - beta_t a_t a_t^T)(alpha_t . M_{t-1}) + beta_t a_t v_t^T,   o = M_t^T q
which is EXACTLY Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025, arXiv:2412.06464). CLAW is therefore a STRICT SUPERSET of its own primary baseline. Phase 1's mushroom-body memory lost 0.981 to 0.998 because it was a weaker SPECIAL CASE of Gated DeltaNet (scalar surprise gate, no per-key delta correction). CLAW cannot lose for that reason; it can only fail to gain. Every ablation arm is reached by turning one switch back toward the reduction, which makes the ablation table mechanically well-formed.

================================================================
6. TRAINING FORM: chunked, and it inherits DeltaNet's algebra
================================================================
Chunk the sequence into C=64-token chunks. Within a chunk, form the k-sparse address Gram Gamma_{ij} = <e_i, e_j> ONCE (this is where the shared-address property pays: H compartments reuse one Gram, so heads are nearly free -- exactly the fly's "all compartments read one KC population", which Phase 2's v0.2 got wrong by giving each compartment its own expansion). Then the chunk update is the standard UT transform
   T = (I + tril(diag(beta) Gamma, -1))^{-1} diag(beta),
a C x C triangular solve, identical to DeltaNet/GDN chunking; the only change is that the state update is an index_add_ scatter into k-sparse rows instead of a dense outer product. Log-cumulative decay within the chunk is applied exactly as in GDN's chunked form; cross-chunk lazy decay uses the per-row timestamps. This is ~200 lines of PyTorch with index_add_ / gather and torch.linalg.solve_triangular. A Triton kernel is a later optimisation, not a prerequisite.

================================================================
7. OFFLINE CONSOLIDATION (A13) -- a SEPARATE, second hypothesis, deliberately off the critical path
================================================================
Slow store: a rank-r adapter S_l on each CLAW layer's W_o (r=32), trained by backprop during the pass.
  (7.1)  Pressure  P <- P + sum_h beta_t^{(h)} ||delta_t^{(h)}||    (integrate PLASTICITY, not time)
  (7.2)  Spacing gate, per address i:  c_i <- c_i exp(-Delta_i/tau_c) + 1[recurrence with gap Delta_i in [Delta_min, Delta_max]]
  (7.3)  Recall gate (Lindsey & Litwin-Kumar 2024):  w_i^rec = sigmoid( a cos( r_i^fast , v_i ) + b )   -- promote what the FAST store ALREADY reads correctly, the OPPOSITE of surprise gating
  (7.4)  When P > P*: sample addresses with probability proportional to (stored write tag beta_i) x nov_i x w_i^rec x 1[c_i >= R]; replay them as (key, value) supervision into S with N_pass >= 4 passes; clamp nu_t = 1 (no forgetting) during the pass; reset P.
  The repo's CONSOLIDATION_TOY.md already establishes that naive uniform single-pass replay is worth ~0.04 (0.02 -> 0.06 beyond the fast horizon). The three named fixes -- tag-based SELECTION, spacing, and multi-pass -- are the hypothesis, and they are tested on their own, never bundled into the main claim.

================================================================
8. BUDGET, MEASURED (d=768, n_a=2048, B=32, J=4, k=64, H=8, d_v=64, L_elig=4)
================================================================
  mixer parameters/layer:  CLAW 627,712   vs  GDN ~6d^2 = 3,538,944      (5.6x fewer)
  MACs/token/layer:        CLAW 758,784   vs  GDN ~3,932,160             (5.2x fewer)
  state floats/layer:      CLAW 1,048,576 (2.10 MB fp16)  vs  GDN 98,304 (0.20 MB fp16)   (10.7x MORE -- this is the cost, declared)
  state bandwidth/token/layer: CLAW 196,608 floats  vs  GDN 196,608 floats  (IDENTICAL)
  addressable rows:        2,048 vs 128   (16x)
  MATCHING PROTOCOL for the experiments: match TOTAL model parameters to within 1% by adjusting d_ff per arm, and match per-token FLOPs to within 5% by adjusting (n_a, k, H, d_v); publish the exact table. Resident state memory is the declared, unmatched cost, and is reported for every arm.

### What is new

Eight components, ranked by novelty and with prior art named honestly. Only (1) is the thesis; (2)-(7) are separately-ablatable riders; (8) is a methodological correction.

1. SPARSE-ADDRESSED, ONE-SHOT, TEST-TIME-WRITTEN FAST WEIGHTS. Sparse top-k key-value memory in an LM is NOT new: Product Key Memory (Lample et al., NeurIPS 2019, arXiv:1907.05242), PEER (arXiv:2407.04153), UltraMem (ICLR 2025, arXiv:2411.12364) and Memory Layers at Scale (arXiv:2412.09764, 128B memory params) all work at scale. But in every one of them the memory VALUES are trained by SGD at training time. Conversely, every test-time-written memory (DeltaNet, GDN, KDA, GDN-2, Titans, ATLAS, TTT, LaCT, MesaNet, PGDN) uses a DENSE state. The cell "sparse address x one-shot gradient-free delta write" is empty as far as this survey can establish. CLAW occupies it. That is the new object, and the dedicated control that attacks it is a Memory-Layer arm at the same n_a and k with SGD-trained values.

2. A SIGNED BIPHASIC ELIGIBILITY KERNEL ON THE WRITE ADDRESS (eq. 3.1). Handler et al. (Cell 178:60, 2019) show the sign of the KC->MBON change flips with the ORDER of the odour and the dopamine. No member of the 2024-2026 fast-weight family has any temporal kernel on the key: they all write against the instantaneous key. Cost: one gather of k*L rows shared across all compartments. This is the cheapest new mechanism in the design and the one with the cleanest biological warrant.

3. PAIRED NON-NEGATIVE CHANNELS READ AS A DIFFERENCE (eq. 4.6). Felsenberg et al. (Nature 544:240, 2017; Cell 2018): extinction writes an opposing trace in a different compartment rather than erasing. Gated DeltaNet's single gated matrix physically cannot exhibit spontaneous recovery; CLAW predicts it, which makes it falsifiable by a behavioural signature and not only by a score.

4. WITHIN-LAYER RESIDUAL ALLOCATION ACROSS COMPARTMENTS (eqs. 4.4-4.7). MBON->DAN feedback (339/300 edges in this repo's extraction; Eschbach 2020; Bennett 2021) means each compartment's teaching signal is what the others left. This makes the compartment bank a boosting cascade with LOCAL errors and no shared error budget, versus the parallel independent heads of every linear-attention model. It is also the correct reading of "dopamine = prediction error" -- local and per-module, not the global scalar Phase 2 used.

5. ONE SHARED SPARSE ADDRESS STREAM READ BY ALL COMPARTMENTS. This is an explicit correction of Phase 2's v0.2, which gave each compartment its own expansion (recorded in LIMITATIONS.md) whereas in the fly all compartments read one KC population. It is not merely more faithful: it is what makes H compartments nearly free, because the k-sparse Gram used by the chunked kernel is computed ONCE (section 6).

6. ARCHITECTURAL HETEROGENEITY OVER >= 10 OCTAVES WITH SIGN AND CAPACITY UNTIED FROM DECAY (Aso & Rubin 2016). Multi-timescale memory exists in Nested Learning/HOPE (arXiv:2512.24695) and in MoM; what is new here is that the ladder is FIXED (not learned), spans 2^2-2^14 rather than Phase 2's 4/16/64/256, and beta is untied from tau so that some compartments are fast-and-flexible and others slow-and-stable.

7. RECALL-GATED + SPACING-GATED CONSOLIDATION WITH TAG-WEIGHTED REPLAY (section 7). Titans and ATLAS gate writes on SURPRISE; Lindsey & Litwin-Kumar (eLife 2024) show the fly gates slow writes on successful FAST RECALL -- the opposite signal -- with a provable consequence (learnable timescale linear rather than logarithmic in repetitions). Combined with Tully's spacing law and Dag's tag-selected sleep replay, no LM implements this.

8. PRECISION AS A FEATURE OF A LEARNED GATE, TESTED ONLY OUT OF DISTRIBUTION (eqs. 4.9-4.10). Every 2026 write gate uses an UNNORMALISED error (Titans: raw gradient norm) or a query-conditioned sigmoid (Gated Attention). A Gamma-tracked inverse variance makes the objective Student-t and the update IRLS. The methodological novelty is as important as the mechanism: because a learned gate provably contains any fixed rule as a special case (Phase 1 H1b, -2.6 seed-SD), the ONLY admissible test is generalisation of the gate to non-stationarity statistics absent from training.

WHAT IS DELIBERATELY NOT CLAIMED AS NEW: the delta rule (it is DeltaNet, Schlag et al. ICML 2021), gating and decay (GDN/KDA), multi-timescale state (HOPE/MoM), hybrid 3:1 scheduling (Qwen3-Next, Kimi Linear), divisive normalisation (RMSNorm/softmax are members of the same family), sparse top-k addressing per se (PKM/Memory Layers), negative eigenvalues for state tracking (allow_neg_eigval, one line), and homeostatic renormalisation (LaCT).

### Regime of the claimed advantage

THE REGIME, stated so that it can be refused: **associative recall and retention per bit of per-token compute, under rebinding and beyond the training length, for the constant-state (non-attention) layers of a hybrid decoder at 60M-125M parameters.**

It is NOT: perplexity, perplexity per FLOP, zero-shot commonsense, energy, or anything against full attention. At ~400M the spread across GDN, GSA, GLA, Raven, Mamba-2, FoX, SWA and a RoPE Transformer on zero-shot commonsense is 0.7 points, and the model with the WORST LAMBADA perplexity (FoX, 48.1) has the BEST recall (FDA 64.3 vs GDN 8.3) -- so perplexity cannot support an architecture claim at this scale, and attention-with-a-forget-gate beats every constant-state layer on recall by ~8x. The honest frame is therefore: **a better 75% layer.** The claim is about the hybrid, not about replacing attention.

PRIMARY BENCHMARKS (pre-registered before any run):
 (a) MQAR-with-REBINDING (Zoology, Arora et al., ICLR 2024, arXiv:2312.04927), at P in {64, 128, 256, 512} key-value pairs, T=1024, with a rho=0.25 rebound subset and three query types (new binding / stable / old-binding recovery). NEVER at P=8: Phase 1 H1 tested 8 pairs against a d_key=65 store, five times below the point at which any difference could exist, and returned the null it had to.
 (b) RULER S-NIAH-1/2/3 (Hsieh et al., arXiv:2404.06654) at 1k/2k/4k/8k/16k with TRAINING at 2k -- i.e. length extrapolation to 4-8x.
 (c) Real-world recall, zero-shot on the trained model: SWDE / FDA / SQuAD-completion (Based, Arora et al., ICML 2024, arXiv:2402.18668; Just Read Twice, arXiv:2407.05483).
 (d) A retention-vs-age curve: accuracy on a fact written once at position p, queried at p + 10^2, 10^3, 10^4 tokens. The prediction is power-law-like rather than exponential decay.
 (e) lm-eval zero-shot (ARC-e, ARC-c, HellaSwag, PIQA, WinoGrande, LAMBADA) as a NO-REGRESSION CHECK ONLY. MMLU excluded (below chance at this scale).

THE STRONGEST COMPETITOR, and it must be trained by us, not quoted: **Gated DeltaNet** (Yang, Kautz & Hatamizadeh, ICLR 2025, arXiv:2412.06464), which is the layer that actually ships in Qwen3-Next-80B-A3B (48 layers, full_attention_interval 4) and which already beat this repository's Phase-1 memory 0.998 vs 0.981 (-5.6 seed-SD). Secondary controls, all trained by us on identical data: KDA (Kimi Linear's per-channel-decay variant), DeltaNet, Mamba-2, and a Memory-Layer arm (PKM-style, same n_a and k, SGD-trained values) which specifically attacks novelty item 1. If GDN-2 (decoupled per-channel erase on the key axis and write on the value axis) is reproducible by the time of the run, it replaces GDN as primary.

THE TWO MATCHING ARMS, both reported:
  MATCHED-FLOPs / MATCHED-PARAMS arm -- where the win is claimed.
  MATCHED-STATE-BYTES arm -- where theory (Arora et al.: recall capability is bounded by recurrent state size) says the effect MUST vanish or shrink. Reporting that null is what makes the first claim credible rather than an artefact of counting.

THE CLAIM, quantified. At 125M params, 3:1 hybrid, 8-10B FineWeb-Edu tokens, 2k training context, 3 seeds, matched parameters (1%) and matched per-token FLOPs (5%):
  - MQAR-rebind, stable-recall accuracy at P=256: **>= +0.15 absolute over the GDN twin**, effect >= 2 seed-SD, paired-bootstrap 95% CI excluding 0.
  - MQAR-rebind, old-binding recovery: **>= +0.20** (a signature GDN structurally cannot produce).
  - RULER S-NIAH-2 at 8k (4x training length): **>= +10 points**.
  - SWDE/FDA/SQuAD: positive on >= 2 of 3.
  - lm-eval zero-shot average: **no regression worse than -0.5 points**.
  - Declared cost: **~10x resident state memory** and **<= 1.5x wall-clock per token** (see the go/no-go gate in the ladder).
That would be a defensible new Pareto point on the recall-vs-per-token-compute frontier for constant-state layers at this scale. It is a narrow, matched, pre-registered claim. It is not "a fly brain beats a transformer", and the write-up must say so in the abstract.

### Killer experiment

**MQAR-REBIND AT MATCHED ACTIVE ROWS** -- module level, 4 CPU cores, no language pretraining, ~1 CPU-day total. It converts the repo's existing no-learning numpy result (0.82 vs 0.17 at P=256) into a backprop-trained, learned-gate, learned-address result against the exact baseline that beat Phase 1 by 5.6 seed-SD.

THE STREAM. Vocabulary of keys; T=1024 tokens; P distinct keys each bound once; a random rho=0.25 subset is REBOUND to a fresh value at a random point in the second half; queries interleaved throughout. Four query types, scored by exact match:
  Q1 NEW-BINDING: query a rebound key shortly after rebinding; correct = new value.  (tests the write)
  Q2 STABLE: query a never-rebound key at the end; correct = its value.               (tests CAPACITY/interference -- the PRIMARY metric)
  Q3 RECOVERY: query a rebound key 3*tau_fast tokens later; correct = OLD value.      (tests non-destructive revision)
  Q4 DELAYED CREDIT: variant in which the value token arrives delta in {1,2,3,4} tokens after the key. (tests the biphasic kernel)
All module parameters (address projection, APL thresholds, gates, decays, readout) are trained by backprop on the recall loss, with the learning rate swept per arm, so every learned control is as strong as it can be. 5 seeds. Paired 20,000-sample bootstrap CI; effects in units of the larger seed SD; the repo's existing scripts/build_results.py verdict machinery.

ARMS (identical front-end, identical d_v, identical per-token active rows k=64):
  A  CLAW full: n_a=2048, k=64 (J=4 blocks x 16), H=8 as 4 opposing pairs, L_elig=4, G=2, decay ladder, APL per-block.
  B  Gated DeltaNet with d_k = 64 = k  -- MATCHED per-token FLOPs and matched active rows. **THE CONTROL THAT DECIDES THE THESIS.**
  C  Gated DeltaNet with d_k = 2048 -- MATCHED STATE BYTES (32x the per-token FLOPs). The honesty control.
  D  CLAW minus opposing pairs (one signed matrix per compartment).
  E  CLAW minus biphasic kernel (L_elig = 1).
  F  CLAW minus boosting (G = 1, parallel compartments).
  G  CLAW minus decay ladder (all tau tied and learned).
  H  CLAW minus APL (one global hard k-WTA instead of per-block feedback normalisation).
  I  Dense learned key of dim 64 into a LEARNED top-64-of-2048 expansion (isolates "sparsity" from "a learned expansion").
  J  Memory-Layer control: same n_a, k, but values trained by SGD, not written online.

PRE-REGISTERED THRESHOLDS:
  **T1 -- THE CENTRAL GO/NO-GO.** A minus B on Q2 at P=256 must be **>= +0.15 absolute, >= 2 seed-SD, CI excluding 0.** FALSIFICATION CONDITION: if A - B < +0.05 at EVERY P in {128, 256, 512}, the central thesis -- that sparse addressing converts state into recall more efficiently than a dense state at matched compute -- is FALSIFIED, and Phase 3 stops here at a cost of about one CPU-day and no GPU time. Nothing downstream is attempted. This is the whole point of putting it first: the thesis is killable for essentially zero cost, and by exactly the baseline that has already beaten this project once.
  T2 -- HONESTY CONTROL. Report A vs C. Expectation: C >= A. If A > C by > 2 seed-SD at matched state, the mechanism is NOT state-economics and the framing must be rewritten before anything else runs.
  T3 -- REBINDING. A - D on Q3 >= +0.20, else drop the opposing-pair mechanism from the design and say so.
  T4 -- TIMING. A - E on Q4 at delta >= 2 must be >= +0.10, else drop the biphasic kernel.
  T5 -- BOOSTING. A - F on Q2 at P=512 >= +0.05, else set G=1.
  T6 -- ADDRESS. A vs I within seed spread is ACCEPTABLE (the claim for block routing is COST, not quality). If I beats A by > 2 seed-SD, abandon block routing, learn the full address, and report the cost.
  T7 -- vs TRAINED MEMORY. A vs J on Q1/Q3 (within-sequence binding, which J cannot do at all) establishes that the online write is doing work rather than the sparse geometry alone.

WHY THIS IS THE CHEAPEST POSSIBLE FALSIFIER: it needs no corpus, no tokeniser, no GPU and no pretraining; it isolates the single load-bearing mechanism; it tests it in the regime (P >> d_k) that Phase 1 provably never entered; and its primary control is the same Gated DeltaNet that produced the project's worst prior result, so a pass cannot be dismissed as a weak-baseline artefact.

SECOND, INDEPENDENT GATE (before any LM training, on the 16 GB GPU, ~2 hours): **WALL-CLOCK PARITY.** Benchmark tokens/s of a CLAW layer versus a GDN layer at matched FLOPs, at T in {512, 2048}, batch 8, in the chunked training form and in the single-token decode form. GO only if CLAW is within 1.5x of GDN. FLOP-matching is not wall-clock matching for gather/scatter, and a matched-FLOPs claim that is 5x slower in practice is not a claim the community accepts. If the gate fails, the remedy is larger blocks (m=128-256) and fewer selected blocks, not a bigger J.

### Why phases 1-2 failed and how this differs

WHY PHASE 1 FAILED -- and why its nulls were the BIOLOGICALLY PREDICTED result rather than a disappointment. Phase 1 tested the connectome as STRUCTURE: the whole graph as a recurrent weight mask (real vs degree-preserving shuffle: -0.6/-0.2/-0.4 seed-SD at N=256/512/1024, all CIs spanning 0, and a dense same-N layer beat it everywhere); the real PN->KC wiring as a fixed sparse expansion (-0.1 seed-SD vs shuffle); and the extracted 148-neuron central-complex ring as a state tracker (0.244 vs 0.669 for an unstructured layer, -9.1 seed-SD). Three independent lines of biology say those had to be null. (i) Within ONE animal, left and right copies of the same cell type share only ~52% of their type->type edges (68% for the strongest) -- so a third to a half of the connectome is a developmental sample, not genomic information, and no result can depend on the specific edges. (ii) Caron et al. (Nature 497:113, 2013) established PN->KC convergence is apparently random; Litwin-Kumar et al. (Neuron 93:1153, 2017) showed the measured fan-in is the mathematical optimum -- so the transferable content was always the SCALAR, never the matrix. (iii) The ring is a specialist for one circular variable with a hard-wired 45-degree shift, and it was being simulated with one forward-Euler step per token, which cannot rotate a bump by a meaningful angle; the fly runs that circuit for hundreds of membrane time constants per unit of heading change. PHASE 3 DIFFERS ABSOLUTELY: it loads ZERO adjacency matrices. The connectome contributes only (a) distributional scalars that are then RE-DERIVED from the data rather than copied (coding level f and fan-in are swept by participation ratio, not set to 5% and 6), and (b) a mechanism inventory -- which functional parts exist and what each computes. No wiring-as-prior experiment is re-run, in any form.

WHY PHASE 2 FAILED -- three separate, now-diagnosed, and each one avoidable. (i) THE ITERATED OBJECTIVE WAS QUADRATIC IN THE LATENT, so K gradient steps are provably equivalent to one affine map: the closed-form minimiser s* = sum_l pi_l mu_l / sum_l pi_l makes the Hessian constant. The repo's own DESIGN.md section 1.1 recorded this BEFORE any run. The measured +0.026 bits/char (0.7 seed-SD) was the arithmetic, not a finding. (ii) AROUSAL WAS A MULTIPLICATIVE GAIN UPSTREAM OF A NORM, which is EXACTLY cancelled: RMSNorm(g*x) = RMSNorm(x), verified in this repo to 3e-7 for g in [0.25, 4]; and a global scalar on the loss cancels under any preconditioned optimiser. The experiment compared the model with itself. This is a theorem, not a tuning failure. (iii) THE MUSHROOM-BODY MEMORY WAS A WEAKER SPECIAL CASE OF ITS OWN BASELINE: a scalar surprise gate with per-head expansions and no per-key delta correction is a degenerate Gated DeltaNet, so 0.981 vs 0.998 (-5.6 seed-SD) was structurally guaranteed. Compounding all three: everything ran at 135k parameters on 4 CPU cores, below any scale at which the mechanisms in question are known to pay (Mixture-of-Recursions only matches vanilla transformers above 360M), and the 'wide' control had 1.94x the parameters and 1.32x the FLOPs, so it was not a matched control at all.

HOW PHASE 3 DIFFERS, MECHANISM BY MECHANISM.
 1. NO ITERATION ON A QUADRATIC OBJECTIVE. There is no per-token inference loop anywhere. The only thing that iterates is the two-step APL feedback, which is a fixed-point normaliser, not an inference procedure. A pre-registration gate is added: before any iterative module is ever built, compute d2F/ds2 and delete the module if it does not depend on s.
 2. NO GAIN UPSTREAM OF A NORM. Axiom A6 makes it illegal by construction. Every modulatory scalar in CLAW targets a write gate, a decay gate, an eligibility time constant, a read mix or an output temperature -- the five mathematically non-inert targets. The old arousal scalar returns as a FEATURE (s_t, eq. 1.5) fed forward, which is what Fairhall et al. (2001) actually say the fly does.
 3. THE MEMORY IS A STRICT SUPERSET OF ITS BASELINE, NOT A SPECIAL CASE. Section 5 gives the exact parameter setting at which CLAW IS Gated DeltaNet. It therefore cannot lose for the reason Phase 2's memory lost; it can only fail to gain. Every ablation arm is one switch back toward that reduction, which makes the ablation table mechanically well-formed rather than a collection of unrelated variants.
 4. NO FIXED BIOLOGICAL RULE COMPETING WITH A LEARNED GATE IN-DISTRIBUTION. Phase 1 H1b already settled that contest (-2.6 seed-SD, learned gate wins). Every CLAW gate is learned; the biology contributes the gate's INPUTS (its own compartment's residual, the novelty read, the normaliser's state, the inferred precision), its GRANULARITY (one per compartment, not one global scalar), and its SIZE (far smaller than the module it controls). The one fixed-rule claim that remains -- inferred precision -- is tested ONLY out of distribution, where a learned gate has no training signal.
 5. THE REGIME IS CHOSEN WHERE THE MEASUREMENT HAS DYNAMIC RANGE. Phase 2 staked on bits/char, where the whole architecture spread at 400M is 0.7 points. Phase 3 stakes on recall-under-rebinding and length extrapolation, where the published spread across the same architectures is 0-91 points, and where the effect size the repo has already measured in isolation is 4-10x.
 6. THE TASK IS SIZED WHERE THE EFFECT CAN EXIST. Phase 1 H1 tested 8 key-value pairs against a d_key=65 store -- five times below the dense control's own capacity ceiling -- so no difference was possible and none was found. Phase 3 tests at P in {64,128,256,512}, straddling the ceiling by construction, with the capacity arithmetic (expected k-of-n overlap = k^2/n) computed BEFORE the benchmark is chosen.
 7. THE FALSIFIER COMES FIRST AND IS CHEAP. Phase 2 built a full model and then discovered the null. Phase 3's central thesis is killable in one CPU-day, before a single GPU-hour is spent, by the exact baseline that has already beaten this project.
 8. HONESTY CONTROLS ARE PRE-REGISTERED AS EXPECTED-NULL ARMS. The matched-state-bytes arm is expected to erase the effect (Arora's state-size bound) and is reported either way. Declaring in advance where the effect must NOT appear is what converts a matched-FLOPs win from a counting artefact into a claim.

### Cost

ENGINEERING: one engineer, ~2 months to a Rung-3 result. Breakdown: 2 weeks for the CLAW layer (chunked PyTorch form with index_add_/gather + triangular solve, ~600 lines including the lazy-decay timestamp bookkeeping and the APL quantile update), 1 week for the harness (arm configs, pre-registration, the existing scripts/build_results.py verdict machinery, the diagnostics suite), 3 days for Rung 0/0b/1, then 5-6 weeks of supervised running and writing. A Triton kernel for the sparse scatter is a further ~1-2 weeks and is NOT required before Rung 3.

COMPUTE, rung by rung:
- RUNG 0 (killer experiment, 10 arms x 5 seeds x 4 LRs x 4 values of P): ~8-16 core-hours on 4 CPU cores. **$0, about one day.** This is the entire cost of falsifying the central thesis.
- RUNG 0b (wall-clock parity gate): ~2 GPU-hours on the owned 16 GB card. **$0.**
- RUNG 1 (5M-param end-to-end sanity, character level): ~48 core-hours. **$0, about two days.**
- RUNG 2 (60M params x 3B tokens = 1.08e18 training FLOPs per run; at ~40 TFLOP/s effective bf16 on a 4090-class 16 GB card, ~7.5 h/run; 4 arms x 3 seeds = 12 runs + 4 LR pilots at ~1 h): **~95 GPU-hours, ~5 days wall-clock, $0 if the card is owned** (~$60-120 rented).
- RUNG 3 (125M x 8B tokens = 6.0e18 FLOPs per run, ~42 h/run on the owned card): 6 arms x 3 seeds + 6 ablations x 2 seeds = 30 runs. **~1,260 GPU-hours = ~7-8 weeks on one 16 GB card ($0), or ~380 A100-hours rented at ~3.3x throughput = $600-950 at $1.6-2.5/A100-hour, in ~4 days.** Recommendation: rent this rung; the wall-clock saving is worth more than the money.
- RUNG 3b (consolidation, continual-pretraining stream, 5 arms x 3 seeds at 60M): **~150 GPU-hours, ~$250 rented or 6 days owned.**
- RUNG 4 (350M x 30B tokens = 6.3e19 FLOPs per run, ~440 A100-hours/run; 2 arms x 2 seeds): **~1,800 A100-hours, $3,000-4,500.** Only if Rung 3 passes.
- Evaluation and storage: lm-eval, RULER, BABILong, SWDE/FDA/SQuAD across 30+ checkpoints is ~40 GPU-hours total; datasets (FineWeb-Edu 10B sample, ~25 GB) and checkpoints ~400 GB.

TOTALS. **To falsify the thesis: ~1 CPU-day, $0.** To a defensible, pre-registered, reviewable claim at 125M: **~$600-1,200 of rented compute (or ~9 weeks on one owned 16 GB card) plus ~2 engineer-months.** To a scale-checked claim through 350M: **~$4,000-6,000 plus ~3 engineer-months.** A 1B+ confirmation is out of scope at roughly $25,000-40,000 and is explicitly not part of this plan; the paper must scope its claim to the scales actually run.

### Ladder

1. RUNG 0 -- 4 CPU cores, ~1 day, $0. THE KILLER EXPERIMENT above (MQAR-Rebind at matched active rows, arms A-J, 5 seeds, LR swept per arm), plus the numpy capacity curve re-run with LEARNED addresses to check that the repo's existing 0.82-vs-0.17 result survives learning. GO/NO-GO: T1 must pass (A - B >= +0.15 on Q2 at P=256, >= 2 seed-SD, CI excludes 0). If T1 fails at every P in {128,256,512}, THE PROJECT STOPS and the null is written up as the third pre-registered negative -- a useful one, because it would close the last quantitative fly-derived hypothesis this repository has. T3-T6 individually decide which riders stay in the design; each failed rider is DELETED, not carried forward 'to be tested later at scale'.
2. RUNG 0b -- 16 GB GPU, ~2 hours, $0. WALL-CLOCK PARITY GATE. CLAW layer vs GDN layer tokens/s at matched FLOPs, chunked-training form and single-token decode form, T in {512, 2048}. GO/NO-GO: within 1.5x. Fail => enlarge blocks (m=128-256, J=1-2) and retest; fail twice => the claim is reframed as FLOPs-and-bandwidth-only and the abstract says so, or the project stops.
3. RUNG 1 -- 4 CPU cores, ~2 days, $0. END-TO-END SANITY at 5M parameters, character-level, d=256, L=4 (3 CLAW + 1 SWA), TinyStories or enwik8-small. Not a claim; a health check that the layer TRAINS inside a real LM. Instrumented diagnostics: dead-address fraction, participation ratio of the address code on real tokens (the transferable quantity, per Litwin-Kumar), per-compartment write volume and gate saturation, per-compartment effective tau, novelty-signal dynamic range, gradient norms through the straight-through top-k. GO/NO-GO: dead addresses < 10% without any auxiliary loss (the APL feedback must be doing its job -- top-k SAEs get up to 90% dead latents without mitigation), no gate saturated at 0 or 1 for > 95% of tokens, no NaN, and the Rung-0 MQAR-rebind advantage survives when the addresses are learned end-to-end inside an LM rather than on a synthetic stream.
4. RUNG 2 -- 1x 16 GB GPU, ~120 GPU-hours (~5 days), $0 if owned. THE FIRST REAL COMPARISON. 60M parameters, 3B FineWeb-Edu tokens, 3:1 hybrid, 2k context. Arms: CLAW, GDN (FLOP-matched), GDN (state-matched), DeltaNet -- x 3 seeds = 12 runs at ~7.5 h each, plus a 300M-token LR pilot per arm. Metrics: MQAR-rebind at P in {64,128,256,512}; RULER S-NIAH-1/2/3 at 1k-8k (extrapolation from 2k); retention-vs-age curve; lm-eval no-regression; tokens/s and resident state MB for every arm. GO/NO-GO: CLAW beats the FLOP-matched GDN twin on MQAR-rebind at P >= 256 AND on S-NIAH at >= 4x training length, both by >= 2 seed-SD with CI excluding 0, AND lm-eval average not worse by more than 0.5 points. If the effect measured at Rung 0 does not survive end-to-end training at 60M, STOP -- that is the most likely failure mode and it must not be explained away.
5. RUNG 3 -- 1x 16 GB GPU for ~6 weeks, or ~350 A100-hours rented (~$700-900), OR both. THE CLAIM. 125M parameters, 8-10B FineWeb-Edu tokens, 3:1 hybrid, 2k context, vocab 32k. Full arm set: CLAW, GDN, KDA, DeltaNet, Mamba-2, Memory-Layer control, each x 3 seeds; plus the six single-switch ablations (opposing pairs, biphasic kernel, boosting, decay ladder, APL, novelty gate) x 2 seeds. Add SWDE/FDA/SQuAD, BABILong qa1-qa5 at 4k-32k, and NoLiMa-style lexically-disjoint needles (the benchmark most likely to reward a genuine associative memory rather than lexical matching). Pre-registered in experiments/p3_claw/PREREG.md BEFORE the first run, with the exact thresholds in regime_of_claimed_advantage. GO/NO-GO for publication: the quantified claim is met on >= 3 of the 4 primary metrics with no lm-eval regression, AND at least two of the six ablations individually cross 2 seed-SD (otherwise the paper is 'sparse addressing works', which is a smaller and partly-anticipated claim, and should be written as such).
6. RUNG 3b -- SEPARATE AND OPTIONAL, ~150 GPU-hours. THE CONSOLIDATION HYPOTHESIS, run only after Rung 3 and never bundled into it. Continual pre-training on a domain-sorted single-pass stream; arms: CLAW+recall-gated+spacing-gated tag-weighted replay, vs CLAW+uniform replay, vs CLAW+recency replay, vs no consolidation, vs the incumbent recipe (LR re-warm/re-decay + 5% uniform replay; Ibrahim et al., TMLR 2024, arXiv:2403.08763) -- all at MATCHED total replay tokens and matched consolidation FLOPs. Metric: retained perplexity on held-out old domains and retention-of-a-single-fact at 10^3-10^5 tokens. Threshold: match the 5%-replay incumbent at <= 1% stored replay. The repo's own toy says naive replay is worth ~0.04, so the prior here is poor and it must not contaminate the main claim.
7. RUNG 4 -- RENTED, ~1,800 A100-hours (~$3-5k), only if Rung 3 passes. THE SCALE-FRAGILITY TEST, which is the single largest risk to the whole programme. 350M parameters, 30B tokens, 2 arms (CLAW, GDN) x 2 seeds, context 4k. The specific worry is measured and documented: in the closest published analogue, the quality cost of activation sparsification GREW from 0.64 points at 370M to 2.09 points at 2.7B. GO/NO-GO: the MQAR-rebind and S-NIAH margins at 350M must be >= 60% of the 125M margins. If the margin is shrinking with scale, the paper must SAY SO and the claim must be explicitly scoped to <= 125M -- which is still publishable and still honest, but is not a general architecture claim.

### Risks

- SCALE-FRAGILITY (the largest risk, probability ~0.5 conditional on a 125M win). The closest published analogue shows a sparsification penalty growing from 0.64 points at 370M to 2.09 points at 2.7B. A sparse-address advantage measured at 60-125M could invert by 1B. MITIGATION: Rung 4 is pre-registered as the condition for any unscoped language; the 60M->125M margin SLOPE is reported as a first-class number, not just the endpoints; and if the slope is negative the claim is explicitly scoped by parameter count in the abstract.
- WALL-CLOCK vs FLOPs (probability ~0.35 of failing the first attempt). A k-of-n gather/scatter has poor tensor-core utilisation against a dense 128x128 matmul; the layer could be 5-20x slower at 'matched FLOPs'. MITIGATION: hierarchical BLOCK addressing with contiguous blocks (section 2), the Rung-0b parity gate BEFORE any LM run, and a documented fallback (larger blocks m=128-256, fewer selected J=1-2). Honest position: if parity fails twice, the efficiency framing must be dropped and the claim becomes FLOPs-and-bandwidth only.
- PRIOR-ART COLLISION (probability ~0.3 over a 6-month horizon). Sparse addressed memory is a crowded area (PKM, PEER, UltraMem, Memory Layers at Scale); a test-time-written variant may appear from another group first, or already exist in a paper this survey could not reach (arxiv/OpenReview/journal hosts were proxy-blocked for several of the briefs feeding this design). MITIGATION: the ablation table IS the paper -- each of the six fly-specific mechanisms is separately claimable -- and the literature check is re-run immediately before Rung 3's pre-registration, not after.
- THE LEARNED-GATE OBJECTION RECURS (probability ~0.4 for the precision component specifically). A learned gate with the same inputs contains any fixed rule as a special case and backprop finds it; Phase 1 H1b measured this at -2.6 seed-SD. MITIGATION, structural: every gate in CLAW is ALREADY learned, with the biological quantity as an INPUT FEATURE; the fixed-rule versions exist only as ablation arms; and the precision claim is evaluated ONLY out of distribution (train at drift rate lambda, test at 0.1*lambda and 10*lambda and at unseen noise levels), which is the only regime where the learned gate has no training signal. If the OOD test is null too, the precision component is dropped and reported -- it is a rider, not the thesis.
- DEAD ADDRESSES AND CODE COLLAPSE (probability ~0.3). Top-k SAEs show up to 90% dead latents without mitigation; SDMLP needed an annealed-k 'GABA switch', non-negative weights, normalised inputs and momentum-free SGD to train at all, and still lost to a plain MLP at 10k neurons. MITIGATION: the per-block APL feedback with a running-quantile threshold is precisely the mechanism that prevents this in biology; dead-address fraction and participation ratio are Rung-1 GO/NO-GO diagnostics, not post-hoc observations; a code-usage entropy regulariser is the documented fallback.
- SEQUENTIAL BOOSTING LATENCY (probability ~0.25). G sequential compartment groups add G small kernel launches per chunk. MITIGATION: G is a hyperparameter with G=1 (fully parallel) as both the ablation control and the fallback; T5 decides whether the mechanism earns its latency.
- THE NON-NEGATIVITY CONSTRAINT HURTS. Cornford et al. (2021) show naive sign constraints degrade performance, and Lipshutz et al. (PLOS Comput Biol 19:e1010864, 2023) show a depression-only non-negative compartment reduces to one-dimensional LDA. MITIGATION: the constraint applies only WITHIN an opposing pair, whose difference is unconstrained, so the pair is strictly more expressive than a single signed matrix of the same total size; arm D tests exactly this, and a failure deletes the mechanism cleanly.
- STATISTICAL POWER AT THE AFFORDABLE SCALE. At 60M/3B tokens, seed-SD on lm-eval averages is ~0.5 points and the whole architecture spread at 400M is 0.7 points, so ONLY the recall and extrapolation metrics have enough dynamic range. MITIGATION: no perplexity or commonsense claim is made at any rung; lm-eval is a no-regression check with a one-sided threshold; primary metrics are chosen because their published spread across architectures is 0-91 points, not 0.7.
- CONSOLIDATION IS PROBABLY NULL (probability ~0.65 of no significant effect). The repo's own toy measured naive replay at 0.02 -> 0.06 beyond the fast horizon, and every CL claim at small scale is discounted by reviewers. MITIGATION: it is Rung 3b, explicitly off the critical path, with its own pre-registration and its own threshold against the Ibrahim et al. 5%-replay incumbent.
- THE EFFECT IS REAL BUT UNINTERESTING: CLAW wins on MQAR-rebind and S-NIAH but a hybrid with MORE attention layers beats both at the same cost, making the better-75%-layer framing moot. In the published hybrid ablations, adding full-attention layers buys ~+50 points at 32k on pure GDN -- far more than any layer change. MITIGATION: the 3:1 ratio is held FIXED across all arms and a ratio-sweep arm (2:1 GDN vs 3:1 CLAW at matched FLOPs) is run at Rung 3 so the comparison cannot be won by simply buying more attention.
- OVERALL BASE RATE. Two pre-registered phases from this repository returned null; the research digest that produced this design records roughly thirty named dead ends, several of them for mechanisms that looked at least this promising. Calibrated estimates: P(Rung 0 T1 passes) ~ 0.70 (the numpy effect is 4-10x and the mechanism is a theorem, but a LEARNED dense key may close much of the gap the random dense key left open); P(survives end-to-end at 60M | T1) ~ 0.40; P(the full quantified 125M claim is met | Rung 2) ~ 0.60. Product: **P(a defensible, reviewable, pre-registered SOTA claim in the named regime at <= 125M) ~ 0.17, call it 15-25%.** P(it survives to >= 350M without shrinking) ~ 0.10. P(at least one of the six fly-specific riders yields a separately publishable pre-registered positive) ~ 0.45. P(the project produces a clean, useful, cheap NULL instead) ~ 0.40 -- and at a cost of one CPU-day for the decisive one, that is an acceptable outcome, not a failure.

## Judge 1

| design | biology | mathematics | SOTA plausibility | falsifiability | novelty | cost | total |
|---|---|---|---|---|---|---|---|
| ATTEST (biology-first) — attested fast weights / evidence channel | 8 | 9 | 6 | 10 | 7 | 9 | 49 |
| SIEVE (mathematics-first) — gated cascade, sparse address, selective promotion | 7 | 7 | 5 | 9 | 7 | 7 | 42 |
| CLAW (sota-first) — compartmentalised latent associative writes | 9 | 4 | 5 | 7 | 7 | 6 | 38 |

**ATTEST (biology-first) — attested fast weights / evidence channel** — rationale: The only one of the three that proposes a NEW INPUT rather than a NEW RULE, and the only one that states why that distinction is the whole lesson of phases 1-2. The separation proposition survives scrutiny in its essential part: given |error| alone, novel (12.98) is separable from the rest, but clean (5.82) and contradicted (6.19) overlap, and no function of the instantaneous error — monotone or not — separates the two classes that decide whether to overwrite. The evidence read does (AUROC 0.955). That is a real argument, not an analogy, and it is the correct structural answer to H1b (learned gate beat error gate by 2.6 seed-SD): a learned gate cannot learn what is not in its inputs. Cost discipline is exemplary: +3.1% state, +2.5% FLOPs, +0.03% parameters, which makes the matched-FLOP AND matched-state comparisons nearly free and removes the phase-2 'wide control had 1.94x params' objection by construction. The nested control (three features masked, gamma tied to beta = provably Gated DeltaNet, same code, same kernels, same tuning budget) is the strongest control design in the set. The correctness gate (masked model must reproduce DeltaNet to 0.5 seed-SD BEFORE any hypothesis is tested) preserves phase 2's verification standard while fixing its real problem — phase 2 verified a bad idea correctly. Two candidate mechanisms were self-falsified in ~20 minutes and reported, including a machine-precision equivalence proof; that is the behaviour of a group that has learned from two nulls. Honest calibration (0.55 / 0.25 / 0.12-0.15) matches my own read. Deductions: the separation claim is slightly overstated — with the heterogeneous decay ladder ATTEST itself specifies, writing +v then -v leaves residuals alpha_h^Delta - 1 that differ across heads, so a gate reading ALL head reads could partially recover conflict; the control is fair only because standard GDN gates do not cross-read heads, and a reviewer will press on it. Biological fidelity is high but the fly's alpha'3 signal is depression FROM baseline (saturating, and eroded by the active forgetting of axiom A7), not an unbounded monotone accumulator, so A3 and A7 are in mild tension.

Fatal flaw: Not fatal, but one unflagged gap that must be closed before rung 0 is run: the 15.7x separation and the AUROC 1.000/0.955 numbers were measured on a SPARSE non-negative 32-of-2048 address, where inner products are non-negative and the crosstalk floor rho*sigma with rho = kappa/n is well-posed. The PRIMARY variant, ATTEST-D, uses dense L2-normalised keys, where crosstalk is approximately zero-mean with scale ~sigma/sqrt(d_k) rather than a positive offset — so subtracting a positive floor is the wrong correction, and the one measurement the entire thesis rests on has not been taken in the configuration that will be built. Fix: re-measure the three-way AUROC with dense learned keys before committing, and pre-register the floor estimator (signed noise scale, not positive offset) as a named ablation. Secondary risk, correctly identified by the design itself as modal: natural text may not contain enough contradiction for the mechanism to pay.

**SIEVE (mathematics-first) — gated cascade, sparse address, selective promotion** — rationale: The best scientific instincts of the three. P-A (rank d_k vs rank n at equal gathered rows) is correct and matches the repo's verified numbers. P-B is correctly derived with the geometric-quadrature power-law approximation AND its log-periodic ripple, with the constant self-flagged as needing verification. Most impressively, P-C explicitly concedes that P-B is NOT the discriminator, because per-channel decay (KDA, GDN-2, Mamba-2) already yields sum_i k_i^2 alpha_i^D, i.e. a mixture of exponentials that can also be log-tailed — and then names KDA as arm A7 in its own killer experiment. Voluntarily naming the competitor most likely to erase your effect, and then building the arm, is exactly the behaviour a rigour review rewards; neither other design does it. The load arithmetic for selection (slow-rung interference load P*(p*q_s + (1-p)*q_n) instead of P, with the horizon scaling in the selectivity q_s/q_n) is simple and right in spirit. Two further strengths no other design has: diagnostic D1 — measure on FineWeb whether repeated rare tokens actually reduce loss beyond 2^10 tokens, run FIRST, ~4 CPU-hours — which converts 'the regime may not exist in text' from a post-hoc excuse into a pre-registered gate; and the chunk-union densification (|U| <= C*k = n/8, then run an unmodified dense chunked kernel on the submatrix), which is the best engineering answer in the set to the wall-clock risk and should be reused regardless of which design wins. Deductions on soundness: (i) the design does not reconcile its own two interference mechanisms — at n=16384, k=32 the per-interferer SNR is already n/k ~ 512, so the sparse address has largely solved the interference that selective promotion is introduced to solve, and the residual case (anisotropic LM keys concentrating address collisions) is asserted rather than computed; (ii) the spacing gate's log-Gaussian kernel exp(-(ln Delta - ln Delta*)^2 / 2s^2) is a reasonable but arbitrary parameterisation of a law whose biological support (10 x 15 min) is a single protocol constant.

Fatal flaw: The headline mechanism is a gate FORM, not a new input, and therefore walks into the exact trap phase-1 H1b already sprang. gamma_{h,t} * beta_{h,t}, with both driven by the same feature set, is a reparameterisation of a single learned gate — a learned gate with those inputs contains recall-gating as a special case. The design is partially rescued by an implementation detail it does not flag: beta_{h,t} takes the rung's OWN read r_{h,t}, while gamma_{h,t} takes the FASTER rung's read r_{h-1,t}, so arm A1-vs-A2 does test an input difference. But r_{h-1} and r_h are strongly correlated in a cascade sharing one address, so the margin is thin, and 'recall gate vs surprise gate' is framed as a contest of functional forms — the losing frame. The genuinely defensible new input is the SPACING gate's per-address last-touch timestamps, which no learned gate in any current memory can reconstruct; the design bills it as a rider and the gate form as the thesis. Fix by inverting that: make spacing the thesis, give A2's gate r_{h-1} explicitly so the recall claim is tested as an input difference or dropped.

**CLAW (sota-first) — compartmentalised latent associative writes** — rationale: The richest and best-sourced biology of the three (16 axioms, each with a stated engineering constraint), and it shows the right discipline where anatomy and evidence conflict — it follows H1a and BioHash to a LEARNED projection instead of copying the fixed random expansion, and rejects spiking for the correct reason (APL is graded and non-spiking, so a spiking design would be LESS faithful). The exact-reduction-to-Gated-DeltaNet argument is correct and is the right structural answer to phase 1's 'weaker special case' failure: CLAW cannot lose for that reason, only fail to gain. Its thesis also has the strongest measured prior evidence of any design here — I verified the 0.82/0.17 at P=256 and 0.70/0.07 at P=512 in docs/phase3/MEMORY_CAPACITY.md. But the mathematics is the weakest of the three, with four concrete defects. (1) The FLOP budget is internally inconsistent inside a single section: 'CLAW 758,784 MACs/token/layer' versus 'TOTAL ~307k MAC/token/layer' eight lines later, with the memory term computed at H=7, k=32 against a stated H=8, k=64. For a design whose entire claim is an economics claim, the accounting must be exact. (2) The killer experiment's decisive arm B is GDN at d_k = 64 = k, described as matched, but that matches ACTIVE ROWS, not FLOPs — CLAW's address stage alone (block routing 2,048 + within-block 16,384 = 18.4k MACs) exceeds arm B's entire memory cost, so the comparison the thesis needs is not the one pre-registered. (3) The Gamma/Student-t/IRLS framing is decorative: log pi enters only as a feature of a learned gate, never as an IRLS weight, so no IRLS step is actually taken. (4) The biphasic kernel makes e_t signed, which destroys the non-negativity that the opposing-channel story requires, so two of the eight riders are mutually inconsistent. Falsifiability is good in outline (10 arms, matched-state honesty control expected to WIN, a wall-clock parity gate before any GPU spend, per-rider delete thresholds T3-T7) but six ablations at 2 seeds each is underpowered against seed-SD thresholds, and eight simultaneously-varied mechanisms make a positive result hard to attribute — the exact failure mode the digest records for bio_inspired_nanochat ('built, not shown to help'). Cost: 2 engineer-months for eight mechanisms plus a chunked sparse kernel plus a full harness is optimistic next to SIEVE's honest 8-12 weeks for less, and 40 TFLOP/s effective on a 16 GB card with gather/scatter is an optimistic MFU.

Fatal flaw: Eq. 4.6 is algebraically inert. With M+ and M- sharing the compartment decay alpha_h and the same gate beta, the readout difference updates as (M+ - M-) <- alpha_h(M+ - M-) + beta*e*(relu(delta) - relu(-delta))^T = alpha_h*M + beta*e*delta^T — EXACTLY the signed gated delta rule. The non-negativity clamp never binds, because both channels only ever receive non-negative additions and multiplicative positive decay. So the 'non-destructive revision' rider is a no-op, arm D must be statistically indistinguishable from arm A, and threshold T3 (A - D >= +0.20 on old-binding recovery) is falsified by algebra before any run — while contributing +0.20 of the quantified SOTA claim. ATTEST independently proved this same equivalence to machine precision (6.7e-16) and measured the untied-decay variant losing 0.38 vs 1.00. The deeper issue this exposes is the design's method: eight mechanisms adopted in parallel, at least one provably inert and two mutually inconsistent, means the ablation table — which the design correctly says IS the paper — is not sound enough to carry it.

### Recommendation

BUILD ATTEST, AS THE SPINE, WITH TWO GRAFTED INPUTS AND ONE GRAFTED GATE ORDERED BEFORE IT.

The decisive argument is not that ATTEST's mechanism is the most exciting — it is the least exciting of the three, and its own authors put P(defensible claim) at 0.12-0.15. It is that ATTEST is the only design whose central object survives the specific failure that killed both prior phases, and it is the only one whose falsifier costs ~1.5% of its budget.

Phases 1 and 2 did not fail because the biology was wrong. They failed because every proposed mechanism was a FIXED FUNCTIONAL FORM inserted into a backprop-trained model, and a learned gate with the same inputs contains any fixed form as a special case. H1b measured that directly (-2.6 seed-SD). AIF-H1 measured the same thing for iteration (+0.7 seed-SD on an objective whose Hessian is constant, so K steps equal one affine map). ATTEST is the only design that states this as the generalisable rule — DO NOT PROPOSE A GATE, PROPOSE A STATE — applies it ruthlessly, and kills two of its own candidate mechanisms with it before writing the document. SIEVE's headline (recall-gating) is a gate form and is largely inside its own control's hypothesis class; CLAW's headline riders include one that is provably inert and two that are mutually inconsistent. Only ATTEST proposes something a learned gate demonstrably cannot reconstruct from its inputs.

Second: ATTEST is the only design whose matched comparison is unimpeachable. +3.1% state, +2.5% FLOPs, +0.03% parameters, and a control that is THE SAME CODE with three input features zeroed and one scalar tied — provably Gated DeltaNet, same kernels, same tuning budget. There is no 'the wide control had 1.94x the parameters' argument available to anyone. SIEVE (14.7 MB/layer/sequence) and CLAW (10.7x resident state) both have to defend a large unmatched cost, on a GPU where bandwidth and memory, not FLOPs, are the binding constraint; and the repo's own capacity doc already concedes the sparse advantage 'does not show ... that the advantage survives learned keys and gates', which is precisely the untested step both of them need.

Third: the falsifier. ATTEST's K0 is 6 CPU-hours guarding a 550-1,450 GPU-hour programme, with a correctness gate that runs first, a state-matched arm that pre-answers 'you just added state', and a no-regression check on contradiction-free streams where the mechanism should be inert. A null is as informative as a win — 'a provably available signal that a learned gate cannot exploit' is a real and publishable finding, which is exactly the property phases 1 and 2 lacked, since their nulls were predicted by outside theory and carried almost no information.

Do NOT build CLAW as specified. Its biology is the best in the set and should be mined, but a design with an inconsistent FLOP budget, a mis-specified matched arm, an algebraically inert headline rider and eight simultaneously-varied mechanisms at 2 seeds per ablation will produce an unattributable result whichever way it comes out.

Do NOT build SIEVE as specified either, but do not discard it: extract its spacing gate, its regime-existence diagnostic and its kernel trick (below). Its instincts are the best of the three; its thesis is aimed at the wrong half of its own design.

ORDERING, which matters as much as the choice: run SIEVE's D1 diagnostic FIRST, before anything else, at ~4 CPU-hours. ATTEST's modal failure is 'text may not contain enough contradiction'; SIEVE's D1 answers the same question for a different mechanism. Generalise it: on FineWeb-Edu, measure the joint distribution of (write-query distance, re-binding/contradiction rate) for entities and rare tokens. If there is no mass beyond ~2^10 tokens and no meaningful contradiction rate at 2k context, then ATTEST, SIEVE and CLAW are all aimed at a regime the training corpus does not contain, and the correct action is to change the corpus or the context length — not to spend 600 GPU-hours discovering it afterwards. That single diagnostic is the cheapest risk reduction available anywhere in these three documents, and only one of the three designs contains it.

### Grafts

GRAFT ONTO ATTEST — in priority order, each as a separately pre-registered rider that passes ATTEST's own "new input, not new rule" test:

1. FROM SIEVE, THE REGIME-EXISTENCE DIAGNOSTIC (D1), RUN BEFORE EVERYTHING. Generalised to measure both mechanisms' preconditions at once: on the target corpus, the joint distribution of write-query distance and of contradictory re-binding for entities and rare tokens, via a kNN-LM-style probe. Pre-register it as a hard gate. ~4 CPU-hours; it is the single highest-value item in the whole set.

2. FROM SIEVE, THE SPACING GATE — the strongest idea in any of the three documents. Per-address (or, for dense keys, per-head per-similar-key) last-touch timestamps, promotion weighted by exp(-(ln Delta - ln Delta*)^2 / 2s^2). It is one int32 vector; it is a genuinely NEW INPUT that no learned gate in DeltaNet/GDN/KDA/GDN-2/Titans/ATLAS/MesaNet/PGDN can reconstruct; it has a hard biological constant behind it (Tully 1994: 10 x 15 min gives 7-day LTM, the same trials massed give none) and an independent human replication (Cepeda 2006). It composes with the evidence channel rather than competing: ATTEST answers "how much was I taught here", spacing answers "when", and the two are orthogonal gate inputs. Add it as a second K0 arm — the merged grid costs ~30 CPU-hours total instead of ATTEST's 6 and SIEVE's 25 run separately.

3. FROM CLAW, THE BIPHASIC ELIGIBILITY KERNEL. A short learned causal depthwise conv over the key stream, L ~ 4-8, initialised biphasic with a zero-crossing at tau ~ 1-2 (Handler 2019: the sign of the KC->MBON change flips with the ORDER of odour and dopamine). This is the cheapest genuinely-new mechanism in the entire set, it is an INPUT difference (order information the instantaneous key does not carry), and it is the one thing no member of the 2024-2026 fast-weight family has. Apply it to the CONTENT key only. Do NOT combine it with non-negative opposing channels — a signed kernel breaks the non-negativity those channels require, which is CLAW's own internal contradiction. Test it with CLAW's Q4 delayed-credit arm (value token arrives delta in {1,2,3,4} tokens after the key).

4. FROM CLAW, THE EXACT-REDUCTION DISCIPLINE, AS THE ABLATION PROTOCOL. Write down the precise parameter setting at which the layer IS Gated DeltaNet, and require every ablation arm to be reached by turning ONE switch back toward that reduction. ATTEST already has the nested control; CLAW's formalisation of it as the organising principle of the ablation table is better and should be adopted verbatim. It makes the table mechanically well-formed instead of a bag of variants.

5. FROM SIEVE, THE CHUNK-UNION DENSIFICATION — mandatory if ATTEST-S (sparse addressing) is ever pursued, and reusable for any sparse-memory work in this repo. Within a chunk of C tokens take the union U of touched addresses (|U| <= C*k), gather, rescale rows by alpha^(t - last_touch), densify, run the unmodified chunked gated-delta kernel on the submatrix, scatter back. It converts a sparse memory from FLOP-viable-but-slow into GPU-viable, and it also solves the backward-pass memory problem neither SIEVE nor CLAW costs out (storing only the touched submatrix per chunk boundary rather than the full state).

6. FROM SIEVE, THE APL SERVO AND ITS NECESSITY ARGUMENT. A scalar threshold controller driven by a running-quantile target rather than a fixed k, applied per block, with the argument stated as a prediction: code overlap k^2/n is invariant to monotone rescaling of the score distribution ONLY if the coding level is held, therefore the address fails under distribution shift without it. Plus SIEVE's D2 shift test (inject a mid-stream rescaling and rotation; servo on vs off) as the ablation. This converts "biologically inspired inhibition" into a falsifiable necessary condition. ATTEST-S's adaptive threshold is similar but weaker; take SIEVE's formulation.

7. FROM CLAW, THE FRONT-END STAGING AND THE BUDGET-TABLE FORMAT. Per-channel Weber-Fechner adaptation, then divisive normalisation with a LEARNED exponent nu initialised at 1.5 (a prediction to test, not a constant to copy), then the normaliser's own log-state fed forward as a feature (Fairhall 2001). ATTEST has phi_t already; CLAW's staging is more explicit. And report, for every arm: parameters / MACs / state floats / state BANDWIDTH per token per layer. CLAW's key observation that bandwidth can be matched even when state bytes cannot is the right framing — but recompute the table, because CLAW's own is internally inconsistent (758,784 vs ~307k MACs/token/layer in the same section).

8. FROM CLAW, THE STANDALONE WALL-CLOCK PARITY GATE before any language-model training, and FROM BOTH, the matched-state honesty arm pre-registered as EXPECTED-NULL and reported either way. Declaring in advance where the effect must NOT appear is what turns a matched-FLOP win from a counting artefact into a claim.

DO NOT GRAFT:
- CLAW's opposing non-negative channel pairs. Provably inert with tied decay (M+ - M- update reduces exactly to the signed delta rule; the clamp never binds), and measured to fail badly with untied decay. Anyone reaching for the obvious implementation of Felsenberg 2018 should be pointed at ATTEST's equivalence proof first — that negative result is itself worth a paragraph in the write-up.
- The Gamma/Student-t/IRLS framing as anything more than one gate feature among seven. The repo's own ADAPTIVE_RATE.md shows the gated rate at 0.81 against a tuned fixed rate at 0.80 and an oracle at 0.81 — robustness and tuning-freedom, not accuracy. Test it out of distribution only, or not at all.
- SIEVE's recall gate as a headline. Keep it as one ablation arm, and give the control gate r_{h-1} as an input so the test is about the input, not the parameterisation.
- Consolidation, in any form, on the critical path. All three designs concede the repo's own null (0.02 -> 0.06 beyond the fast horizon, verified). It is a second hypothesis with its own pre-registration or it is nothing.
- Adaptive compute / learned halting as a headline. Phase 2's adaptive-K never fired (K_mean = 3.000 in every run); the literature record for learned halting gates is poor. Keep it as an ablation with a Lagrangian budget and report the gate-firing distribution as a standing diagnostic at every rung — a degenerate gate distribution must be a stop condition, not a post-hoc discovery.

## Judge 2

| design | biology | mathematics | SOTA plausibility | falsifiability | novelty | cost | total |
|---|---|---|---|---|---|---|---|
| ATTEST | 9 | 9 | 6 | 10 | 8 | 9 | 51 |
| SIEVE | 7 | 9 | 7 | 9 | 8 | 8 | 48 |
| CLAW | 8 | 7 | 7 | 9 | 7 | 7 | 45 |

**ATTEST** — rationale: BIOLOGY (9). This is the only design that takes one mushroom-body fact and uses it as load-bearing mechanism rather than as an ingredient list. The fact is real and correctly read: because KC->MBON weights are non-negative and dopamine-driven plasticity only depresses (Hige et al., Neuron 88:985-998, 2015), a compartment's weight vector is a monotone record of the teaching delivered at each KC address, and the alpha'3 compartment reads exactly that accumulation as familiarity, habituating over minutes via KC->MBON depression (Hattori et al., Cell 169:956-969, 2017). The sign mapping is correct too: alpha'3 responds high to novel and low to familiar, matching nov_t = exp(-a_t/mu_t). It gets three things right that phases 1-2 got wrong and that the other designs get partly wrong: ONE shared sparse address read by H cheap per-head value matrices (the repo's own LIMITATIONS.md line 76 records that v0.2 gave each compartment its own expansion; in the fly all compartments read one KC population), modulation confined to plasticity/timescale/threshold and never to gain upstream of a norm (RMSNorm(gx)=RMSNorm(x)), and zero adjacency read anywhere, which also immunises it against the repo's own data defects (all monoaminergic edges stored at weight 0; DPM mis-predicted dopaminergic in docs/CIRCUITS.md). Its section 8 is a genuine contribution to fly-to-ML translation: it proves that the obvious coordinate-sign reading of Felsenberg et al. (Nature 544:240, 2017; Cell 175:709, 2018) -- M+ = relu(r), M- = relu(-r) -- collapses to the gated delta rule to 6.7e-16 with tied decays and biases stored values positive with untied decays, because the fly's opposing compartments hold DIFFERENT memories read with fixed opposite signs, not the +/- coordinates of one value vector. That is correct fly neuroscience and it kills a mechanism the other two designs still carry. Deductions: axiom A2 states alpha'3's 'output is used as a control signal', which overstates Hattori 2017 -- that paper establishes the novelty readout and its dopamine-dependent habituation, not that alpha'3 gates plasticity in other MB compartments. The nearest real support is indirect (Charonitakis et al. 2025, eLife reviewed preprint 106322: dishabituation requires PAM DANs and APL GABAergic input), and should be tagged uncertain. Also the derived scalar cfl_t = a/(||vhat||_1 + a + eps) has no fly correlate; it is an engineering construct presented inside a biological axiom chain. MATH (9). The separation proposition is correct and honestly caveated: for a state S_t = A_t (*) S_{t-1} + k_t g_t^T with g_t any function of the token and of S_{t-1}^T k_t, the histories 'never written at a' and 'written +v then -v at a' leave identical state, reads and gate values at a forever, while an accumulator of |written residual| returns 0 vs ~2 beta|v|. That covers DeltaNet, Gated DeltaNet, KDA, GDN-2, Titans, ATLAS, MesaNet and every learned-gate variant, and the design itself flags that a model with more state could in principle encode it elsewhere. The crosstalk floor rho*sigma with rho = 1/sqrt(d_k) for unit-norm keys or kappa/n for a sparse code is right, and sigma is maintained in O(1), so the correction is free. The measured AUROCs (novel-vs-contradicted 1.000 by evidence vs 0.000 by |error|; clean-vs-contradicted 0.955 vs ~0.5) are reported, not asserted. Soft spot, correctly listed as R4: floor estimation under anisotropic learned LM keys, where the separation measured 15.7x with the floor and 1.3x without. SOTA (6). The smallest lever of the three. +3.1% state, +2.5% memory FLOPs, +0.03% parameters buys no capacity at all -- the entire effect must come from better gating of the same GDN dynamics. Recall wins in this literature come mostly from state size, so a gate improvement plausibly buys points in the contradiction regime and little elsewhere. Against that, the nested control makes any win maximally credible, and the regime (rebinding, stratified recall) is where the effect must exist if it exists. FALSIFIABILITY (10). The best in the set. K0 costs ~6 CPU-hours and five arms differ only in which three gate inputs are unmasked and one tied scalar; a DeltaNet correctness gate runs FIRST and blocks reporting if the masked model does not reproduce the reference; a state-matched arm (A5) pre-answers 'you just added state'; a contradiction-free arm pre-registers no-regression where the mechanism should be inert; and a three-way evidence AUROC > 0.8 is required. It also pre-commits that a null is publishable, because 'a provably-available signal a learned gate cannot exploit' is itself a result. NOVELTY (8, docked from 9). The core -- a per-address accumulator of teaching MAGNITUDE used as a gate input, separating never-taught from taught-contradictorily -- is new and the separation proof is the strongest methodological idea in any of the three designs ('do not propose a gate, propose a state'). But the familiarity head M_fam is prior art the design does not cite: Dasgupta, Sheehan, Stevens & Navlakha (PNAS 2018), the fly Bloom filter, already stores one weight per KC, depressed when that KC fires and passively recovering, with novelty read as the mean weight of the active KCs. It is in this repository's own docs/aif/LITERATURE.md line 1086. The differential claim survives -- a Bloom filter counts occupancy and cannot separate clean-familiar from contradicted-familiar -- but it must be stated relative to Dasgupta 2018, not as though no such object exists. COST (9). Most realistic. ATTEST-D is a subclass of fla's GatedDeltaNet plus one purely linear recurrence (no (I - tril)^-1 term, so it chunks more cheaply than the path it rides on) and a 7->16->4 MLP. Rungs 0, 1 and 1.5 together cost ~36 CPU-hours, ~8 GPU-hours and ~10 engineer-days, and any one can kill the programme before a language-model token is trained; the falsifier is ~1.5% of the total budget.

Fatal flaw: None. The nearest thing is a citation-discipline problem, not a design problem: axiom A2 asserts that MBON-alpha'3's output is used as a control signal over other compartments' plasticity, which Hattori et al. 2017 does not establish (it establishes the novelty readout and its dopamine-dependent habituation). The design does not actually depend on that -- it needs only that the accumulated-teaching state exists at the address and is readable, which is established by the depression-only, non-negative KC->MBON rule -- so the fix is to retag A2 as promising/uncertain and cite Charonitakis et al. 2025 as the indirect support. Second, smaller: the familiarity head duplicates Dasgupta et al., PNAS 2018 uncited. The real risk is the one the design itself names as modal -- backprop may fail to convert a provably available signal into recall -- and that is a risk, priced at 6 CPU-hours, not a flaw.

**SIEVE** — rationale: BIOLOGY (7). The sensory front end is the best-executed translation in the whole set: per-channel Weber-Fechner adaptive gain at tau_a = 64 tokens, then divisive normalisation with a LEARNED exponent nu initialised at 1.5 (Olsen, Bhandawat & Wilson, Neuron 66:287-299, 2010, which compresses a 57x ORN range to 6.6x), then log(eps + sum|u_hat|) fed forward as a feature. That last step is the correct implementation of Fairhall et al. (Nature 412:787-792, 2001) -- whenever you divide by a running statistic you must transmit that statistic -- and it is the precise, principled repair of phase 2's inert arousal unit, which used surprise as a gain rather than as a feature. The spacing biology is solid and correctly quoted (Tully, Preat, Boynton & Del Vecchio, Cell 79:35-47, 1994: 10 sessions with 15-min rests give 7-day protein-synthesis-dependent memory, the same trials massed give none), and the ecological argument -- an insect's day is overwhelmingly one-off experience punctuated by a few associations worth a week, which is also the structure of text -- is the best regime argument any of the three makes. But three fidelity problems. (i) The APL servo is a SINGLE global scalar threshold. Amin et al. (eLife 9:e56954, 2020) measured APL's electrotonic space constant at ~50 um and showed it behaves as at least four independent local normalisers, with distance-weighted connectivity making each KC inhibit itself ~40% more than it inhibits another; Prisco et al. (eLife 13:e74172, 2024) show calyx normalisation applied per PN bouton. A single global sparsity scalar is listed as an explicit dead end in the digest, and both other designs get this right where SIEVE does not. (ii) The spacing gate is per-address, via last-touch timestamps ell[i] and a log-Gaussian kernel. The fly's spacing integrators are molecular and compartment-level, not per-address: cytosolic PKC-delta over 30-120 min, mitochondrial over 3-4.5 h, sustained MP1/PPL1-gamma1pedc calcium for ~2 h after spaced training (Comyn, Preat, Pavlowsky & Placais 2024). The fly does not track when a particular KC ensemble was last taught. (iii) Deepest: the promotion cascade is a complementary-learning-systems architecture in mushroom-body clothing. SIEVE's own axiom A4 states the biology correctly -- compartments are PARALLEL readouts of one shared code with fixed heterogeneous decay, each written by its own DANs (Aso & Rubin, eLife 5:e16135, 2016) -- and then the architecture builds a sequential chain in which rung h+1 is fed by rung h's success. Recall-gating itself is a normative theory (Lindsey & Litwin-Kumar, eLife 13:e90793, 2024), and the closest real fly evidence for a readout reinforcing the DAN that writes durable memory is the MBON-alpha1 -> PAM-alpha1 positive-feedback loop over a ~1 h post-training window (Ichinose et al., eLife 4:e10719, 2015), which SIEVE does not cite. MATH (9). Proposition P-A is a theorem and correct: a delta-written dense state is rank <= d_k at cost Theta(d_k d_v), a k-of-n state has rank up to n at cost Theta(k d_v), so exact capacity per unit compute is n/k rather than 1 -- and the repo's numpy study confirms it (0.82 vs 0.17 at P=256; 0.70 vs 0.07 at P=512). P-B's geometric quadrature of t^-gamma with log-periodic ripple 2|Gamma(gamma + 2 pi i / ln c)|/Gamma(gamma) is the standard Mellin result and is explicitly flagged 'verify the constant'. The outstanding move is P-C's preamble, where the design states against its own interest that P-B alone does NOT separate it from per-channel decay, because KDA/GDN-2/Mamba-2 already give sum_i k_i^2 alpha_i^D, and then narrows the claim to the one thing per-channel decay cannot do: decide after the fact WHICH associations occupy the slow modes. The load argument (slow-rung interference P*(p q_s + (1-p) q_n) instead of P, horizon scaling with selectivity q_s/q_n) is sound but informal -- an argument, not a proof, and correctly not called one. SOTA (7). A bigger lever than ATTEST: 16,384 addresses against a dense d_k of 128, aimed at the axis where constant-state models visibly and quantitatively fail (Gated DeltaNet scores FDA 8.3 against a forgetting Transformer's 64.3 at ~400M). Against it: 14.7 MB of resident state per layer per sequence, declared honestly but a serious deployment objection, and the matched-state arm is pre-registered as expected-null. FALSIFIABILITY (9). Seven arms with an unambiguous stop condition ('if A1 is within 1 seed-SD of A2 at every chaff rate, the design collapses to a sparse multi-timescale memory and does not deserve a GPU'), plus two further named informative outcomes, plus arm A7 (per-channel decay) deliberately included as the thing most likely to kill it. And D1 -- measuring on FineWeb-Edu, with a kNN-LM-style probe, whether repeated rare tokens actually reduce loss beyond 2^10 tokens, run FIRST for ~4 CPU-hours -- is the single best methodological idea in the three designs: it tests whether the biological ecology maps onto text at all before anything is spent. Docked one point only because A2's learned per-rung rate can approximate much of the recall test from its own inputs, so the separation is empirical rather than proven, which is exactly the gap ATTEST closes. NOVELTY (8). Putting a TEST on the write before routing it to a longer-lived store appears unoccupied in LM memory, and the spacing gate has no counterpart in a literature that gates uniformly on magnitude. But recall-gating is Lindsey 2024, spacing is Tully 1994, the cascade framing is CLS, and the three-way decoupling is PKM/Memory Layers plus HOPE recombined. COST (8). The arithmetic is the most carefully done of the three (60M x 3B = 1.08e18 FLOPs, ~5 h/run, 24 runs = ~120 GPU-hours; 125M x 8B = ~500 GPU-hours), the falsifier is ~25 CPU-hours guarding ~700 GPU-hours, and the chunk-union densification -- take the union U of addresses touched in a chunk, |U| <= C*k = n/8, rescale rows by alpha^(t - ell_i), run the unmodified dense chunked kernel on the submatrix, scatter back -- is the concrete engineering insight that makes a sparse delta memory trainable. 8-12 weeks for one engineer is honest but tight given the servo, the timestamps and the decay renormalisation.

Fatal flaw: Not fatal, but load-bearing and unacknowledged: the promotion cascade is a complementary-learning-systems architecture attributed to the mushroom body. SIEVE's own axiom A4 states that MB compartments are parallel readouts of one shared code with fixed heterogeneous decay, each written by its own DANs; the architecture then builds a sequential chain in which each rung's write is gated by the previous rung's readout. The fly has cross-compartment MBON->DAN feedback and a real acquisition-to-durable shift across compartments, so a cascade reading is defensible, but it is a substantially stronger claim than the axioms license and the design does not flag the tension. Recall-gating is a normative theory (Lindsey & Litwin-Kumar 2024) whose closest fly support -- the MBON-alpha1 -> PAM-alpha1 loop over ~1 h post-training (Ichinose et al., eLife 4:e10719, 2015) -- is not cited. Secondary and cleanly fixable: the single global APL threshold contradicts measured APL locality (Amin et al., eLife 9:e56954, 2020), which the digest lists as a dead end; and per-address spacing timestamps are not what the fly's compartment-level molecular spacing integrators do.

**CLAW** — rationale: BIOLOGY (8). The broadest and mostly accurate inventory of the three, and it gets several things right that the others miss. Per-BLOCK APL feedback with a running-quantile threshold correctly implements the measured locality of APL inhibition (Amin et al., eLife 9:e56954, 2020; Prisco et al., eLife 13:e74172, 2024) where SIEVE uses one global scalar. The biphasic eligibility kernel over the last L=4 address sets, initialised as a difference of exponentials with a zero-crossing near tau ~ 1-2, is the best-sourced single fly mechanism anywhere in these three documents: Handler et al. (Cell 178:60-75, 2019) show the sign of the KC->MBON change flips with the ORDER of the odour and the dopamine, through Dop1R1-Gs-cAMP depression and Dop1R2-Gq-Ca2+ potentiation with opposite second messengers, corroborated by Tanimoto et al. (Nature 430:983, 2004) on backward-pairing relief memory. Nothing in DeltaNet, Gated DeltaNet, KDA, GDN-2, Titans, ATLAS, LaCT, MesaNet or PGDN has any temporal kernel on the key. The design's decision to LEARN the expansion rather than freeze a random one, stated as 'the evidence overrules the anatomy' and backed by the repo's own H1a (dense learned projection +0.0413, CI includes 0; k-WTA +0.0666, +1.1 seed-SD) plus BioHash (Ryali, Hopfield, Grinberg & Krotov, ICML 2020), is exactly the right scientific call -- the fly's randomness is a developmental constraint, since a ~13,000-gene genome cannot specify 10^4 synapses, not an optimum. The decay ladder spans 2^2 to 2^14, which matches the real 10^3-10^5 range of memory phases rather than phase 2's 64x. No spikes, no adjacency, modulation on plasticity only, precision demoted to an OOD-only rider: all correct. Deductions: the opposing-pair write is wrong (see fatal flaw); the novelty compartment M_fam is the Dasgupta, Sheehan, Stevens & Navlakha fly Bloom filter (PNAS 2018), which is in this repository's own docs/aif/LITERATURE.md line 1086, presented here as new; and 'novelty gates writes in other compartments' carries the same Hattori-overstatement as ATTEST. The structural problem is that this is a stack of eight mechanisms, which is the exact failure mode the digest names ('bio-inspired mechanism stacking... built, not shown to help'). CLAW mitigates it with per-mechanism thresholds T3-T7 -- real discipline -- but the ablation space for 6 interacting switches is 2^6, not 6, and the headline thesis is the least fly-specific component while the strongest fly mechanism is a rider. MATH (7). Section 5 is excellent and is the structural insurance phase 1 lacked: setting n_a = d_k, dropping top-k, collapsing H to one signed matrix, L_elig = 1, G = 1, nu = 1 recovers exactly M_t = (I - beta_t a_t a_t^T)(alpha_t . M_{t-1}) + beta_t a_t v_t^T, i.e. Gated DeltaNet, so CLAW is a strict superset of its own baseline and cannot lose the way the phase-1 memory lost (0.981 vs 0.998 because it was a weaker special case of its own control). The k^2/n overlap arithmetic, the chunked UT-transform reuse with a single shared k-sparse Gram across compartments, and the per-layer budget table (627,712 mixer params and 758,784 MACs against GDN's 3.54M and 3.93M, at 10.7x the resident state but identical 196,608-float per-token bandwidth) are all correctly computed. Docked to 7 because eq. 4.6 is mathematically the construction ATTEST proved degenerate, and because the sequential boosting across G groups adds a non-collapsible serial dependency whose benefit is asserted rather than derived. SOTA (7). The biggest lever and the strongest measured prior -- the repo's own 0.82 vs 0.17 at P=256 at matched active rows. But the differential claim against prior art narrows sharply once Product Key Memory (Lample et al., NeurIPS 2019), PEER, UltraMem and Memory Layers at Scale (arXiv:2412.09764, 128B memory parameters) are accounted for: what is actually unoccupied is 'sparse address x one-shot gradient-free delta write', which is real, but it is an ML observation, not a fly one. CLAW says so, which is honest, and includes the Memory-Layer control arm J that attacks it directly. FALSIFIABILITY (9). Ten arms, seven named thresholds, an ablation-by-single-switch structure made well-formed by the exact GDN reduction, and a separate wall-clock parity gate (rung 0b, 1.5x, before any LM run) that correctly recognises FLOP-matching is not wall-clock matching for gather/scatter. Docked one point for convention-mixing -- some thresholds are absolute accuracy deltas (+0.15, +0.20, +0.10) and some are seed-SD-normalised -- and for a larger forking-paths surface (7 thresholds x 4 values of P x multiple arms) than ATTEST's. NOVELTY (7). The biphasic kernel is genuinely new to ML and well sourced; the sparse-address-with-online-write cell is genuinely unoccupied. Against that: the headline is adjacent to a crowded literature CLAW itself names, the novelty compartment is uncited 2018 prior art, and the opposing-pair mechanism repeats an error already falsified inside this project's own design process. COST (7). The compute arithmetic is sound (60M x 3B = 1.08e18 FLOPs, ~7.5 h/run, ~95 GPU-hours at rung 2; 125M x 8B = 6.0e18, ~42 h/run, 30 runs = ~1,260 GPU-hours or ~380 rented A100-hours at $600-950), and the 1-CPU-day falsifier is the right shape. But ~600 lines and two weeks for block routing, per-block quantile servos, an L=4 eligibility ring buffer, lazy per-row decay with timestamps, sequential boosting groups, non-negativity clamps on paired channels, and a triangular solve over k-sparse Grams is optimistic by roughly a factor of two, and eight interacting mechanisms make debugging and attribution materially harder than ATTEST's single added recurrence.

Fatal flaw: Not unrecoverable, but it is a mechanism this project has already measured to fail: eq. 4.6 implements Felsenberg's non-destructive revision by splitting ONE value correction coordinate-wise into relu(delta) and relu(-delta) and routing them to M+ and M-. ATTEST's section 8 proves this construction reduces exactly to the gated delta rule when the decays are tied (max|(M+ - M-) - M_gdn| = 6.7e-16) and, when untied, biases every stored value positive, measured at 0.38 vs 1.00 reversion accuracy. The biology is being misread: the fly's opposing compartments hold DIFFERENT memories -- an extinction trace in one compartment, the acquisition trace in another -- read out with fixed opposite signs (Felsenberg et al., Nature 544:240, 2017; Cell 175:709, 2018), not the positive and negative coordinates of a single value vector. CLAW's own axiom A9 states this correctly; the implementation does not follow it. Arm D will catch it at rung 0, so the cost is bounded, but it should be replaced before the run rather than tested. The broader structural risk: the load-bearing thesis (sparse addressing) is the least fly-specific element, so the most likely outcome is 'an online-written sparse memory layer works, and none of the six fly-specific riders individually cleared 2 seed-SD' -- an acceptable ML result, but a fourth null on the question this project actually asks.

### Recommendation

RECOMMEND: ATTEST (total 51), with four grafts from SIEVE and CLAW specified below.

The decisive reason is biological, not tactical. ATTEST is the only one of the three that identifies a mushroom-body PROPERTY rather than a mushroom-body PART LIST, and then builds the minimum object that has that property. The property is: because dopamine-driven plasticity at KC->MBON only depresses and the weights are non-negative (Hige et al., Neuron 88:985-998, 2015; normative derivation as online LDA in Lipshutz et al., PLOS Comput Biol 19:e1010864, 2023), a compartment's weight vector is a monotone physical record of the teaching delivered at each address, so 'never taught here' and 'taught contradictorily here' are physically distinguishable states of the circuit -- and the fly devotes a compartment, alpha'3, to reading that record (Hattori et al., Cell 169:956-969, 2017). That is a real fact about the fly that has a real consequence for the machine-learning object: every 2024-2026 fast-weight layer keeps one signed matrix whose read at a contradicted address is identical to its read at a never-written address, forever.

This also solves the methodological problem that killed both previous phases, and it solves it structurally rather than by trying harder. Phase 1's H1b measured a learned gate beating a hand-specified error gate by 2.6 seed-SD (0.9966 vs 0.9814), and phase 2's K=3 free-energy steps beat K=0 by 0.7 seed-SD against a control with 1.94x the parameters. Both failed for the same reason: a fixed rule lies inside a learned gate's hypothesis class, so backpropagation finds it and the biology is redundant at best. ATTEST's rule -- 'do not propose a gate, propose a state' -- is the correct generalisation of those two nulls, and it is the one contribution here that would survive even if the model itself fails. The design applies that rule to itself ruthlessly: it falsified two of its own candidate fly mechanisms in about twenty minutes before writing anything, including the proof that coordinate-sign opponent channels ARE the gated delta rule to 6.7e-16, which is a genuine piece of fly-to-ML translation that CLAW still needs.

Second reason: the control is nested and exact. The baseline is the same code with three input features masked and gamma tied to beta, which is provably Gated DeltaNet at identical parameter count, identical kernels and identical tuning budget. There is no reimplementation asymmetry to argue about, which is precisely the criticism that made phase 2's result uninterpretable in both directions. And the delta is small enough (+3.1% state, +2.5% memory FLOPs, +0.03% parameters) that matched-FLOP, matched-parameter and matched-state arms are all cheap, so there is no excuse for an unmatched control anywhere in the programme.

Third: the falsifier costs ~6 CPU-hours and runs before any GPU time, and a null is informative rather than merely disappointing -- 'here is a provably available signal that a learned gate cannot exploit' is a publishable statement about test-time-learning memories, which is more than either prior phase's nulls delivered.

Why not SIEVE (48): its front end is the best in the set and its D1 corpus diagnostic is the best single methodological idea anywhere in these three documents, but its central architecture is a complementary-learning-systems cascade attributed to the mushroom body, in tension with its own axiom A4, and its separation from the control is an argument about interference load rather than a proof. It also gets APL wrong (one global scalar against measured ~50 um locality and >=4 independent regions) and makes the spacing integrator per-address when the fly's is molecular and compartment-level. It is the right second design and several of its parts belong in the winner.

Why not CLAW (45): it has the broadest and largely accurate biology, and the biphasic write kernel is the best-sourced fly mechanism in the set. But it is a stack of eight mechanisms whose headline claim -- sparse addressing -- is the least fly-specific of them, sitting in Product Key Memory / Memory Layers territory by its own admission, while the genuinely fly-derived parts are riders with individually low priors. Its opposing-pair implementation is the construction ATTEST already falsified. The expected outcome is a modest ML result plus a fourth null on the fly question, which is the wrong shape for this project.

CONDITIONS ON THE RECOMMENDATION -- three corrections to make before writing PREREG.md:
(1) Retag axiom A2. Hattori et al. 2017 establishes MBON-alpha'3 as a novelty readout with dopamine-dependent habituation; it does NOT establish that alpha'3 gates plasticity in other MB compartments. Move that clause to 'promising', and cite Charonitakis et al. (2025, eLife reviewed preprint 106322) as the indirect support -- dishabituation requires PAM DANs and APL GABAergic input, and heterosensory dishabituation requires PPL1 DANs to gamma KCs -- while noting it is moderate-confidence and is about behavioural dishabituation, not about write-gating. The thesis does not depend on this clause; only the adaptive-compute rider (4a) does.
(2) Cite Dasgupta, Sheehan, Stevens & Navlakha (PNAS 2018), the fly Bloom filter, for the familiarity head. It is already in this repository's docs/aif/LITERATURE.md line 1086: one weight per KC, depressed when the KC fires, passively recovering, novelty read as the mean weight of the active KCs. That is M_fam. The differential claim survives intact and must be stated as such: a Bloom filter counts OCCUPANCY and therefore cannot separate clean-familiar from contradicted-familiar, which is exactly what the d_z-dimensional teaching-magnitude accumulator does (measured clean-vs-contradicted AUROC 0.955). Claiming M_fam as new would be an avoidable and checkable citation error.
(3) Drop 'evidence' as a load-bearing word in the write-up and keep the anatomy explicit. In the fly this is a separate compartment's synapses onto a separate MBON, depressed by its own DAN, reading the same KC population -- which is what a separate Z matrix over the shared address actually is. Say that, so a referee can check it against Aso et al. (eLife 3:e04577, 2014) and the repo's own extraction (2,019/2,045 KCs, 48/49 MBONs, 158 PAM + 8 PPL1, one APL covering 100%/99.95% of KCs).

### Grafts

GRAFT FOUR THINGS ONTO ATTEST. Each is chosen because it survives ATTEST's own admission test -- it adds a STATE or a MEASUREMENT, not a hand-specified rule that a learned gate already contains.

GRAFT 1, HIGHEST VALUE -- CLAW's biphasic eligibility kernel (CLAW architecture sec. 3). This is the best-sourced fly mechanism in any of the three designs and ATTEST omits it entirely. Handler et al. (Cell 178:60-75, 2019) show the sign of the KC->MBON weight change flips with the ORDER of the odour and the dopamine, via Dop1R1-Gs-cAMP depression and Dop1R2-Gq-Ca2+ potentiation -- two receptors with opposite second messengers on one transmitter -- with the crossover near zero offset; Tanimoto et al. (Nature 430:983, 2004) corroborate with backward-pairing relief memory; Cohn et al. (2015) add unpaired potentiation. Nothing in DeltaNet, Gated DeltaNet, KDA, GDN-2, Titans, ATLAS, LaCT, MesaNet or PGDN has ANY temporal kernel on the key: they all write against the instantaneous key. Crucially this graft is legal under ATTEST's own doctrine, because a short causal history of the address stream is a STATE the control cannot reconstruct from its inputs, not a gate form the control contains. Implement exactly as CLAW does: e_t = sum_{tau=0}^{L-1} kappa_tau a_{t-tau}, renormalised, L = 4, kappa initialised biphasic as c1 exp(-tau/tau_1) - c2 exp(-tau/tau_2) with tau_1 = 0.8, tau_2 = 3.0, learned thereafter; one gather of k*L rows serves all heads because they share the address; it reduces exactly to the instantaneous key at L = 1, kappa = [1], which is a second nested control. Cost is a ring buffer of ~256 entries. Add CLAW's Q4 DELAYED-CREDIT query type to K0 -- the value token arrives delta in {1,2,3,4} tokens after the key -- with CLAW's threshold T4 (>= +0.10 at delta >= 2) as its own pre-registered rider. Flag honestly that the kernel width in the fly is seconds and UNCERTAIN (the digest could not retrieve it from the Handler abstract), so the token-scale initialisation is a guess that is then learned.

GRAFT 2 -- SIEVE's D1 corpus diagnostic, retargeted, run at rung 0 beside K0 for ~4 CPU-hours. SIEVE's best idea is to ask whether the corpus actually contains the structure the mechanism needs BEFORE spending anything. ATTEST names exactly this as its transfer risk ('contradiction may be rare in natural text') but provides no cheap test, so a rung-2 null would be uninterpretable. Build the ATTEST analogue: on FineWeb-Edu at 2k and 8k context, measure the REBINDING AND CONTRADICTION RATE -- how often a repeated key-like unit (entity, name, variable, definition) is followed within the window by a conflicting value, and at what gap distribution -- using a kNN-LM-style probe (Khandelwal et al., ICLR 2020, arXiv:1911.00172) over a small pretrained model. If the contradicted-address density is negligible at 2k, either move the language rung to 8k-32k context where it is not, or scope the claim to synthetic and code/agent streams and say so in the abstract. This converts ATTEST's most likely failure (regime absent) from a mysterious null into a measured boundary, which is the whole point of the pivot.

GRAFT 3 -- SIEVE's front-end normalisation stack, replacing ATTEST's single phi_t. ATTEST honours Fairhall et al. (Nature 412:787-792, 2001) with one scalar; SIEVE implements it properly and ATTEST needs the stronger version for a specific technical reason. Take SIEVE sec. 1 verbatim: (a) per-channel Weber-Fechner adaptation a_t = rho a_{t-1} + (1-rho)|u_t| with tau_a = 64 tokens (the ORN tau ~ 250 ms arm), (b) divisive normalisation with a LEARNED exponent nu initialised at 1.5 and learned sigma, kappa (Olsen, Bhandawat & Wilson, Neuron 66:287-299, 2010, which compresses a 57x ORN range to 6.6x -- and note the design makes nu a testable prediction rather than a copied constant, since no ML work has swept non-quadratic pooling exponents), (c) s_t = log(eps + sum_j |u_hat_j|) fed forward as a feature into every gate. The technical reason this matters for ATTEST specifically: its crosstalk floor rho*sigma assumes near-isotropic addresses, and its own risk R4 is that anisotropic learned LM keys break the estimate -- separation measured at 15.7x with a good floor and 1.3x without. Whitening and normalising before the address stage is the mechanism that keeps the floor estimable, and the only preregistered attempt at fixed sparse hashing on un-whitened LM states in the digest hard-failed at ~100k keys. Pre-register the front-end ablation as a possible cause of a null.

GRAFT 4 -- CLAW's per-block APL with hierarchical block routing, IF and ONLY IF ATTEST-S is pursued after rung 1. ATTEST-S already specifies per-block adaptive thresholds, which is right; take CLAW sec. 2's concrete form because it solves ATTEST's rung-1.5 throughput gate. Score B contiguous blocks first (cost B*d_pn), take the top J, score only within those (cost J*m*d_pn), then run two iterations of graded per-block feedback inhibition g^(b) = (1/m) sum relu(z_i); z_i <- relu(z_i - lambda_b g^(b) - theta_b) with theta_b driven to a running (1-f) quantile. At B = 32, J = 4, m = 64, d_pn = 64 that is 18,432 MACs, 28x cheaper than dense scoring, and the selected rows are CONTIGUOUS so the gathers coalesce -- which is the difference between passing and failing a 0.5x-of-GDN wall-clock gate. This is also the biologically correct form: Amin et al. (eLife 9:e56954, 2020) measured APL's ~50 um space constant and >= 4 independent local regions, and Prisco et al. (eLife 13:e74172, 2024) show calyx normalisation per PN bouton, so local pools are the anatomy and one global k-WTA is not. Derive f and the fan-in by maximising the participation ratio (tr C)^2/tr(C^2) on real token data (Litwin-Kumar et al., Neuron 93:1153-1164, 2017) and report the measured PR as the transferable diagnostic; do not copy K = 6 or f = 5%, and note that the right coding level for next-token prediction versus episodic key storage is unmeasured (Xie et al., eLife 12:e82914, 2023), so it must be swept in [0.01, 0.20].

ALSO ADOPT, CHEAPLY: (i) CLAW sec. 5's exact reduction to Gated DeltaNet, printed as an equation with the parameter setting that recovers it. ATTEST has this operationally (mask three features, tie gamma to beta) but CLAW writes it as a theorem, and publishing it that way is the structural insurance against the phase-1 criticism that the fly memory was a weaker special case of its own baseline. (ii) SIEVE's habit of declaring in advance where the effect MUST NOT appear -- ATTEST has arm A5 (state-matched) and the contradiction-free no-regression arm; keep both and frame them explicitly as expected-null honesty controls, since that is what converts a matched-compute win from a counting artefact into a claim. (iii) CLAW's wider decay ladder: ATTEST's tau_h in {8,...,8192} is 10 octaves and fine, but untie the write capacity beta from tau across heads, per Aso & Rubin (eLife 5:e16135, 2016) -- some compartments fast-and-flexible, some slow-and-stable, with the heterogeneity architectural and not learned.

DO NOT GRAFT -- four things, each for a stated reason:
(a) CLAW's opposing-pair write (eq. 4.6). It is the coordinate-sign construction ATTEST already proved reduces to the gated delta rule at 6.7e-16 with tied decays and biases stored values positive with untied decays. The fly's opposing compartments hold different MEMORIES read with fixed opposite signs (Felsenberg et al., Nature 544:240, 2017; Cell 175:709, 2018), not the +/- coordinates of one value vector. If non-destructive revision is wanted later, implement it as ATTEST does -- a separate compartment holding a separate trace with its own decay -- not as a coordinate split.
(b) SIEVE's single global APL threshold. Contradicted by Amin 2020; use graft 4 instead.
(c) SIEVE's per-address spacing timestamps as a PRIMARY mechanism. The fly's spacing integrators are molecular and compartment-level -- cytosolic PKC-delta over 30-120 min, mitochondrial over 3-4.5 h, sustained MP1/PPL1-gamma1pedc calcium ~2 h after spaced training (Comyn, Preat, Pavlowsky & Placais 2024) -- and Tully et al. (Cell 79:35-47, 1994) is a statement about training SESSIONS, not about individual addresses. Keep spacing only inside the off-critical-path consolidation arm, where both ATTEST (4c) and SIEVE already put it, and expect a null there: this repo's own CONSOLIDATION_TOY.md measured naive replay at 0.06 vs 0.02 retention beyond the fast horizon.
(d) CLAW's sequential boosting groups (G = 2), until there is T5-style evidence. MBON->DAN feedback is real anatomy (339/300 edges, 2,861/2,605 contacts in this repo's extraction; Eschbach et al., Nat Neurosci 23:544-555, 2020; Bennett et al., Nat Commun 2021), but 'each head writes the residual the others left' is a boosting interpretation of it, not a measurement, and it buys a non-collapsible serial dependency inside the chunked kernel. Keep ATTEST's parallel heads; if it is ever tested, test it as a rider with CLAW's own threshold (>= +0.05 at P = 512) and G = 1 as both the ablation and the fallback.

ONE CROSS-CUTTING NOTE FOR THE NEXT DESIGNER. All three designs, and this repository's whole phase-3 reading, treat MBON-alpha'3 novelty as a control signal over other compartments' plasticity. Hattori et al. 2017 does not show that. The honest formulation is: the novelty state exists, is a physical consequence of depression-only non-negative KC->MBON plasticity, and is read by a dedicated compartment; whether the fly uses it to gate plasticity elsewhere in the MB is open, with the nearest support being dishabituation (Charonitakis et al. 2025) and the APL/DPM coupling. Use the state, claim the state, and mark the gating as the hypothesis under test -- which is, conveniently, exactly what the killer experiment measures.

## Red team (on the leader)

### Verdict

DO NOT BUILD ATTEST-D. The thesis is not dead, but the design as written is, and one variant of it is falsified now rather than in six CPU-hours.

THE CORE FINDING. The design rests on one claim — that a per-address accumulator of teaching is a signal no shipped architecture can compute — supported by one proof and one in-session measurement. I tested both. The proof quantifies over states S_t = A_t (*) S_{t-1} + k_t g_t^T, which is gated linear attention; Gated DeltaNet's erase term S_{t-1}(I - beta k k^T) is a right-multiplication and lies outside that class. Measured: history H_B leaves ||S^T k|| = 0.00000 under GLA at every beta, and 0.398 / 1.593 / 4.078 / 6.373 under Gated DeltaNet at beta = 0.25 / 0.5 / 0.8 / 1.0, against ||v|| = 3.186. The proof does not cover the primary control. The measurement's headline — "|error| is ALL a Gated DeltaNet or Titans gate can see", novel-vs-contradicted AUROC 0.000 — is false in the specific way that matters: the delta rule computes vhat = S^T k as its own intermediate, and ||vhat||_1 gives novel-vs-contradicted AUROC 1.000 and clean-vs-contradicted 0.997 on the same stream, while a NINE-PARAMETER logistic on (log||e||, log||vhat||) reaches 3-way accuracy 0.965 and clean-vs-contradicted 0.998, beating the evidence channel's 0.907 on the one axis that is supposed to be ATTEST's alone.

THE PRIMARY VARIANT IS FALSIFIED OUTRIGHT. Spec 3.3 asserts z_t = Z^T k_t in R_{>=0}, but 3.1 makes k_t L2-normalised and therefore signed; with Z >= 0 enforced by A3, 51.1% of evidence-read components come out negative. The monotone-accumulator property — the whole biological warrant, which holds in the mushroom body precisely because the KC code is non-negative — does not survive. Consequences measured at d_k=128, d_z=4, row-stochastic P, 8 seeds: sparse non-negative keys give AUROC 1.000 / 0.844; dense L2-normalised keys give 0.500 / 0.500, i.e. exact chance, and stay at chance under LM-like anisotropy; an ORACLE label-aware floor lifts them only to 0.737 / 0.565. At the design's own stated initialisation rho = 1/sqrt(d_k), 100.0% of dense addresses have a_t = 0, which makes nov_t = 1 and cfl_t = 0 constants and makes d a_t/d rho identically zero. ATTEST-D at initialisation IS arm A2, with no gradient path out. The variant that works is ATTEST-S, which the design explicitly puts off the critical path. The design has this exactly backwards, and it contradicts the repo's own most reliable number — MEMORY_CAPACITY.md, dense delta 0.17 vs sparse 0.82 recall at 256 pairs — which it cites in support.

WHAT IS LEFT, AND IT IS SMALLER THAN CLAIMED. Strip out what ||vhat||_1 recovers for free, and what MesaNet's accumulated key covariance k^T M k already computes and inverts, and the unique residue is ONE non-negative scalar per address: accumulated |written residual|, separating "written twice consistently" from "written twice inconsistently" at AUROC 0.81-0.88 — not 0.955 — and only with sparse non-negative keys. d_z=4, the row-stochastic P, nov, cfl, sigma, the learned floor, the slow clock, the feature path: my measurement says d_z=1 captures the whole effect. That is a real observation with a clean mechanism — the delta rule's error correction actively DESTROYS accumulation, so one uncorrected channel restores it — but it is a one-scalar contribution, and it is the size of delta the risk section already flags as contestable.

WHICH CONTROL IT LOSES TO, IN ORDER. (1) A2', a learned gate given log||vhat||_1 and log(k^T M k) alongside log||e||^2 — free, no new state, no new recurrence, and not in K0. (2) A6, the content store widened to d_v+1 with the extra channel written Hebbianly with ||e_t||: 0.8% state instead of 3.1%, one column, no second recurrence, and it is the same accumulator. (3) A5 corrected to raise d_k rather than d_v — recall is rank-bounded by d_k, so as specified A5 is a strawman on the metric that decides the claim (Arora et al., ICLR 2024 arXiv:2312.04927; ICML 2024 arXiv:2402.18668). (4) MesaNet, which the design lists in its digest but omits from the gate competitor set.

ADJUSTED PROBABILITIES. P(K0 as currently specified "passes") ~ 0.8 — but near-uninformatively, because A2 is confounded by the deleted ||vhat|| path and the task was built to require the feature. P(A1 beats the CORRECTED control A2' by > 2 seed-SD) ~ 0.2. P(ATTEST-D specifically works, given the dense-key measurement) < 0.1. P(a defensible pre-registered claim in the named regime at rung 3) ~ 0.05, versus the design's 0.12-0.15, and essentially all of that mass now sits on ATTEST-S with its 16-20x state cost, which is a different paper against Product Key Memory (arXiv:1907.05242) and Memory Layers at Scale (arXiv:2412.09764) rather than a nested +3% modification of Gated DeltaNet.

WHAT I WOULD DO INSTEAD. Spend 20 CPU-minutes on K-minus-1 above before anything else, and commit the result whichever way it goes — "a provably-available signal that turns out to be freely computable by the control, and a non-negativity invariant that fails for dense keys" is a publishable methodological finding and costs almost nothing. Then, if and only if the sparse variant survives, re-scope the programme around ATTEST-S with d_z=1, the corrected control set (A2', A5-by-d_k, A6, MesaNet), a dead-fraction monitor as a first-class diagnostic, softplus rather than relu on the floor so the gradient to rho cannot die, and a corpus contradiction-rate measurement beyond the 512-token sliding window as a GATE before rung 2 rather than a mitigation after it. And fix the falsification asymmetry: as written, the pre-registration has a wide middle band that will be reported as "partially supported", which is precisely how phase 2's +0.7 seed-SD survived as a live hypothesis.

The discipline in this design is genuinely better than phase 1's or phase 2's — two candidate mechanisms falsified in twenty minutes before writing, an exactly nested control, an honest risk list. The failure is that the one claim it did NOT test is the one it built everything on, because it believed it had proved it. That is the same error as phase 2, in a new place: an axiom taken as settled and never measured.

### Strongest objections

- THE SEPARATION PROPOSITION DOES NOT APPLY TO GATED DELTANET, WHICH IS THE PRIMARY CONTROL. The proposition quantifies over states of the form S_t = A_t (*) S_{t-1} + k_t g_t^T, i.e. a Hadamard/diagonal decay plus a rank-1 addition. That is gated linear attention (GLA, Mamba-2, RetNet, lightning attention). Gated DeltaNet's recurrence is S_t = alpha_t * S_{t-1}(I - beta_t k_t k_t^T) + beta_t k_t e_t^T: the erase term is a RIGHT-MULTIPLICATION by a rank-1-perturbed identity, not a Hadamard product, so GDN is OUTSIDE the quantified class. I measured this (scratchpad/redteam_prop.py, Part 1): under GLA, history H_B ('+v then -v at the same gate') leaves ||S^T k|| = 0.00000 for beta in {0.25, 0.5, 0.8, 1.0}, exactly as claimed. Under Gated DeltaNet it leaves ||S^T k|| = 0.398 / 1.593 / 4.078 / 6.373 respectively, against ||v|| = 3.186 — i.e. the residual is the same order as the stored value itself, and GROWS with beta. The 'QED' is a theorem about a family that excludes DeltaNet, Gated DeltaNet, GDN-2, KDA, DeltaProduct and Titans-with-delta-memory. Every architecture ATTEST names as the thing it separates from is either outside the class or only inside it at a single value of beta. A reviewer who checks this will find the central proof does not cover the central control, and the paper collapses at that point.
- '|ERROR| IS ALL A GATED DELTANET OR TITANS GATE CAN SEE' IS FALSE, AND THE PRIMARY CONTROL A2 IS CONFOUNDED BECAUSE OF IT. The delta rule computes vhat_t = S_{t-1}^T k_t as an explicit intermediate — it is the first term of e_t = v_t - vhat_t. So ||vhat_t||_1 is available to any GDN gate at ZERO extra state, ZERO extra FLOPs and ZERO extra recurrence. I measured the 3-way address-class AUROCs on the design's own stream (scratchpad/redteam_sep.py, 8 seeds, sparse 32-of-2048, 96 keys, 96 distractor writes): by |error|, novel-vs-contradicted 0.441 and clean-vs-contradicted 0.536 (useless, as the design says); by ||vhat||_1, novel-vs-contradicted 1.000 and clean-vs-contradicted 0.997; by the ATTEST evidence channel a_t, 1.000 and 0.907. A 9-PARAMETER multinomial logistic on just (log||e||, log||vhat||_1), with NO evidence channel, reaches 3-way classification accuracy 0.965 (chance 0.333) with AUROCs 0.991 / 1.000 / 0.998 — BEATING the evidence channel on the clean-vs-contradicted axis that is supposed to be ATTEST's unique contribution. This is not a marginal point: in the ATTEST spec, ||vhat||_1 enters the gate ONLY through cfl_t = a_t/(||vhat_t||_1 + a_t + eps). Arm A2 'masks the evidence features to zero', which deletes cfl_t and therefore deletes ||vhat||_1 from the control's input set entirely. The pre-registered primary comparison A1 - A2 is therefore confounded with a free, state-derivable, non-evidence feature. Any A1 > A2 win is attributable to ||vhat|| rather than to Z, and the design would have no way to tell.
- THE PRIMARY VARIANT, ATTEST-D, IS AT CHANCE ON DENSE KEYS AND HAS NO GRADIENT PATH OUT. The design's spec (3.3) asserts z_t = Z_{h,t-1}^T k_t in R_{>=0}^{d_z}, but (3.1) defines k_t = norm(W_k x_t) — L2-normalised, hence SIGNED. With Z >= 0 (enforced by A3) and signed k, the read z = Z^T k is a signed sum and cancels: I measure 51.1% of evidence-read components NEGATIVE over 4,000 random unit keys (redteam_sep.py). The stated non-negativity invariant is simply wrong for the primary variant, and with it goes the biological warrant — the mushroom body's read is a monotone function of teaching precisely because the KC code is non-negative. Consequences measured (redteam_dense.py, redteam_floor.py, d_k=128, d_v=128, d_z=4, P row-stochastic, 8 seeds): SPARSE non-negative keys give a_t AUROC 1.000 (novel-vs-contradicted) / 0.844 (clean-vs-contradicted). DENSE L2-normalised isotropic keys give 0.500 / 0.500 — EXACT CHANCE. Anisotropic keys (s_i ~ i^-0.5 and i^-1.0, i.e. LM-like) give 0.500 / 0.500 and 0.504 / 0.502. Giving ATTEST an ORACLE, label-aware floor swept over 60 values including rho = 0 lifts dense to only 0.737 / 0.565, at a rho 23x smaller than the design's stated init. Worse: at the design's OWN stated initialisation rho = 1/sqrt(d_k) = 0.0884, 100.0% of addresses have a_t = 0 (against 16.9% for sparse). And a_t = 0 implies nov_t = exp(0) = 1 (constant), cfl_t = 0 (constant), and d a_t / d rho = -sigma * 1[||z||_1 > rho*sigma] = 0. All three evidence features are dead constants with NO gradient to rho. ATTEST-D at its own stated initialisation IS ARM A2, and backpropagation cannot move it off that point. The one variant that works is ATTEST-S, which the design explicitly places 'off the critical path' as 'a separate, later hypothesis' and whose thesis the design says it does not rest on. The design has it exactly backwards.
- THE DESIGN CONTRADICTS THE REPOSITORY'S OWN MOST RELIABLE IN-HOUSE MEASUREMENT. docs/phase3/MEMORY_CAPACITY.md shows that at 256 stored pairs a DENSE delta store with d_key=100 recalls 0.17 while a SPARSE 100-of-2048 store recalls 0.82, and at 512 pairs 0.07 vs 0.70 — the whole reason being the k^2/n near-orthogonality of sparse codes versus sqrt(P)-growing crosstalk for dense ones. The evidence read is a STRICTLY LOWER-SNR object than the content read: it recovers one non-negative scalar (or d_z=4) per address, with no redundancy across a d_v-dimensional value vector to average crosstalk away, and its floor must be estimated from a global statistic sigma that grows with the whole stream while the signal grows only with writes at that address. If dense addressing destroys CONTENT recall at 256 pairs — the repo's own number — it must destroy EVIDENCE attestation sooner, which is exactly what I measure. The design cites MEMORY_CAPACITY.md as support for ATTEST-S and then builds ATTEST-D on dense keys anyway.
- THE UNIQUE CONTRIBUTION REDUCES TO ONE SCALAR, AND HALF OF IT IS ALREADY SHIPPED. Strip out what other quantities already give you: (a) novel-vs-{clean, contradicted} is fully recovered by ||vhat||_1 (AUROC 1.000, measured) and independently by MesaNet's accumulated key covariance M = sum beta k k^T contracted as k^T M k, which MesaNet MAINTAINS AND INVERTS as its core operation (fla/layers/mesa_net.py, 30 CG steps/token) — that IS 'how much have I been taught at this address', so the risk section's claim that 'none accumulates teaching history at an ADDRESS' is false for MesaNet's attestation half; (b) what remains uniquely to Z is ONLY clean-vs-contradicted, i.e. the |e|-weighting, measured at AUROC 0.844 (sparse, oracle floor) and 0.806-0.907 across my configurations — not the 0.955 claimed. So the load-bearing content of the entire design is a SINGLE non-negative scalar per address: an accumulator of |written residual|. Everything else — d_z=4, the row-stochastic P, nov_t, cfl_t, sigma, the learned floor rho, the 7-input DAN net, the evidence-as-feature path (3.7), the slow document clock (3.8) — is scaffolding around it, and the design's own 'cheap ablation' d_z=1 is very likely the whole effect. A one-scalar mechanism is a workshop-paper contribution, not a phase-3 programme, and it is exactly the size of delta the risk section itself flags as 'novelty contested at review'.
- THE PRIMARY BENCHMARK DOES NOT INSTANTIATE THE PROPOSITION, AND THE BACKBONE SEES THE CONTRADICTION ANYWAY. The proposition's H_B is EXACT CANCELLATION (+v then -v at the same gate). MQAR-N generates 'silent re-binding of a fraction of keys mid-sequence', i.e. v then v'. These are different objects: I measure that in re-binding mode ||vhat||_1 gives novel-vs-contradicted AUROC 1.000 and, at beta=0.8, clean-vs-contradicted 0.997 — the proposition's separation simply does not bite in the regime the benchmark constructs. Separately, the backbone is a 3:1 hybrid with sliding-window attention at window 512. Any contradiction whose two writes fall within 512 tokens is visible VERBATIM in the token stream to the SWA layer; MQAR-N at P in {64,128,256,512} with T of a few thousand will place most contradicting pairs inside that window. And there is a sign error at long range: the design sets tau^Z = 8 * tau_h so 'evidence outlives content'. At an address whose CONTENT has decayed away but whose EVIDENCE is still high, the policy 'do not overwrite a heavily attested address on one noisy observation' suppresses exactly the write that should happen. Either the DAN net learns to invert the stated semantics — in which case nov/cfl are doing no interpretive work and the contribution is 'three extra learned features' — or the policy is wrong at the timescale the design chose.
- THE STATE-MATCHED CONTROL A5 IS MIS-SPECIFIED IN ATTEST'S FAVOUR. A5 is 'A2 with d_v raised so that total state bytes equal A1's'. But for associative recall the binding constraint is the RANK of the store, which is d_k, not d_v: a d_k x d_v matrix holds at most d_k mutually orthogonal key-value bindings regardless of d_v (this is the whole content of Arora et al., Zoology, ICLR 2024, arXiv:2312.04927, and the recall-vs-state-size frontier in Based, ICML 2024, arXiv:2402.18668, and it is reproduced in this repo's own MEMORY_CAPACITY.md, where dense delta at d_key=100 collapses past ~100 pairs). Raising d_v adds value bandwidth and buys almost nothing on MQAR-N, whose primary metric is the number of retrievable pairs. The honest state-match raises d_k by d_z (128 -> 132), adding d_z * d_v = 512 values AND 4 units of rank. A5 as written is a strawman, and it is the arm the design itself calls 'mandatory' and 'expected to be the tightest'. As specified it will lose for reasons that have nothing to do with the thesis.
- COST AND TRAINABILITY: THE 'NEARLY FREE' PROPERTY — WHICH THE DESIGN CALLS ITS SINGLE MOST IMPORTANT ONE — IS THE PROPERTY MOST AT RISK, AND K0 CANNOT DETECT IT. The +2.5% figure is FLOP arithmetic; it is not wall-clock. Two specific problems. (i) In Gated DeltaNet every gate is a function of x_t alone, which is precisely why all gates for a chunk can be computed in ONE parallel projection before the chunkwise UT-transform runs. ATTEST makes beta_t, gamma_t and lambda_t functions of z_t = Z_{t-1}^T k_t, i.e. STATE-DEPENDENT. The design's own fix — 'compute gates from the chunk-start state with an intra-chunk correction' — introduces an error that grows with chunk size, with beta (allowed up to 2), and with the write density; and sigma_t grows monotonically within the chunk, so the floor is SYSTEMATICALLY stale in one direction, which for a relu threshold is the difference between a live and a dead feature (see objection 3). If the approximation forces chunk 64 -> 16-32, the UT-transform kernel loses matmul efficiency and the real cost is multiples of 2.5%. (ii) The 7->16->4 DAN net per head per token is an elementwise/launch-bound op with essentially zero arithmetic intensity; its 200 MACs cost far more than 200 MACs of wall-clock. K0 is a 4-CPU-core numpy/torch module test: it measures NONE of this, so the design's headline property is untested until rung 1.5, by which point the engineering is sunk.
- THE BASE RATE IS WORSE THAN THE DESIGN ADMITS, AND THE NEAREST PRIOR RESULT IS THIS REPO'S OWN H1b. Phase-1 H1b: a learned input-dependent gate beat the fly-derived error gate by -2.6 seed-SD (0.9966 vs 0.9814, CI excluding 0). The general reason is structural — a learned gate whose input set contains the fixed rule's inputs contains the fixed rule as a special case — and ATTEST's answer is to change the INPUT SET rather than the form, which is the right response in principle but only works if the added input is genuinely unavailable, which objections 2 and 3 show it is not for ATTEST-D. Externally, 'Titans Revisited' (arXiv:2510.09551, 2025) reports that memory-update gating alone is insufficient unless the backbone's key/value projections co-adapt — ATTEST changes the gate and adds a read, and does not co-adapt projections beyond ordinary backprop. And the design's own session record is 3 for 3 against: coordinate-sign opponent channels falsified, confirmation-routed two-trace memory falsified, hand-specified evidence gate collapsed to 0.044 recall. A programme with that base rate should be weighting its priors down, not quoting P(rung 0 passes) ~ 0.55 for a task it constructed itself specifically to require the feature.
- THE PRE-REGISTRATION HAS AN ASYMMETRY THAT MANUFACTURES A NON-NULL. SUPPORTED requires four conjuncts (A1-A2 at >=2 of 3 P and >=2 of 3 sigma; A1-A5 > 0 at >=2 of 3 P; no regression on contradiction-free streams; evidence AUROC > 0.8). FALSIFIED requires only that the A1-A2 CI include 0 at >=2 of 3 P, OR that A1 lose to A5. The large middle — A1 clearly beats A2 (near-certain, since A2 is confounded per objection 2 and the task is built to need the feature) but A1 ties A5, or passes at 2 of 3 P but fails the sigma condition — falls into neither bin and will be written up as 'partially supported'. That is the exact route by which phase 2's +0.7 seed-SD became a live hypothesis rather than a null. Additionally, the primary metric is 'recall on the POOLED natural mix', but the pooled mix is set by the generator's own rates (re-binding at 0.004/key/step), so the effect size on the primary metric is a free parameter of the benchmark its author chose.
- THE EVIDENCE BASE IS A SELF-DESIGNED BENCHMARK. MQAR-N is new, constructed by the proposer, and its defining property ('addresses written more than once, sometimes inconsistently') IS the mechanism's precondition — the design states plainly that 'if every address is written at most once, the evidence channel is provably worthless'. The secondary benchmarks do not rescue this: RULER S-NIAH multi-needle (Hsieh et al., arXiv:2404.06654) measures INTERFERENCE between distinct needles, not CONTRADICTION at one address; SWDE/FDA/SQuAD (Arora et al., arXiv:2402.18668) measure single-write extraction; BABILong qa1-qa5 (Kuratov et al., NeurIPS 2024 D&B, arXiv:2406.10149) is mostly consistent fact accumulation. So no secondary benchmark tests the thesis; they test only non-regression. The entire positive claim would rest on one benchmark whose generator the claimant wrote. The missing, cheap, decisive measurement — what fraction of addresses in FineWeb-Edu at 2k-8k context are written more than once inconsistently BEYOND the 512-token SWA window — is listed only as a mitigation for a rung-2 null instead of as a gate before rung 1.

### Predicted failure modes

- MOST LIKELY (I put ~0.6 on this): ATTEST-D's evidence features are dead at initialisation and stay dead. Measured: at the design's stated rho = 1/sqrt(d_k), 100.0% of addresses give a_t = 0 with dense L2-normalised keys, which makes nov_t = 1 and cfl_t = 0 constants and makes d a_t/d rho identically zero. The run completes, the loss curves look normal, A1 and A2 are statistically indistinguishable, and the post-mortem reads 'the learned gate did not need the signal' — the MODAL FAILURE the design names. It will be the WRONG post-mortem: the gate never saw the signal. This is the exact analogue of the dead-latent pathology in top-k SAEs (Gao et al., arXiv:2406.04093, up to 90% dead latents without mitigation), and the design has no auxiliary loss, no resampling and no dead-feature monitor.
- SECOND MOST LIKELY (~0.5, and compatible with the first): A1 beats A2 comfortably and the team declares the thesis supported, when the entire margin is ||vhat||_1 leaking in through cfl_t. A2 masks the evidence features, which deletes the only path by which the control could see ||vhat||_1. The win is then attributed to the non-negative accumulator Z when it is actually attributable to a quantity Gated DeltaNet already computes. My measurement: a 9-parameter logistic on (log||e||, log||vhat||_1) reaches 3-way accuracy 0.965 and clean-vs-contradicted AUROC 0.998, above the evidence channel's 0.907 on the same stream. Nothing in K0 as pre-registered can distinguish these two explanations.
- A1 ties or loses to a CORRECTLY specified state-matched control (A5 with d_k raised rather than d_v). Recall capability is bounded by recurrent state size and, for a d_k x d_v store, by rank d_k; the repo's own MEMORY_CAPACITY.md shows dense delta collapsing from 0.88 to 0.07 recall as P goes 32 -> 512 at fixed d_key. Converting the 512 extra values into 4 more units of rank is a direct, theory-backed buy on the primary metric, while converting them into gate quality is a second-order buy. The design already calls this 'the tightest' control and then specifies it in the one way that makes it weak.
- The chunked implementation silently changes the mechanism. beta_t depends on z_t = Z_{t-1}^T k_t, resolved from the chunk-start state. sigma grows monotonically within a chunk, so the floor rho*sigma is systematically UNDER-estimated intra-chunk, so a_t is systematically OVER-estimated, so cfl_t drifts toward 1 and nov_t toward 0 as the chunk progresses — a position-within-chunk artefact in a feature that is supposed to be positional-invariant. Training then fits the artefact. Nothing in K0 (sequential numpy) exposes this; it appears for the first time at rung 1.5 when the kernel is written, and it will be very hard to attribute.
- TRANSFER FAILURE WITH THE WRONG DIAGNOSIS: rung 0 and rung 1 pass on sparse keys and MQAR-N; rung 2 returns null on FineWeb-Edu; the team concludes 'contradiction is rare in natural text' when the actual cause is that the 3:1 hybrid's sliding-window attention at window 512 already resolves every in-window contradiction exactly, so the recurrent layer never faces the regime. This is unfalsifiable after the fact unless the SWA-window ablation and the corpus contradiction-rate measurement are run BEFORE rung 2.
- The +2.5% FLOPs / +3.1% state accounting does not survive contact with a GPU: either the intra-chunk approximation forces chunk 64 -> 16-32 (losing matmul efficiency in the UT transform), or the per-head 7->16->4 DAN net becomes launch-bound. The design's 'single most important property' — that matched-compute controls are nearly free — evaporates, and the team is back in the phase-2 position of an unmatched control, this time in the other direction.
- NOVELTY COLLAPSE AT REVIEW: a reviewer observes that MesaNet already maintains and inverts the accumulated key covariance sum beta k k^T, whose contraction k^T M k is per-address attestation mass; that PGDN's accumulated per-channel second moment is a coarse address-space histogram of the same quantity; and that GDN-2 already decouples the erase and write gates. The residual delta is 'we added |e| weighting to an accumulator that already exists', which is a one-scalar contribution and does not carry a phase-3 claim.
- SCOPE INVERSION (the most likely GOOD outcome, and it still breaks the plan): the effect is real but ONLY for ATTEST-S. The sparse variant then carries 16-20x the state (n x d_z rather than d_k x d_z), which destroys the '+3% state, nearly free' framing, moves the comparison set to Product Key Memory (Lample et al., NeurIPS 2019, arXiv:1907.05242) and Memory Layers at Scale (arXiv:2412.09764), and reintroduces the wall-clock risk the design deliberately deferred. The programme succeeds scientifically and fails as the cheap, nested, drop-in modification it was sold as.

### What would change my mind

- A CORRECTED PRIMARY CONTROL THAT STILL LOSES. Add arm A2' — identical to A2 but with the gate input set u = [log1p(||e||^2/m), log1p(||vhat||_1), log1p(||vhat||_1/(||e||_1+eps)), log1p(k^T M k) with M = sum beta k k^T, phi, s]. Every one of those is derivable from quantities the delta rule or MesaNet already computes, at zero new recurrence. If A1 still beats A2' by > 2 seed-SD with a paired-bootstrap CI excluding zero at >= 2 of 3 P values and >= 2 of 3 noise levels, I withdraw objections 2 and 5 and accept that Z carries something genuinely new. This is the single most important change to K0 and it costs one extra arm.
- THE DENSE-KEY RESULT REVERSING UNDER A FIXED, PRE-REGISTERED FLOOR. If, with dense L2-normalised keys at d_k = 128 and a floor rule fixed in advance (not oracle-selected), the evidence channel reaches novel-vs-contradicted AUROC > 0.9 and clean-vs-contradicted > 0.8, with a reported dead-fraction (addresses with a_t = 0) below 20%, then ATTEST-D is viable and objection 3 dies. My measurement says 0.500/0.500 at the stated init with 100% dead, and 0.737/0.565 even with an oracle floor 23x smaller. Show me otherwise and the primary variant is back.
- A MEASURED CONTRADICTION RATE IN REAL TEXT, BEYOND THE ATTENTION WINDOW. Instrument a trained small LM (or just the tokenised corpus with a learned-key proxy) and report: what fraction of key-addresses in FineWeb-Edu at 2k-8k context receive more than one write, what fraction of those are inconsistent, and crucially what fraction of the inconsistent ones have their two writes separated by MORE than the 512-token sliding-window. If that last number is materially above zero, the regime exists and rung 2 is worth running. If it is near zero, the transfer failure is predicted in advance and rung 2 should not be funded. This is cheap, it is measurable today, and it should be a GATE before rung 1, not a mitigation after rung 2.
- AN ABLATION SHOWING THE APPARATUS EARNS ITS KEEP. If d_z = 4 with the row-stochastic P beats d_z = 1 with uniform P by more than 1 seed-SD, and if the evidence-as-FEATURE path (3.7) and the slow document clock (3.8) each contribute beyond noise, then the design is more than one scalar and objection 5 weakens. My measurement finds d_z = 1 captures the whole effect (0.844-0.907 clean-vs-contradicted, matching d_z = 4). If the full apparatus does not beat its own cheapest ablation, cut to d_z = 1 and re-scope the claim accordingly — which is a better paper anyway.
- A SIGN-CORRECTED, GRADIENT-ALIVE FORMULATION. If the read is redefined so the monotone-accumulator property actually holds for signed keys — e.g. a_t = sum_i |(Z^T k)_i| with Z accumulating against |k_t| rather than k_t, or a non-negative key map (relu/softplus/k-WTA) before the evidence read, or replacing relu(x - floor) with softplus((x - floor)/tau) so the gradient to rho never dies — and the dense-key AUROCs recover, then the spec defect is repairable and objection 3 becomes a bug report rather than a refutation. I would want to see the dead-fraction reported at every training step as a first-class diagnostic.
- GDN BROUGHT INSIDE THE PROPOSITION, OR THE PROPOSITION RESTATED. If the separation is re-proved for the actual Gated DeltaNet recurrence S_t = alpha_t S_{t-1}(I - beta_t k k^T) + beta_t k e^T — including the erase term — and the residual ||S^T k|| after H_B is shown to be uninformative about teaching history for that recurrence, I withdraw objection 1. As it stands I measure that residual at 0.40-6.37 against ||v|| = 3.19, i.e. it is not small and it is beta-dependent, which means it carries information.
- A1 BEATING A6, THE MINIMAL ALTERNATIVE. Add an arm A6: the ordinary content store widened to d_v + 1, where the extra value channel is written HEBBIANLY (uncorrected) with ||e_t||, everything else unchanged. That is d_k extra state (0.8%, not 3.1%), one extra column, no second recurrence, no P, no sigma, no floor, no new sequential coupling, and it is an accumulator of teaching at the address. If A6 matches A1, the entire Z apparatus is over-engineering and the finding is 'the delta rule's error correction destroys accumulation; one uncorrected channel restores it' — which is a cleaner, cheaper and more defensible result than ATTEST.

### Minimal experiment to kill it

EXPERIMENT K-MINUS-1, "IS THE SIGNAL ACTUALLY UNAVAILABLE?" — ~20 CPU-MINUTES, NO TRAINING, NO GRADIENTS, NO MODEL. It costs about 1/20th of K0 and runs before it. I HAVE ALREADY RUN IT AND ATTEST-D FAILS IT. Scripts and raw output are in the scratchpad at redteam_sep.py, redteam_prop.py, redteam_dense.py, redteam_floor.py; they should be hardened, seeded, and committed under experiments/ with a PREREG before any ATTEST code is written. [Editor's note, added when the repository was published: the red team's scratch scripts named here and above were not kept; the measurement they call for was re-done under a pre-registration as E0, `experiments/p3_e0_signal`.]

WHY THIS IS THE RIGHT KILLER, NOT K0. K0 asks the SECOND question: "can a learned gate convert the signal into recall?" The FIRST question is "is the signal unavailable to the control?" — and the design claims to have settled that BY PROOF, so it never tests it. That is where the design is weakest, and it needs no training to test. A null here kills the thesis for a STRUCTURAL reason (the signal is available / the read is dead), which is far more informative than K0's null, which would be an optimisation-failure reason and is confounded anyway.

PROTOCOL.
1. Generate the design's own stream: 3 x 32 addresses in classes {never-written, written-twice-consistently, written-twice-inconsistently}, plus 96 distractor writes, shuffled, into a gated-delta store. Cross two key codes — SPARSE 32-of-2048 non-negative (ATTEST-S) and DENSE d_k=128 L2-normalised (ATTEST-D, the PRIMARY variant) with anisotropy s_i ~ i^-a for a in {0, 0.5, 1.0} — with beta in {0.25, 0.5, 0.8, 1.0} and both contradiction modes (exact cancellation, as the Proposition assumes; and silent re-binding, as MQAR-N actually generates). 8 seeds.
2. At each address compute FOUR scalars: (i) ||e_t||_1; (ii) ||vhat_t||_1 = ||S^T k_t||_1, WHICH THE DELTA RULE ALREADY COMPUTES; (iii) k_t^T M k_t with M = sum beta k k^T, which is MesaNet's key covariance; (iv) ATTEST's a_t = relu(||Z^T k||_1 - rho*sigma), with rho at the design's stated init AND swept over 60 values including an oracle label-aware best (the most charitable possible treatment).
3. Report, per configuration: pairwise and pooled 3-way AUROC for each scalar alone; the 3-way accuracy of a 9-PARAMETER multinomial logistic on (log||e||, log||vhat||_1) with NO evidence channel; and the DEAD FRACTION, the proportion of addresses with a_t = 0.

PRE-REGISTERED KILL CONDITIONS (any one fires).
K1. The 9-parameter (||e||, ||vhat||) logistic reaches pooled 3-way AUROC within 0.02 of the evidence channel's. Then the Proposition has no practical content and A2 is confounded, because ||vhat|| is free.
K2. On DENSE keys the evidence channel's clean-vs-contradicted AUROC is below 0.7 at the stated floor initialisation. Then ATTEST-D is dead and only ATTEST-S survives.
K3. The dead fraction on dense keys exceeds 0.5 at initialisation. Then the features are constants with no gradient to rho and K0's A1 arm is literally A2.
K4. ||S^T k|| after H_B under the true Gated DeltaNet recurrence exceeds 0.1 * ||v|| for beta in [0.25, 1.0]. Then the Proposition does not cover its own primary control.

WHAT I MEASURED (8 seeds each).
K1 FIRES: logistic 3-way accuracy 0.965 (chance 0.333), AUROCs novel-vs-clean 0.991, novel-vs-contradicted 1.000, clean-vs-contradicted 0.998, against the evidence channel's 1.000 / 0.907; ||vhat||_1 alone gives 0.972 / 1.000 / 0.997 while |error| alone gives 0.091 / 0.441 / 0.536.
K2 FIRES: dense isotropic 0.500 / 0.500, dense anisotropic a=0.5 0.500 / 0.500, a=1.0 0.504 / 0.502; with an ORACLE floor only 0.737 / 0.565; sparse non-negative 1.000 / 0.844.
K3 FIRES: dead fraction 1.000 on dense (0.169 on sparse), and 51.1% of evidence-read components are negative with signed keys, contradicting spec 3.3's stated invariant z_t in R_{>=0}.
K4 FIRES: ||S^T k|| after H_B = 0.398 / 1.593 / 4.078 / 6.373 at beta = 0.25 / 0.5 / 0.8 / 1.0, against ||v|| = 3.186 — versus exactly 0.00000 under GLA, the class the Proposition actually quantifies over.

HONEST CAVEATS ON MY OWN MEASUREMENT. My stream is not bit-identical to the design's (my |error| means are novel 12.98 / clean 8.80 / contradicted 13.10 against their 12.98 / 5.82 / 6.19, which implies a different probe value at contradicted addresses); but every comparison I draw is BETWEEN scalars on the SAME stream, so the relative conclusions are internally controlled and do not depend on matching their generator. "Available" is not the same as "a trained gate uses it" — that cuts both ways and is exactly why K0 is still needed AFTER this, with arms A2' and A6 added. And the clean-vs-contradicted separability from ||vhat|| is beta-dependent (0.997 at beta=0.8, 0.456-0.502 at beta=0.5), which is itself the point: the design's separation is a property of a hyper-parameter, not a theorem.

TOTAL COST TO KILL: ~20 CPU-minutes and roughly half a day of engineering, against K0's ~6 CPU-hours and 3 days. If it survives, run the CORRECTED K0 (add A2' with vhat and key-covariance features; add A6, the d_v+1 Hebbian-written channel; re-specify A5 to raise d_k not d_v; add a dead-fraction monitor to every arm; make the evidence-as-feature and evidence-as-gate paths separately ablated) and measure the corpus contradiction rate beyond the 512-token window BEFORE funding rung 2.
