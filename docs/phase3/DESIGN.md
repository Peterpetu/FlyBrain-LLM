# Phase 3 design record: a state, not a gate

*Design document for the third architecture of this repository. It is written to the standard the
repository set for itself in phases 1 and 2 (a written specification, matched controls, hash-locked
pre-registration, mechanical verdicts, formal verification against the specification), with one
addition that both earlier phases lacked: every load-bearing assumption is either MEASURED in this
repository before it is built on, or marked as the hypothesis under test. Nothing in this document
is a claim about performance. Sections 2 and 5 are measurements; sections 3, 4 and 6 are proposals
with their decision gates.*

*Status (October 2026). The proposals of sections 3-7 have now been tested as far as the ladder's
own rules allowed. The ladder table in section 4 carries the measured status of every rung, the
predictions of section 5 and the odds of section 7 carry their outcomes, and the design history
(why rung 1 was built three times and rung 1b twice) is in EXPERIMENT_PLAN.md. The rest of the
text is left as written before the rungs ran, so that a reader can check the predictions against
the outcomes.*

Reading order for a reviewer with an hour: section 1 (the doctrine), section 2 (what is measured),
section 3.3 (the exact reduction to Gated DeltaNet), section 4 (the ladder and its kill
conditions), section 7 (odds).

---

## 1. The question and the doctrine

**The question.** Does the fruit fly's memory system contain a computational principle that
improves the best current sequence-memory layer, the gated delta rule family (DeltaNet, Gated
DeltaNet and its 2025-26 descendants), at matched compute, matched state and matched parameters,
in a regime that natural text contains?

**Why the first two phases returned nulls, in one sentence each.** Phase 1 treated a connectome as
a prior (a whole-graph mask was indistinguishable from a degree-preserving shuffle; the extracted
ring tracked state worse than an unstructured layer, -9.1 seed-SD). Phase 2 imported a vocabulary
(active inference) as fixed functional forms inside a backprop-trained model; every such form lies
inside a learned gate's hypothesis class, so backpropagation found it and the biology was redundant
(H1b: the learned gate beat the fly-derived error gate by 2.6 seed-SD; AIF-H1: K = 3 inference
steps beat K = 0 by 0.7 seed-SD, below the bar, against a control with 1.94x the parameters). Both
are recorded in docs/phase3/REFLECTION.md.

**The doctrine (five methodological axioms, docs/phase3/AXIOMS.md M1-M5).**
- **M1. Propose a state, not a gate.** A fixed rule that a learned gate with the same inputs
  contains as a special case cannot win. The only thing that can win is a per-address STATE the
  gate cannot reconstruct from its inputs. The gate stays learned; the input set changes.
- **M2. The control is the same code with the new inputs zeroed.** Equal parameters, equal
  kernels, equal tuning budget, by construction. There is no "wide control" argument for anyone.
- **M3. Signal availability is measured before any gate is trained.** A learned gate can only use
  information that is in its inputs; whether a candidate state carries information the control's
  own intermediates do not is a question with no training in it (E0).
- **M4. Regime existence is measured before scale.** A memory principle only matters on streams
  where addresses are written more than once, sometimes inconsistently, beyond the attention
  window; whether text contains that regime is a corpus measurement (D1), not a hope.
- **M5. No middle band.** Every pre-registration names SUPPORTED and FALSIFIED so that they
  exhaust the outcomes; "partially supported" is the route by which phase 2's +0.7 seed-SD stayed
  alive and it is closed.

The doctrine came out of the design panel (docs/phase3/DESIGN_PANEL.md): three independent designs
(ATTEST, SIEVE, CLAW), two judges, one red team. The judges chose ATTEST for M1 and M2; the red
team then showed that ATTEST's central proof did not cover its own control and that its primary
(dense-key) variant was dead at initialisation, and demanded M3 and M4 as gates. Both gates have
now been run. Section 2 is what they said.

---

## 2. What is measured (the ground the design stands on)

Every number below is from a pre-registered or documented run in this repository.

**2.1 Capacity of sparse addresses (docs/phase3/MEMORY_CAPACITY.md, numpy).** At equal active rows
per event, a delta-rule store over a sparse k-of-n code keeps recalling far past the dense store's
rank ceiling: at P = 4 d_key the sparse store is at 0.90 where the dense one is at 0.25; sparse
100-of-2048 at 256 pairs 0.82 against dense d_key = 100 at 0.17. The price is state n/k times
larger (16-20x) and a gather/scatter access pattern. The delta rule matters as much as sparsity
(sparse Hebb decays to 0.33 where sparse delta holds 0.90). Not shown: that the advantage survives
learned keys and gates; that is E1 (module level, queued) and rung 1.

**2.2 Rate as inferred precision (docs/phase3/ADAPTIVE_RATE.md).** A surprise-gated write rate
matches a change-point Kalman oracle without knowing the regime (0.81 vs 0.81 at sigma = 0.3) and
beats a TUNED fixed rate by a point (0.81 vs 0.80). It is robustness and tuning freedom, not
accuracy. It enters the design as one gate input, never as the thesis.

**2.3 Naive consolidation is negative (docs/phase3/CONSOLIDATION_TOY.md).** Replay from a fast
store into a slow one retained 0.06 against 0.02 beyond the fast horizon. Consolidation is off the
critical path; the two missing ingredients it identified (selection, a confidence gate) are what
the states below would supply, so it returns as a separately pre-registered rider only.

**2.4 Genome bottleneck (scripts/genome_bottleneck.py).** Type-level description length, type
repetition, hemilineage-to-transmitter predictability and bilateral reproducibility of MaleCNS v1.0
were measured; they say the connectome is a sample from a distribution, not a blueprint, and they
are why this design reads NO adjacency anywhere (AXIOMS A1, A8).

**2.5 Signal availability, E0 (experiments/p3_e0_signal, 8 seeds, no training).** Same current
event at every address (a fresh value), different histories {never written, twice consistent,
twice inconsistent}; the control is a logistic regression on the delta rule's own intermediates
(||e||, ||vhat||, cos(v, vhat), k^T M k); each candidate state is added in turn.
- On sparse 32-of-2048 addresses at write rate beta < 1 and no observation noise, the control is
  already at 0.92-1.00: the store's residual read norm is smaller after a contradiction, so the
  delta store itself remembers inconsistency. At beta = 1 (full overwrite) the control sits at the
  novel-vs-taught ceiling (0.64-0.68) and the states add +0.21 to +0.37.
- Ranking of the states on sparse keys, clean-vs-contradicted AUROC: the content-based
  confirmation count `nconf` 1.00 at every beta; the familiarity-gated contradiction mass `zc`
  0.92-1.00; ATTEST's evidence channel `a` at its stated floor 0.65-0.98, rising with beta;
  last-touch age nothing (+0.007 accuracy).
- On dense signed keys (d = 128) `a` is dead (dead fraction 1.00, AUROC 0.50, red-team kill
  conditions K2 and K3 fire in 24 of 24 configurations); `zc` survives weakly (0.64-0.73);
  `nconf` moderately (+0.12 to +0.24 accuracy at beta >= 0.8). Anisotropic keys degrade everything.
- The exact-cancellation "separation proposition" does not cover the gated delta rule (residual
  beta^2 ||v|| after +v, -v; K4 fires for beta >= 0.5). Only linear attention cancels.
- Admitted to E2 by the pre-registered rule: `a` (+0.065), `zc` (+0.065), `nconf` (+0.099).
Two caveats stated with the result: E0 has no observation noise (E2 does), and at beta < 1 the null
is a ceiling effect for the control, not evidence that the states are redundant under noise.

**2.6 The corpus regime, D1 v2 (experiments/p3_d1_corpus).** PERSISTENT re-bindings (a key-like
token's next token changes and the new binding is used again) against a within-key permutation
null. At W = 8192, far (> 512-token gap) persistent re-bindings occur at 1.5-2.0 per 1,000 tokens
on WikiText-2 and 1.4 on PTB, 1.5-1.9x the null (z 21-39). Consistent repeats (the copy regime a
context window serves) are 1.5-1.9x the null and about four times more frequent. The regime exists,
beyond the window, as a minority of long-range repeats: about three persistent re-bindings per
2,048 tokens. v1 of this diagnostic was retracted on its first run as an instrument defect and is
kept in the directory as the record.

**2.7 The panel's own falsifications (docs/phase3/DESIGN_PANEL.md).** Opponent channels by
coordinate sign reduce exactly to the gated delta rule with tied decays (max deviation 6.7e-16) and
lose with untied decays; a confirmation-ROUTED two-trace memory (slow trace on agreement, labile on
contradiction, as a fixed rule) loses to the tied single-trace control on every metric; a
hand-specified evidence gate latched shut (recall 0.044). All three are fixed rules, all three are
consistent with M1, and none is in this design.

---

## 3. The architecture

### 3.1 Shape of the bet

A minimal, nested extension of Gated DeltaNet inside a conventional hybrid decoder. The backbone
carries no fly content by design. The memory layer gains per-address STATES that E0 showed carry
teaching history the gate's free intermediates do not, and a learned gate that may read them. The
exact control is the same layer with the state slots zeroed. Two address codes are carried as
separate hypotheses: DENSE (L2-normalised keys, the Gated DeltaNet twin; carries `nconf` and `zc`
only, since E0 killed `a` there) and SPARSE (non-negative k-of-n code over an expansion; carries all
three states and the capacity advantage of 2.1, at n/k times the state).

### 3.2 Backbone

Decoder-only, pre-norm, SwiGLU MLP, tied embeddings, RoPE, layer schedule 3:1 (three memory layers
to one sliding-window attention layer, window 512), matching current hybrids. Rung-1 config:
d = 256, 4 layers (3 memory + 1 SWA); rung-2: d = 512, 8 layers (~22M non-embedding); rung-3:
d = 768, 12 layers, H = 4 heads, d_k = d_v = 128 (~110M). No pure-recurrent model is state of the
art anywhere; the claim, if any, is "a better 75% layer", never "attention replaced".

### 3.3 The memory layer, per head, with its exact reduction

Per-sequence state at head h (all reset at sequence start):

| state | shape | update | role |
|---|---|---|---|
| content S | d_k x d_v | gated delta rule | the ordinary store (Gated DeltaNet) |
| confirmation C | d_k x d_c | delta write of the confirmation vector along P v; reset to 0 on contradiction | "how many consecutive observations agreed with what is stored here" |
| contradiction mass Zc | d_k x 1 | monotone accumulation of beta * \|e\| at addresses that already hold something and disagree | "how often was this address contradicted" |
| evidence mass Z, sigma | d_k x 1, scalar | monotone accumulation of beta * \|e\| and its total (ATTEST); SPARSE only | "how much was written here" |
| trackers a, b | 2 scalars | Gamma tracker of the squared error, forgetting 0.9 | the expected error, for the confirmation criterion and the precision input |

P in R^{d_c x d_v} is a learned projection (d_c in {4, 16}; d_c = d_v is the module-level form
tested in E2); it is what keeps the confirmation vector at +3-12% state instead of +100%.

Per token, with q_t, k_t L2-normalised (dense) or the sparse code (3.4), v_t = W_v x_t, and the
states indexed so that X_{t-1} is the state before token t:

    vhat_t  = S_{t-1}^T k_t ;  e_t = v_t - vhat_t          (the delta rule's own intermediates)
    nconf_t = (C_{t-1}^T k_t) . normalise(P vhat_t)        (confirmations counted along the stored value)
    zc_t    = log1p( relu( Zc_{t-1}^T k_t ) )
    a_t     = log1p( softplus( Z_{t-1}^T k_t - rho sigma_{t-1} ) )   (SPARSE variant; rho = k/n. Dense keys: rho = 1/sqrt(d_k), and E0 says it is dead there)
    u_t     = [ W_g x_t, v_t, e_t, log ||e_t||^2, log ||vhat_t||, log(pi_t ||e_t||^2) ; nconf_t, zc_t, a_t, 0 ]
    beta_t, gamma_t = sigmoid(.), lambda_t = softplus(.)   (a 32-unit tanh MLP on u_t: write, erase, forget)
    alpha_t = exp(-lambda_t / tau_h)                       (tau_h: a fixed ladder, log-spaced from 8 to 2048 tokens across heads)
    S_t     = alpha_t (I - gamma_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
            = alpha_t S_{t-1} + k_t (beta_t v_t - gamma_t r_t)^T ,  r_t = alpha_t S_{t-1}^T k_t   (the decayed prediction)
    y_t     = S_t^T q_t                                    (the layer's read, after the write)
    ed_t    = v_t - r_t                                    (the error the delta rule actually corrects)
    agree_t = [ ||ed_t||^2 / d_v < 1.5 max(b_t / a_t, 1e-4) ]        (relative to the tracked expected error: adapts to the noise level)
    a_{t+1} = 0.9 a_t + 0.5 ;  b_{t+1} = 0.9 b_t + 0.5 ||ed_t||^2 / d_v
    C_t     = C_{t-1} + k_t ( agree_t (C_{t-1}^T k_t + normalise(P v_t)) - C_{t-1}^T k_t )^T     (+1 along the value on agreement, reset on contradiction)
    Zc_t    = Zc_{t-1} + beta_t |ed_t|_mean (1 - agree_t) [ ||vhat_t|| >= 0.25 ] k_t          (a disagreement at an address that already holds something)
    Z_t     = Z_{t-1} + beta_t |ed_t|_mean k_t ;  sigma_t = sigma_{t-1} + beta_t |ed_t|_mean
    o_t     = W_o [ RMSNorm(y_1..y_H) ; nconf_1..H ; zc_1..H ]        (the states are features too; separately switchable)

**The two-store form (E4).** With `n_traces = 2` the layer holds two content stores S^(0), S^(1)
(slow and labile: the gate biases start beta^(0) low and beta^(1) high), each updated by the same
recurrence with its own (beta, gamma, lambda)_t from one gate MLP whose input carries both stores'
errors and read norms; the history states are tracked against the labile store; and the read is
a learned mixture at the query address,

    y_t = (1 - mu_t) S^(0)_t^T q_t + mu_t S^(1)_t^T q_t ,
    mu_t = sigmoid( MLP[ cos(y^(0), y^(1)), log ||y^(0)||, log ||y^(1)|| ; nconf_q, zc_q, a_q, 0 ] )

with the slots read at q_t after the write. The module ladder found this the only form in which
the states converted into accuracy (E4, replicated); `n_traces = 1` is the single-store layer
bit-for-bit (frozen reference snapshot in tests/data).

**Exact reduction (the control, M2).** With the four state slots of u_t zeroed the layer is Gated
DeltaNet with decoupled erase and write gates, a learned gate on the delta rule's own intermediates
and 2H dead feature columns (arm `free`); with gamma_t tied to beta_t it is Gated DeltaNet. Same
parameters (dead weights), same kernels, same sweep. The reduction is a unit test: with the states
off and constant gates the layer reproduces a textbook Gated DeltaNet recurrence to 1e-5.

**Gradient policy.** The gate receives gradient only through W_g x_t; v_t, e_t, the three logs and
the slots are detached (sensing inputs). The write coefficients of Zc and Z (beta_t, |ed_t|) are
detached, so the zc feature trains only the keys; the nconf feature trains P, the history in C and,
through S, the projections and the gate. The confirmation criterion is a hard threshold. The
two-store readout receives gradient through cos(y^(0), y^(1)) and the two log norms (so the mixing
weight trains both stores' contents), while its slots are detached, as in the module-level E4. Two
further facts of the two-store form: the slow/labile asymmetry at initialisation is in the WRITE
gate only (beta logits -1 and +2; erase and forget start equal), and the precision input
log(pi ||e||^2) uses the labile store's error.

**What is new, and what is not.** New: per-address states that record teaching HISTORY
(confirmation count, contradiction mass) read at the same address as the content, plus the
measurement that they carry information the gate's intermediates do not (E0). Not new and not
claimed: sparse hashed memory (Product-Key Memory, Memory Layers), the delta rule, gated decays,
multi-timescale heads, divisive normalisation, MesaNet's key covariance (which supplies
"how much was written here" and is therefore in the control's input set as k^T M k where the
kernel permits), the fly Bloom filter (Dasgupta et al. 2018: occupancy, not consistency).

### 3.4 The sparse address (the fly variant; a separate hypothesis)

Learned compression to d_pn = 128, divisive normalisation with a learned exponent (initialised at
1.5), block-sparse expansion to n = 4096 rows with per-block adaptive thresholds driven to a target
coding level (the local APL rule), hard top-k at inference; the code is non-negative and unit norm.
Per-block routing keeps the gathered rows contiguous. Chunk-union densification (gather the union of
touched rows per chunk, run the unmodified chunked delta kernel on the sub-matrix, scatter back)
makes it GPU-viable. The fan-in and coding level are DERIVED by maximising the participation ratio
of the code on token data, not copied from the fly's K = 6, f = 5%. State: n x (d_v + d_c + 2)
per head. This variant is gated behind E1 (module level, queued) and a wall-clock parity gate at
0.5x of the dense layer's throughput.

### 3.5 Cost, exactly, and the implementation (`flyprinc/hybrid.py`)

Measured by `flyprinc.hybrid.layer_cost` at d_k = d_v = 128, d_c = 16, H = 4, per head per token
(MACs count the recurrence, the reads, the two P projections and the gate MLP; parameters count the
whole layer at d_model = 768):

| arm | MACs | state values per head | parameters | delta vs the twin |
|---|---|---|---|---|
| `free` (the Gated DeltaNet twin with a learned gate) | 58,176 | 16,384 | 1,653,219 | - |
| `state` (nconf, zc) | 66,624 | 18,560 | 1,653,219 | +14.5% MACs, +13.3% state, +0 parameters |
| `state` + a (sparse variant only) | 66,880 | 18,689 | 1,653,219 | +15.0% MACs, +14.1% state |
| `hebb_column` (one uncorrected accumulator) | 58,432 | 16,513 | 1,653,219 | +0.4% MACs, +0.8% state |
| two stores, `free` | 112,224 | 32,768 | 1,657,623 | +93% MACs, +100% state vs the single twin |
| two stores, `state` (nconf, zc) | 124,896 | 34,944 | 1,657,623 | +11.3% MACs (the state reads at the key and at the query), +6.6% state vs two-store `free`; +0 parameters |

The parameter counts are equal by construction within a substrate (the slots and the feature
columns exist in every arm; in the free arm they are dead weights, and the confirmation projection
P receives no gradient there). The two-store form doubles the memory MACs and state of the
single-store twin: that is the price of the substrate E4 supported. Rung 1's second control,
`single_wide`, is one store with d_k raised from 64 to 145 so that its memory state per layer
equals the two-store state arm's (37,120 values); it is rank- and state-matched, not
parameter-matched (4,075,745 parameters against 3,575,429, from the wider projections), and a win
of `single_wide` is read with that in mind. The rank-matched control `twin_dk` raises the twin's
d_k so that its memory state equals the state arm's (64 -> 81 at the rung-1 size), because recall
is rank-bounded by d_k, not by d_v (the red team's correction); that arm therefore has 3.0% more
parameters than the others at the rung-1 size (3,674,849 vs 3,568,361), which its pre-registration
states, and a win of `twin_dk` over `state` is read with that in mind.

Implementation facts, each covered by `tests/test_hybrid.py` or measured in the verification pass
recorded below:
- The sequential form is the specification. The chunked form (chunk size n) is exact for every
  state given the gates, in every edge case tried (T not a multiple of n, n = 1, n = T, B = 1,
  H = 1, every state set, gamma != beta, non-fresh initial states, forget drives from 0 to 40):
  the content and confirmation stores use the UT-transform (a unit-lower-triangular solve per
  chunk), the cumulative decay is kept in log space and masked before exponentiation (an unmasked
  exp overflowed in the backward pass at large forget drives; fixed and tested), and the
  contradiction and evidence masses are prefix sums.
- The gate inputs are state-dependent (the red team's engineering risk). They are computed from
  the chunk-start states and then refined from the exact intra-chunk reads of the previous pass, a
  Picard iteration on the gates. Because the intra-chunk dependency is causal, after p passes the
  gates of the first p + 1 tokens are exact, so the iteration is EXACT within chunk - 1 refinement
  passes (chunk solve rounds; each round is one batched unit-triangular solve for the R stores and
  one for the confirmation store). "Exact" means to the float32 rounding of the solves (measured
  1e-7 relative); an optional early stop on the gate change is available but costs a device
  synchronisation per round, so the GPU configuration uses a fixed budget of rounds and measures the
  trained model's residual error instead (correctness gate ii). An independent verification pass found that
  a FIXED two passes were not enough once the gate had trained (relative error 0.2 at chunk 64 for
  the state arm, 1e-4 for the free arm: a hypothesis-specific confound), and a trained state arm
  measured here needed 8 passes at chunk 32 to reach 1e-5 and 16 to reach 1e-7. Rung 1 therefore
  uses chunk 16, where the iteration costs at most 15 small triangular solves per chunk, and the
  script measures the trained model's chunked-versus-sequential error on a held-out batch and
  voids any run above 1% (correctness gate ii). Measured on briefly trained arms (CPU, one
  thread): with the early stop, 4-6 rounds at every chunk size for the state arm and up to the full
  bound in one layer of a trained free arm; the rung-1 configuration therefore fixes 7 rounds and
  lets gate (ii) police it. The cost is real and is not specific to the states: any gate that
  reads the store pays it, Gated DeltaNet's input-only gate does not. At the rung-1 size on this
  CPU (batch 1, T = 2,048, one thread) a training step is about 16 s for the two-store state arm,
  12 s for the two-store free arm, 7 s with one round, and 3 s for the input-only gate at chunk 64;
  the layer is bound by the number of small operations (about two million per step), not by FLOPs,
  so the GPU figure must be measured (rung 1's probe), never extrapolated.
- Gradients of the chunked form agree with the sequential form to about 1e-5 relative (learned
  gates; the test bound is 1e-4) and 1e-7 (constant gates). The single-store form is frozen against
  a committed reference snapshot to 1e-6 (bit-identical except one float re-association in the
  chunked read of the evidence mass).
- The state slots shown to the gate are detached; the state features shown to the output
  projection are not, and each path can be switched off separately (`slot_path`, `feature_path`),
  which rung 1 uses for its two ablation arms.

### 3.6 What this design deliberately does not contain

Opponent channels by coordinate sign (proved inert); confirmation ROUTING as a fixed rule (loses;
the confirmation COUNT survives as a state); consolidation on the critical path (repository null);
adaptive compute or learned halting (phase 2's adaptive K never fired); any read of the connectome
adjacency; a global arousal gain upstream of a norm (cancelled exactly by RMSNorm, a theorem).

---

## 4. The evidence ladder and its decision gates

Every rung is pre-registered with a hash lock before it runs, has a SUPPORTED and a FALSIFIED
condition that exhaust the outcomes (M5), >= 5 seeds, the 2 seed-SD plus bootstrap-CI rule, and a
six-fresh-seed replication before any positive verdict is reported as supported.

| rung | what | status | SUPPORTED iff | FALSIFIED iff |
|---|---|---|---|---|
| 0a E0 | signal availability, numpy | DONE: a, zc, nconf admitted; dense evidence channel dead | (diagnostic) | (diagnostic) |
| 0b D1 | corpus regime | DONE: in scope, ~3 far persistent re-bindings / 2k tokens | (diagnostic) | (diagnostic) |
| 0c E1 | sparse random vs dense addressing at equal active rows, delta store, module level | DONE: SUPPORTED and REPLICATED on six fresh seeds (0.993 vs 0.368 at P = 128; the gain is the sparse code, not the state width; a learned expansion is marginally better than random) | sparse_random beats dense at P = 128 by > 2 SD, CI excl. 0 | otherwise (then the sparse variant is dropped from 3.4) |
| 0d E2 | K0 at module level: `learned_state` vs `learned_free`, equal parameters, four regimes (noise x outliers) | DONE: NULL (+0.0001, +0.3 seed-SD). The states did not convert; a single store cannot hold "the previous observation", which the decisive decision needs | state arm beats free arm by > 2 SD, CI excl. 0 | difference inside the seed spread: the admitted states carry nothing the gate needs on this task |
| 0e E3 | K0 on the two-trace substrate: learned per-trace gates and a learned readout over (slow, labile) traces, with vs without the states, against a state-matched single store | DONE at 80 steps: NULL (all learned arms co-located 9 points below the oracle ceiling: under-trained). E4 = the same test at 800 steps on fresh seeds: SUPPORTED and REPLICATED on six further seeds (+0.012 vs the nested control, +0.007 vs the state-matched store; the gain on outlier keys; about one ninth of the gap to the oracle) | state arm beats the free two-trace arm AND the state-matched single store by > 2 SD, CI excl. 0, on the decisive key classes | otherwise: the substrate, not the state, is the finding, and it goes to rung 1 as such |
| 1 | hybrid at d = 256 on token streams with re-binding and outliers BEYOND the 512 window, GPU, eight arms plus four ablations at the central setting | RUN in its third form (v3, `experiments/p3_s1_hybrid`): the hybrid form could not be learned in budget by any arm, so the test is memory-only on the two-store substrate E4 supported, T = 256, 24 keys, two outlier rates, five seeds, 95 runs. DONE: FALSIFIED on all three conjuncts. `state` against `free`: 0.793 / 0.777 and 0.758 / 0.752 at outlier rates 0.15 / 0.3 (+0.016, +0.006; 0.3 seed spreads, intervals through 0); `single_wide` 0.833 and 0.851 (ahead; -4.2 SD at 0.3); a 1.2 SD regression on contradiction-free accuracy at 0.3. A plain single store with half the memory is within the noise of `state` | on the decisive strata beyond the window (re-bound, after-outlier; equal-weighted) `state` beats `free` by > 2 SD, CI excl. 0, at >= 2 of 3 P AND >= 2 of 3 outlier rates; AND beats the rank-matched twin with CI excl. 0 at >= 2 of 3 P; AND no loss > 1 SD on the inert strata or the contradiction-free stream (v3's rule: the same three conjuncts at every rate, with `single_wide` as the second control) | any conjunct fails |
| 1b | added after E1: the Kenyon-cell address inside a trained Gated DeltaNet layer on the capacity form of the recall task, against GDN with the same parameters, GDN at equal compute, GDN at equal memory, and product keys (`experiments/p3_s1b_sparse`) | DONE: FALSIFIED under its rule (0.7 / 1.9 / 6.1 seed spreads at P = 32 / 64 / 128; the sparse arm higher at all 15 seed pairs, 0.997-0.9995 against 0.79-0.81); equal-memory dense and product keys reach the same ceiling a few thousandths higher; the exact sparse kernel is slower than the equal-memory dense step | sparse beats dense by > 2 SD, CI excl. 0, at >= 2 of 3 P (the equal-compute, equal-memory and product-key comparisons are reported with fixed rules, BEATS / LOSES / MATCHES / INCONCLUSIVE, outside the verdict) | otherwise |
| 2 | language modelling at 2k and 8k context (WikiText-103, then a FineWeb-Edu subset), ~22M non-embedding; primary metric: accuracy on persistent re-binding probes embedded in text (the D1 regime), perplexity as non-regression | NOT RUN: reserved by rules fixed in advance for a mechanism rungs 1 or 1b supported; neither was. Exploratory prototypes (34M, WikiText-103 and TinyStories, two seeds, not pre-registered) are reported as such in EXPERIMENT_PLAN.md | probe accuracy: `state` beats `free` by > 2 SD, CI excl. 0, at both contexts; perplexity within 1 SD | otherwise |
| 3 | 110M against a self-trained Gated DeltaNet twin, same data, same tokens, same tuning budget | not reached | as rung 2 plus the standard zero-shot suite within noise | otherwise |

Correctness gates that run before any hypothesis at rungs 1-3: (i) the exact reduction (slots
zeroed, gamma tied) reproduces a reference Gated DeltaNet's plain-MQAR accuracy within 0.5 seed-SD
and its forward pass bitwise; (ii) a dead-feature monitor (fraction of tokens with a_t = 0, with
nconf_t = 0, with zc_t = 0) is logged every step, and a degenerate distribution (> 0.9 dead for
100 steps) is a stop condition, not a post-hoc discovery; (iii) wall-clock parity within 1.15x of
the twin (dense) or 0.5x (sparse) is measured on the standalone layer before any training.

Arms at rung 1 (all the same code): `state` (slots live), `free` (slots zero: the primary control),
`twin_dk` (free with d_k raised to match state values: rank-matched), `hebb_column` (free with
one extra value channel written without error correction by ||e||: the red team's minimal
alternative), `best_fixed` (beta swept, no learned gate). If `hebb_column` matches `state`, the
finding is "the delta rule's error correction destroys accumulation; one uncorrected channel
restores it", which is reported as the result.

---

## 5. Predictions the design makes before rungs 1-3 run

1. E2 (module level, noise present): `learned_state` beats `learned_free` most on the outlier
   regimes (p_outlier = 0.1), where the adoption-versus-outlier decision needs history, and least
   or not at all at sigma = 0.3, p_outlier = 0, where E0 says the free intermediates suffice.
   OUTCOME: falsified. No difference in any regime (0.8597 vs 0.8591 on the outlier regimes).
2. If E2 is null, rung 1 is not run; the programme's result is "a provably available signal that a
   learned gate does not exploit at module level", published as such. OUTCOME: E2 was null; rung 1
   is not run on those states. The per-class results point at the substrate (two traces hold the
   previous observation; one store cannot), and E3 tests that before anything else is funded.
3. At rung 1 the effect, if present, appears only on streams whose contradictions fall beyond the
   512-token window; inside the window the attention layer resolves them verbatim, and the arms tie.
   OUTCOME: not testable as written. The hybrid form of rung 1 (v2) could not be learned in budget
   by any arm (EXPERIMENT_PLAN.md, "Learnability"); rung 1 v3 is memory-only, so there is no window.
   On it the effect is absent: the states add +0.016 and +0.006 at the two outlier rates (0.3 seed
   spreads each, intervals through 0), a single store with the same memory is ahead of the
   two-store design (0.833 and 0.851 against 0.793 and 0.758), and the verdict is FALSIFIED on all
   three conjuncts.
4. On dense keys `a` stays dead (E0); the sparse variant is the only carrier of the ATTEST
   evidence channel, and its state cost is n/k times larger. If the sparse variant is what wins,
   the comparison set becomes Product-Key Memory and Memory Layers, and the write-up says so.
   OUTCOME: the sparse address was ahead on capacity at every seed pair (rung 1b: 0.997-0.9995
   against 0.79-0.81), though FALSIFIED under its two-spread rule; the comparison set was product
   keys and the equal-memory dense store as this
   prediction required, and both reach the same ceiling a few thousandths higher (0.999-1.000;
   LOSES by the fixed rule, by 0.0004-0.0027). The write-up says so.
5. `nconf` contributes more than `zc` and `a` combined (E0 ranking); if the per-slot ablation at
   rung 1 shows otherwise, the E0 proxy was wrong in a way that must be diagnosed before rung 2.
   OUTCOME: not testable. Rung 1 ran per-path rather than per-slot ablations (`state_slots_only`
   0.778, `state_feats_only` 0.760, `single_state` 0.763, at the lower outlier rate), and with the
   states together adding nothing beyond the noise there is no contribution to rank among them;
   rung 2 was not reached.

---

## 6. Verification plan (what "NASA grade" means here)

- A written specification (this document, section 3.3) with every equation; the implementation is
  checked line by line against it, as `flyai` was against its v0.2 spec (docs/aif/VERIFICATION.md).
- Property tests: equal parameter count between `state` and `free`; bitwise reduction to the twin;
  non-negativity of Zc, Z, sigma; reset of the confirmation vector on contradiction; the confirmation
  criterion's adaptation to noise (a consistent repeat under sigma = 1 is still counted); dead
  fractions on synthetic streams below 0.2.
- Every experiment directory carries `PREREG.md` hash-locked at the first run, `SUMMARY.md` and
  `summary.json` produced by the harness, and mechanical verdicts in RESULTS.md computed by
  `scripts/build_results.py`; no verdict is typed by hand.
- Instruments are validated before use: D1 v1 was retracted the moment its own sanity checks
  failed, and the retraction is in the record; E0's delta rule was checked against the closed form
  before the run.
- A finding is reported with its scale attached; nothing is extrapolated; the prohibitions of
  docs/RESEARCH_DESIGN.md (no model is "a fly brain", no state-of-the-art claim without matched
  controls, contact counts are anatomy) apply verbatim.

---

## 7. Odds, stated before E1/E2 finish

| event | probability |
|---|---|
| E2: the state arm beats the free arm by > 2 seed-SD at module level | 0.4 -> measured: NULL |
| E3/E4: the two-trace substrate with a learned readout beats the state-matched single store, and the states add to it | 0.3 (the substrate) -> measured: NO; 0.15 (the states) -> measured at 800 steps: YES, small (+0.007 to +0.012), REPLICATED |
| rung 1 passes given E3 passes | 0.4 -> FALSIFIED (E3 itself was null; E4 passed and was taken to rung 1): the states add +0.016 and +0.006, 0.3 seed spreads, and a single store with the same memory is ahead |
| rung 1b (added after E1): the Kenyon-cell address beats Gated DeltaNet in a trained model | not stated in advance; a one-seed pilot predicted a large gain -> measured: higher at all 15 seed pairs, FALSIFIED under the 2-spread rule; equal-memory dense and product keys reach the same ceiling a few thousandths higher |
| rung 2 recall-probe win in a trained LM given rung 1 | 0.5 -> not run: neither rung earned it under the rules fixed in advance; exploratory prototypes show the memory does not recall far facts at that size |
| a defensible pre-registered claim in the named regime at rung 3 | 0.05-0.08 -> not reached |
| any perplexity-per-FLOP win at scale | < 0.02 -> not reached; the exploratory prototypes show a perplexity loss for the sparse address |
| the programme yields a publishable result either way (nested control, measured signal, measured regime, closed middle band) | ~0.8 -> this is the result: two mechanisms measured to their ceilings against nested controls, with the regime in which each matters and does not |

The base rate is the repository's own: three fly-derived mechanisms falsified by the panel in
minutes, two phases null. These odds weighted that base rate, and the outcomes landed on the
pessimistic side of them. The decision points were: whether to fund rung 1 on a GPU if E2 passes
(about 40 GPU-hours at d = 256; funded after E4 on a different form), and whether to fund rung 2
(about 300 GPU-hours) if rung 1 passes (not funded; a 20-GPU-hour exploratory prototype was run
instead, labelled as such).
