# FlyAI v0.2 architecture (what runs on the GPU)

```
 tokens ──► embedding ──► ┌─ Layer 1: FlyAI block ─┐ ──► ┌─ Layer 2: FlyAI block ─┐ ──► ┌─ Layer 3: FlyAI block ─┐ ──► ┌─ Layer 4: attention ─┐ ──► … ──► RMSNorm ──► tied head
                           │ pre-norm ▸ mixer ▸ +res  │      (3 : 1 layout, every 4th block is softmax attention with RoPE; the GDN twin puts Gated DeltaNet in the same slots)
                           │ pre-norm ▸ MLP   ▸ +res  │
                           └─────────────────────────┘
```

## One FlyAI block, one token (v0.2 closed form)

```
 state carried across tokens:  s1 (fast latent)   M_h (fast weights per compartment)   s3 (ring bank)   pi0, pi1, pi3 (grouped Gamma posteriors)   Sbar (running surprise)
                                                                                                                                           gain g, lambda (core homeostasis)

  x_t (block input, observed)
     │
     │            ┌──────────────────────── PREDICT (from the previous state) ─────────────────────────────┐
     │            │  mu2 = MBON read  = memory(M, key = KC-code(s1_{t-1}))            [episodic prediction]    │
     │            │  s3  = ring step(s3, velocity(s1_{t-1}))  ;  mu3 = g3(readout(s3))  [slow belief, amp = certainty]
     │            │  mu1 = core step(s1_{t-1}, td[mu2, mu3])      Dale-signed tanh core, gain g regulated to lambda ≈ -0.02
     │            └───────────────────────────────────────────────────────────────────────────────────────────┘
     ▼
  INFER (closed form; exact minimiser of the quadratic free energy)
     w = softmax over levels of ( log pi_l + c_l )   per group of channels        pi0 ×= arousal gamma_t
     s1_t = w0 · x_t  +  w1 · mu1  +  w3 · mu3
     │
     ├──► precision updates (no gradient):  pi0 ← Gamma(x_t − s1_t),  pi1 ← Gamma(s1_t − mu1),  pi3 = ring amplitude
     ├──► arousal:  Sbar ← running Mahalanobis surprise;  gamma_t = exp(clamp(w·log Sbar + c, log ¼, log 4))
     ├──► WRITE memory (delayed key):  eps = M[key(s1_{t-1})] − u(s1_t);  beta = σ(a·log(pi_slow·|eps|²) + b + log gamma_t)
     │                                  M ← alpha_h · M  −  beta · key ⊗ eps        (only the 5 % active rows; alpha_h = 1 − 1/tau_h, tau_h ∈ {4,16,64,256})
     ├──► novelty MBON:  familiarity of the current address (Bloom filter)  → exposed as `nov`
     └──► every 8 tokens (once per chunk in the block): Lyapunov JVP, homeostatic gain step
     ▼
  y_t = RMSNorm(s1_t)  → residual stream
```

Colour key used in the diagrams elsewhere: **frozen numerical priors** (expansion fan-in,
5 % coding level, 30.8 % inhibitory units, ring topology, decay priors), **learned by
backprop** (all projections, gate scalars, gains, offsets), **fast state rewritten every
token** (M, s3, precisions, arousal, gain).

## Cost per token per block (d = 768, 12 heads, 2048 KCs, 102 active)

| part | work | notes |
|---|---|---|
| core | one d×d masked matmul (fan-in 32) + tanh | sequential in t |
| memory read + write | PN projection (d→64), expansion (2048×64, fan-in 8), top-k, 2 gathers + 1 scatter over 102 rows × 32 | sparse; O(k·d_v) |
| ring bank | 8 rings × 8 units | negligible |
| fusion + precisions | elementwise | negligible |
| attention layers (every 4th) | standard SDPA over the prefix | chunk-parallel |

Training keeps O(T / 64) copies of the fast-weight state (per-chunk checkpointing);
the delayed-write arm keeps one per chunk and applies the chunk's rank-1 updates at once.
