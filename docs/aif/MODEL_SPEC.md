# FlyAI model specification (draft v0 — before design panel and literature revision)

The model is a *filter*: at every token it predicts, observes, and updates a hierarchy of
latent states by minimising variational free energy, with precisions inferred online,
fast-weight memory written by precision-gated prediction errors, homeostatic
normalisation in every population, and a recurrent core regulated to the edge of chaos.
Everything below is implementable in plain PyTorch and trained by backprop through the
unrolled dynamics; the *architecture* is where the biology enters, not the optimiser.

## 1. Objects

| symbol | meaning | size (CPU scale) |
|---|---|---|
| o_t | token at step t | vocab V |
| e_t = E[o_t] | embedding (level-0 observation vector) | d_0 = d |
| s^1_t | fast latent ("PN / sensory") | d_1 = d |
| s^2_t | episodic level: fast-weight memory read-out ("MBON") | d_2 = d |
| M_t | fast weights (KC -> MBON) | n_kc x d_v per head |
| s^3_t | slow belief state: J continuous attractors ("CX rings") | J x n_ring (angle + amplitude each) |
| pi^l_t | precision of level-l prediction error (inferred) | scalar per level (per head for M) |
| gamma_t | global arousal (neuromodulator) | scalar |
| lambda_t | running Lyapunov estimate of the recurrent core | scalar |

## 2. One token step (predict -> observe -> infer -> write)

**(a) Predict (top-down, before seeing o_t).** Each level proposes a prior mean from the
previous state:

    mu^3_t = R(s^3_{t-1}, v_t)                        ring dynamics with velocity v_t = tanh(W_v s^1_{t-1})
    mu^2_t = read(M_{t-1}, code(s^1_{t-1}))           memory prediction for the *next* content
    mu^1_t = f_core(s^1_{t-1}, g^2(mu^2_t), g^3(mu^3_t))   recurrent core with top-down input
    p(o_t | mu^1_t) = softmax(W_out * norm(mu^1_t))   the language-model prediction; its NLL is the training loss

**(b) Observe.** o_t arrives. The level-0 prediction error is the categorical error
    eps^0_t = onehot(o_t) - softmax(W_out mu^1_t)     (a genuine prediction error, not an embedding)

**(c) Infer (K steps of free-energy descent on s^1_t, s^3_t).** Initialise with amortised
guesses s^1 <- mu^1_t + A(e_t, mu^1_t) (a learned bottom-up correction; with K = 0 this
is the whole update and the model degenerates to an ordinary recurrent network). Then for
k = 1..K:

    F(s) = gamma_t * [ pi^0 * NLL(o_t | s^1) + pi^1/2 ||s^1 - mu^1_t||^2
                     + pi^2/2 ||P_2 s^1 - g^2(s^2)||^2 + pi^3/2 ||P_3 s^1 - g^3(s^3)||^2 ]
    s^l <- s^l - eta_l * dF/ds^l                        (eta_l learned per level)

K is adaptive: stop when F falls by less than a threshold, up to K_max (spend more steps
when surprised). Gradients of the training loss flow through these steps.

**(d) Update precisions (Gamma posterior with forgetting rho).**

    a^l_t = rho a^l_{t-1} + 1/2,   b^l_t = rho b^l_{t-1} + ||eps^l_t||^2 / (2 n_l),   pi^l_t = a^l_t / b^l_t
    gamma_t = sigmoid(w_g * log F_bar_t + c_g),  F_bar_t = running mean of F   (arousal: two learned scalars)

**(e) Write fast weights (mushroom body).** With sparse code k_t = kWTA(E * P_pn s^1_t)
(fixed expansion E, 5 % active) and target value u_t = W_u s^1_t (the content the memory
should have predicted):

    eps^M_t = read(M_{t-1}, k_t) - u_t
    M_t = alpha_t M_{t-1} - beta_t * k_t eps^M_t^T,   beta_t = sigmoid(a_M log pi^M_t + b_M)   (dopamine = inferred precision of the memory error)

**(f) Recurrent core with homeostasis and criticality.**

    h <- (1 - tau) h + tau * DN( W_in x + g * (mask . |W| . sign) h )
    DN(z) = relu(z) / (sigma + mean(relu(z)))                       divisive normalisation (APL / antennal lobe)
    sign: 30 % of units inhibitory (fixed), Dale's law
    lambda_t: largest Lyapunov exponent estimated online by propagating a unit perturbation
              delta through the step Jacobian (one JVP per step): lambda_t <- (1-r) lambda_{t-1} + r log ||J delta||
    g <- g - eta_g * lambda_t                                       homeostatic gain control toward lambda = 0 (no backprop needed)

**(g) Act (decoding, generation only).** Expected free energy over candidate tokens c in
the top-k of p(o_{t+1} | mu^1_{t+1}):

    G(c) = -log p(c | mu^1_{t+1})  -  beta_e * IG(c),   IG(c) = F_before - F_after(c) (free-energy drop if c were observed and inferred)
    emit argmin_c G(c) (or sample from softmax(-G/T))

## 3. What is fixed, learned, inferred

- Fixed (from the fly or by design): expansion E (fan-in 5), 5 % k-WTA, 30 % inhibitory
  fraction, ring topology (or the extracted 148-neuron matrix), DN nonlinearity.
- Learned by backprop: E_emb, W_out, W_v, W_in, |W|, g^l maps, A, P_l, W_u, eta_l, a_M,
  b_M, w_g, c_g, tau, alpha (decay), thresholds.
- Inferred online (no gradient, per sequence): s^l_t, M_t, pi^l_t, gamma_t, lambda_t, g.

## 4. Controls that make each ingredient falsifiable

| ingredient | ablation | matched baseline |
|---|---|---|
| K-step inference (AIF-H1) | K = 0 (amortised only) | same FLOPs: K = 0 with K_avg x wider core or K_avg blocks |
| inferred precision gate (AIF-H2) | learned sigmoid(w.x) gate; constant gate | Gated DeltaNet memory |
| arousal gamma (AIF-H3) | gamma = 1 | tuned constant precision |
| Lyapunov-regulated E/I core (AIF-H4) | no regulation; gain fixed at 0.5 / 1.5 | tuned dense GRU |
| EFE decoding (AIF-H5) | beta_e = 0 (greedy), nucleus | -- |
| whole model | all ablations off | Gated-DeltaNet hybrid at matched params and FLOPs |

## 5. Scale path

The token step is O(K * d^2 + n_kc_active * d_v + J n_ring) per layer and parallelises
over batch; it is sequential in time like any recurrent model. At GPU scale the same
block replaces the fast-weight slots of a 3:1 hybrid decoder (attention every 4th block),
which is how it would be compared to Gated DeltaNet / Mamba-3 at 125-350 M parameters.


## 6. Revisions after the literature sweep (v0.1; before any AIF experiment ran)

| item | v0 | v0.1 | reason |
|---|---|---|---|
| core nonlinearity | divisive normalisation | tanh; DN kept only at the expansion stage | DN cancels the gain, so lambda cannot be regulated through g (measured here) |
| core init | unit fan-in weights | E/I balanced per target, spectral radius 1 at g = 1, log-parametrised magnitudes | Rajan-Abbott outlier eigenvalue; Cornford 2021 on naive sign constraints |
| homeostasis | proportional controller to lambda = 0 | extremum-seeking controller to lambda = -0.02 | driven homeostasis settles subcritical; sign of d lambda / d g must be estimated |
| inference step | eta / (pi_1 + pi_3) | eta / (gamma (pi_0 + pi_1 + pi_3) + 1) | step exploded when precisions hit their floor |
| precisions | unbounded | clamped [1e-3, 1e3], multipliers exp([-3, 3]) | entropy collapse / cross-level imbalance (Zhai 2023; deeper-PCN 2025) |
| attractor units | tanh(relu) (amplitude saturates) | threshold-linear, soft ceiling, slow amplitude leak | certainty must live in amplitude (Kutschireiter 2023) |
| EFE test | greedy control only | + entropy-threshold policy | the obvious non-AIF heuristic must be beaten, not just greedy |


## 7. v0.2 (after the design panel; docs/aif/DESIGN.md) — what the code now does

* **Categorical model (flyai/model.py).** Prior mean = precision-ratio fusion of the core
  prediction mu_1 and the attractor prediction mu_3, w_l = softmax_l(log pi_l + c_l) per
  group (G = 2 groups over the 128-d latent). pi_1 is a grouped Gamma posterior of
  (s - mu_1); pi_3 is the mean ring amplitude x exp(c) (Gamma tracker as control).
  Level 0: pi_0 = exp(log_temp) x gamma_t. Inference: amortised init, then K steps with
  step eta / (pi_0 c_NLL + pi_prior + 1), c_NLL = 0.5 sigma_max(W_out)^2. Arousal
  gamma_t = exp(clamp(w (log S_t - log log V) + c, log 0.25, log 4)) from the running token
  surprise S_t; it scales pi_0 and enters the write-gate logit as log gamma. The level-2
  tracker is gone.
* **Memory (flyai/memory.py).** One PN projection, one fixed expansion (fan-in 7 of 61 PN
  channels, sqrt rule), one k-WTA (5 %); H compartments with their own M_h, u_h, gate and
  decay prior alpha_h = 1 - 1/tau_h, tau_h in {4, 16, 64, 256}. Delayed key: read with
  code(s_{t-1}), write (code(s_{t-1}), u(s_t)). Gate 'precision' = sigmoid(a log(pi_slow
  ||eps||^2 / n) + b + log gamma), rho_slow = 0.98. Novelty MBON exposed as `nov`.
* **Residual block (flyai/block.py).** Closed-form fusion s* = w_0 x + w_1 mu_1 + w_3 mu_3
  with grouped precision ratios (G = n_heads); pi_0 from a Gamma posterior of (x - s*) x
  gamma; 'learned' gate and 'loop' arms retained; chunked activation checkpointing
  (64 tokens); the Lyapunov estimate and the homeostatic gain step run once per chunk, at
  the chunk boundary and outside the checkpointed region (never during recompute); the
  categorical model regulates every 8 tokens. No amortised network, no dense g0.
* **Core (flyai/primitives.py).** tanh, stratified fan-in (30.8 % inhibitory sources per
  unit), Rajan-Abbott balanced, spectral radius 1 at gain 1, log-parametrised magnitudes,
  extremum-seeking gain control toward lambda = -0.02, re-balance projection after every
  optimiser step, no weight decay on log-parameters (flyai.primitives.param_groups).
  The Lyapunov exponent is estimated in every condition (it is a reported diagnostic);
  the gain is acted on only when regulation is enabled.
* **Precision tracker (flyai/primitives.py GammaPrecision).** a_t = rho a_{t-1} + n/2,
  b_t = rho b_{t-1} + ||eps||^2 / 2 for a group of n error dimensions treated as n i.i.d.
  scalars sharing one precision: the same stationary precision as the v0 line in section 2(d)
  (which divides both increments by n), with a prior pseudo-count of n/2 instead of 1/2.
