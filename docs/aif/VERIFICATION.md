# Verification of the FlyAI implementation against the v0.2 design

*A formal check of the phase-2 code (flyai/, experiments/aif_*, flybrain/harness, scripts/build_results.py,
benchmarks/) against docs/aif/DESIGN.md §2, docs/aif/MODEL_SPEC.md §6-7, the six PREREG files and
benchmarks/PREREG_tier2.md. Run after AIF-H1 had finished and before AIF-H2 started; AIF-H2 was
stopped two minutes into its first run and restarted on the fixed code.*

## Method

1. **Requirement extraction.** Two independent readers turned the design documents into 227
   numbered, testable requirements (110 model, 117 protocol).
2. **Audits.** Twelve independent auditors, one per module or lens (primitives, memory, model +
   attractor, block + LM, experiments H1-H3, experiments H4-H6 + decoding/environment, autograd,
   numerics + batch mixing, harness + results builder, benchmarks + HF wrapper + configs, tests,
   dynamic invariant testing), each reading its files completely and recording pass / fail per
   requirement plus independent findings. 84 findings were raised.
3. **Adversarial verification.** Every finding of the four core-module audits was re-examined by a
   verifier whose default stance was that the finding is wrong (34 confirmed, 2 documented as
   intentional, 0 refuted). The remaining audits' verifiers were cut off by a usage limit; those
   findings were verified by hand, by reading the code with its callers and by running a
   reproduction (the property checks below).
4. **Property checks** written and run by the author of this document, independent of the
   auditors (table below).
5. **Fixes**, one regression test per fix where an invariant could be stated, and a full test run.

## Property checks (all pass after the fixes)

| check | result |
|---|---|
| DaleCore: spectral radius at gain 1; E/I column imbalance after construction and after rebalance | 1.0000; 0.0 |
| Dale sign per source row; inhibitory units = round(0.308 n); stratified fan-in exact (32 sources, 10 inhibitory); no self-loops | ok |
| Lyapunov estimate monotone in gain (0.3 / 1.0 / 6.0 -> -0.37 / -0.15 / +0.05); identical with and without no_grad | ok |
| dithered controller converges from gain 0.2 and 4.0 to within 0.06 of the target and to the same gain (+/- 35 %) | ok |
| block: checkpoint (chunk 4, chunk 2) vs single chunk, logits and all parameter gradients, regulation off | 0.0 |
| block: checkpoint vs plain chunked loop **with regulation on** (gain moves at every boundary) | identical after fix F-9 |
| block: delayed_write chunk 1 vs sequential, train mode, outputs and gradients | 1e-16 |
| block: closed-form fusion equals softmax_l(log pi_l + c_l) fusion computed by hand | 1e-6 |
| batch-row independence (categorical model, train and eval, logits and every state entry) | 0.0 |
| GammaPrecision: 20,000 zero-error updates stay finite and at the upper bound | ok |
| HMM forward-algorithm predictive vs brute-force path enumeration (S=3, V=4, T=5) | 2.5e-8 |
| task semantics: every MQAR / retention target equals the current binding of its query key | ok |
| HF wrapper: logits vs raw model; incremental decode; prefix + rest; greedy generate | 0.0 / 2.4e-7 / 1.8e-7 / identical |
| EFE decoding at beta_e = 0 equals greedy decoding; training mode restored afterwards | ok |
| FLOPs/token (fwd+bwd, matmul): K3 1.42M, K3 fixed 1.37M, K0 1.09M, K0_wide 1.88M | measured |

## Findings and their status

Severity: **blocker** = an experiment result or the GPU run would be wrong or misleading;
**major** = a real correctness or spec deviation; **minor** = robustness or reporting. Every fix
is in the commits after `dfc1429`; "documented" means recorded in LIMITATIONS.md / RESULTS.md.

### Blockers (all fixed)

| # | finding | fix |
|---|---|---|
| F-1 | AIF-H2 / AIF-H4 drivers scored FlyAI's *before-observation* logits against targets placed at the query position (the phase-1 convention for causal models whose logits at t include x_t): the model could not see the query; every arm would have scored at chance | `flyai.tasks.same_index_logits` (one extra filler step, first prediction dropped); PREREG change logs; regression test |
| F-2 | GPU configs (50M / 125M / 350M) set `"arousal": true` while the block expects a mode string; the string test `arousal in ("both", "pi0")` was silently false, so arousal was OFF in the hypothesis arm and `ar_w`, `ar_b` were dead parameters | configs set `"both"`; `BlockConfig` / `FlyAIConfig` validate every mode and raise on anything else; test |
| F-3 | `--gdn_key` defaulted to 128 in the training script and overrode the head-dim default of `make_lm`, so the GDN twin actually trained was not the pre-registered one | default `None` -> head dim; `gdn_key_used` recorded in each run's config |

### Major (all fixed unless marked)

| # | finding | fix |
|---|---|---|
| F-4 | Precisions were unbounded ratios `a/b`; `b` decays geometrically and underflows to 0 in float32 after a few thousand low-error tokens, so `a/b` -> inf and every downstream log / softmax -> NaN. MODEL_SPEC §6 and LIMITATIONS.md claimed a clamp [1e-3, 1e3] that did not exist | `GammaPrecision.ratio()`: floored `b`, ratio clamped to [1e-3, 1e3]; every consumer (model, block, memory gate) reads through it; test |
| F-5 | Fixed-gain arms of AIF-H4 would have reported a Lyapunov exponent of 0.0 (the estimator only ran under regulation) although the PREREG promises the estimate for every run | estimate in every condition, act only when regulating (model and block); test |
| F-6 | Extremum-seeking controller estimated the slope sign from one EMA increment against one gain move; noise flipped the sign and the gain stepped away from the target (AIF-H1 diagnostics: gain drifting to 2.2 while lambda fell to -0.24); at the clamp bounds the estimate froze | dithered extremum-seeking controller (square-wave dither +/- 10 %, demodulated single-step increment, EMA slope, dead-band); tightened convergence test |
| F-7 | The `const` write gate was `sigmoid(log gamma)` under the H2 default arousal mode, i.e. not constant | exactly 0.5, never modulated; unknown gate names raise; test |
| F-8 | AIF-H5's entropy-threshold control read the value slice of the prediction for the *next transcript token* ('?' or '='), not the model's answer distribution | the state is advanced through a hypothetical '=' and the value slice of the prediction at the answer position is read; PREREG change log |
| F-9 | Checkpointed chunks were recomputed in backward with the gain the controller had moved to *after* the forward, so under regulation the backward pass differentiated a different forward | the chunk's gain is passed into the checkpointed function as a tensor; test |
| F-10 | `--compile` with `delayed_write` mutated a Python list inside the compiled step, forcing a recompile per token and a silent fall-back to eager | `step` returns the proposal; the chunk loop collects it outside the compiled function |
| F-11 | Harness: a diverged (NaN) seed at the first learning rate won the LR selection through NaN comparison; a run that diverged *after* its first evaluation kept a finite metric | NaN counts as worst; a diverged run's metrics are NaN; tests |
| F-12 | Results builder silently dropped a missing pre-registered control from the verdict ("beats BOTH" could pass with one); AIF-H6a's absolute threshold, AIF-H2's pre-rebind guard, AIF-H4's `fixed_1.0` partial-null clause and AIF-H5's thresholds (a)-(c) were never evaluated mechanically; the replication rule was only a label | missing control -> INCOMPLETE; every clause evaluated; `--variant replication` grid with its own summary reported side by side; H5 verdict computed by the driver |
| F-13 | Comparisons paired seeds by list position (a missing seed mispairs silently); single-seed comparisons gave `effect_in_sd = inf` and could pass; with < 3 seeds the bootstrap CI is degenerate | pairing by seed id; `n < 3` never passes; test |
| F-14 | PREREG "K = 1" (H2-H5) and "2 inference steps" (H6) were run as *adaptive* K <= K_max | fixed K in every driver; K_mean logged in H6 |
| F-15 | Benchmarks: `run_tier2a.sh` used `model.pt` as the finished marker although it is written at every eval (a crashed run was skipped as complete); LR-sweep runs shared one directory; FLOPs/token for FlyAI included the checkpoint recompute (absent for the twins); the recurrent state size was not logged; validation used 8 batches instead of the 64 documents; under DDP each rank drew its own validation documents (cross-rank leakage) and all-reduced every micro-batch | `done` marker; LR in the run name; FLOPs measured with checkpointing disabled; state size logged; whole validation set; fixed-seed validation set hash-excluded from every training stream; `no_sync` |
| F-16 | Under `--bf16` the GDN twin's fast-weight state was bf16 for the whole sequence while FlyAI's state stayed fp32 (asymmetric numerics) | fp32 state in both GDN paths |
| F-17 | HF wrapper ignored `attention_mask`: left-padded batches (lm-eval `generate_until`) fed pad tokens through the recurrent layers first | left padding raises; right padding warns; the recall suite runs at batch size 1 |
| F-18 | Tier-2 PREREG threshold named arms that do not exist (`mb_dense_expanded`, `K0`); several ingredient arms of the pre-registered ladder had no config entry | phase-2 revision recorded in PREREG_tier2.md (unrun, unlocked); arms added to the 80 GB configs; `lr_sweep` encoded |
| F-19 | Extra parameter `c_level` was dead in the learned-gate arm and the learned gate was 83k parameters per block (not "at equal parameters") | per-group shared gate (867 parameters at d = 768); `c_level` created only for the arms that use it; test |
| F-20 | `precision_tracking` (H6) paired the precision *after* observing o_t with the predictive for o_t | precision entering step t |
| F-21 | AIF-H5: no divergence guard (a NaN model would have been decoded by all policies), no config / env / commit record per seed | guard, provenance, exclusion of diverged seeds from the verdict |
| F-22 (documented) | Adaptive K never stops early at tol 0.02: `flyai_K3` and `flyai_K3_fixed` are the same computation in AIF-H1 | LIMITATIONS.md, RESULTS.md |
| F-23 (documented) | `flyai_K0_wide` is over-provisioned (1.3x FLOPs, 1.9x parameters), not compute-matched | LIMITATIONS.md, RESULTS.md |
| F-24 (documented) | `c_NLL` bounds the categorical curvature with respect to the normalised latent, not the latent (RMSNorm Jacobian omitted) | LIMITATIONS.md |

### Minor (fixed)

Fusion offsets `c_level` received weight decay; `NoveltyMBON` applied the recovery tick to firing KCs;
the level-3 Gamma tracker was updated every token although only the control arm reads it; the
K-loop raised in training mode under `no_grad`; `info_gain` evaluated the free energy without the
arousal factor that the model's own inference uses; `efe_decode` left the model in eval mode; the
dead `kwta` / `SparseExpansion` copies in flyai.primitives (the model's only k-WTA is
`MBMemory.code`) and the unused `key_v` / `key_i` state were removed; LM blocks seeded their fixed
random structures by layer index only (now `seed * 1000 + layer`); `--compile` acted only on CUDA
and in CUDA-graph mode (now any device, default mode; CUDA graphs opt-in); stale statements in
MODEL_SPEC §7, DESIGN.md row 10, ARCHITECTURE.md, README.md, benchmarks/README.md and
LIMITATIONS.md (regulation cadence, divisive normalisation, clamps, bootstrap CI width,
Gamma pseudo-counts, the arm names of the GPU thresholds) corrected.

### Recorded, not changed

* AIF-H1 ran on the code before F-4 / F-6: its conditions all regulate (F-5 irrelevant), its
  precisions could not have hit the bound in 96-token sequences (F-4 inert), and the old
  controller's behaviour is part of its result (gain and lambda diagnostics in `runs/`).
* The AIF-H1 PREREG says "4 rings x 16 units"; the code ran 8 (v0.2 default). Frozen; recorded.
* PREREG hash locks hash file contents; they can be removed by deleting `.prereg.sha256`. The
  protocol relies on the git history for that (every run records its commit and hash).

## Coverage

All 227 requirements were checked by at least one auditor; 131 requirement checks failed or were
not implemented before the fixes, all traceable to the findings above. Files read completely by
at least one auditor: every file under flyai/, experiments/aif_*, flybrain/harness/, benchmarks/
(scripts, configs, PREREG), scripts/build_results.py, scripts/make_figures.py, tests/test_flyai_*.
Tests: 63 pass after the fixes (18 new verification tests in tests/test_flyai_verification.py).

## What remains open

* The dithered controller is validated on a random-input core; whether lambda(g) under trained
  token drive has a crossing at the target or only a peak below it is what AIF-H4 and the GPU
  diagnostics will show.
* The level-0 precision loop of the residual block (residual after fusion) is bounded but
  self-referential; its behaviour at scale is a GPU-pilot measurement, not a proof.
* lm-eval task names in `run_lm_eval.sh` follow lm-eval-harness >= 0.4.5 and could not be checked
  against an installed copy here.
