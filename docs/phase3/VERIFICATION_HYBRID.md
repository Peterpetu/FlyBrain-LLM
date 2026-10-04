# Verification record: the system-level memory layer (`flyprinc/hybrid.py`) against its specification

*Method, as for the phase-2 model (docs/aif/VERIFICATION.md): an independent adversarial pass with
no write access to the repository, briefed to find deviations between docs/phase3/DESIGN.md 3.3-3.5
and the code, hidden bugs, and untested edge cases, with scratch measurements for every claim.
Findings are listed with their resolution. Nothing was dismissed without a measurement.*

## Findings and resolutions

| # | severity | finding (evidence) | resolution |
|---|---|---|---|
| 1 | blocker | In the rung-1 script the dead-feature monitor loop rebound the name of the layer config, so every run crashed at its end before writing metrics. | Fixed (loop variable renamed) before the report arrived; the report's independent discovery confirms the smoke run that caught it. |
| 2 | blocker | The fixed-rate reference arms received learning rate 0, so their backbone never trained (max parameter change 0.0 after 3 steps): a "tuning-advantaged" control at chance. | Every arm trains its backbone at the same sweep; only the gate is constant in those arms. |
| 3 | blocker/major | A fixed two refinement passes reproduced the sequential form only at initialisation (1e-3); after 200 training steps the state arm's layers deviated by 3-5% at chunk 16 and by 0.22 at chunk 64, the free arm's by 1e-4: a hypothesis-specific confound (the trained state arm was not the specified layer). Refine 8: 5e-5; 16: 4e-7. Reproduced after the first fix (cap of 16 passes): a free arm trained 1,500 steps at chunk 32 had two layers at 3e-2 and 1e-2 with 16 passes and at 3e-6 and 6e-7 with 32, i.e. the iteration needs up to chunk - 1 passes, and the trained-model gate (ii) caught it in the probe (1.8%). | The refinement is a Picard iteration on the gates that is exact within chunk - 1 passes (the intra-chunk dependency is causal: after p passes the first p + 1 tokens' gates are exact), with an early stop at a change below 1e-6; a trained state arm measured here needed 8 passes at chunk 32 and 16 at chunk 64, so rung 1 uses chunk 16; a test with a deliberately sharp gate requires < 1e-4; the script measures the trained model's chunk error on a held-out batch and voids any run above 1% (correctness gate ii). |
| 4 | major | The log-space decay exponentiated the full (t, s) matrix before masking, so the backward pass produced NaN for forget drives above 11.3 on the fastest head (0 x inf); the end-of-chunk factor G_end / G divided 0 by 0 at large drives; log(exp(-lam/tau)) was a lossy round trip. | Masked before exponentiation, end factor as exp(lg_end - lg), cumulative decay from -lam/tau directly; tests at forget drive 12-13 with finite gradients. |
| 5 | major (spec) | The design's content update erased twice (S(I - gamma k k^T) + beta k e^T with e containing S^T k) and was dimensionally inconsistent; the code was the correct textbook recurrence. | Specification corrected: S_t = alpha (I - gamma k k^T) S_{t-1} + beta k v^T; read after write; the decayed error in the agreement test, the tracker and the masses; RMSNorm on the reads; the floors and rho for dense keys stated. |
| 6 | major | A query drawn right after an unconfirmed re-binding was labelled `single` (27% of `single` queries had several prior presentations); 39% of re-bindings occurred at a key's last presentation and were never confirmed, against the generator's own docstring. | A re-binding is only drawn at an eligible presentation (not the last, not one followed by a query); the next presentation always confirms it. Invariants re-checked on 1,979 queries. |
| 7 | major (design) | The decisive strata were 17% of the natural far mix, so a 10-point effect there moved the primary metric by 1.7 points, below any seed-SD bar. | Primary metric is now the equal-weighted mean of the two decisive far strata; the natural mix and the inert strata are secondary, the latter with a no-regression clause. |
| 8 | minor | Nominal per-presentation rates overstated the effective ones (0.127 and 0.096 per placed presentation); the far share was 0.48 not 0.5. | Rates stated per eligible presentation; the measured composition is in the pre-registration (`lmtasks.rates`). |
| 9-11 | minor (spec) | The gate input omitted v_t; gates are sigmoid, the spec said softplus; the confirmation write uses the normalised projected value; the decayed error is used where the spec had the undecayed one. | v_t added to the gate input (cost table updated); the specification now states sigmoid gates, the normalised write and the decayed error. |
| 12 | minor | The detach set was broader than stated (v, e, logs detached in the gate input; beta and the error detached in the mass writes, so the zc feature has no gradient to the gate). | Documented as the gradient policy in the specification. |
| 13 | minor | `layer_cost` omitted the two P projections: the state arm's overhead is +14.5% MACs, not +8%. | Fixed; the design's cost table recomputed. |
| 14 | minor | `twin_dk` has 3.0% more parameters than the other arms (rank-matched, not parameter-matched). | Stated in the pre-registration with the numbers. |
| 15 | minor | The device id was stored in the run config, so a resume on another device would be refused. | Runtime facts kept out of stored configs. |
| 16 | minor | Correctness gates (ii)-(iv) were only logged, not applied. | (ii) and (iv) are applied by the script and void the run; (iii) is a manual gate before freezing (runbook). |
| 17 | minor | "Bitwise" reduction was tested at 1e-5; no tests for T not a multiple of the chunk, chunk 1, tied gamma, non-fresh state, learned gates with the evidence mass, gradients. | Wording corrected; all six cases added to the tests (13 tests, all passing). |
| 18 | note | `F.normalize` at an exactly zero projected read has an unbounded gradient. | eps = 1e-3 in the confirmation read direction. |
| 19-21 | note | Spec ambiguities (read timing, output norm, floors); the task's behaviour at T < 4; no slot-only / feature-only arm. | Specification amended; `slot_path` and `feature_path` switches added with two ablation arms at the central setting. |

## Verified correct by the pass (measured)

The chunked form equals the sequential reference given the gates at about 1e-7 relative in every
edge case tried (T = 61 with chunk 8, T smaller than the chunk, chunk 1, chunk = T, B = 1, H = 1,
each state set, forget drives 0 to 40, gamma != beta including gamma = 1 with beta = 0.2 and
gamma = 0, non-fresh initial states, tied gamma); the tracker recurrences are exact; parameter
gradients agree at 1e-7 (constant gates) and 2e-6 (learned gates, converged refinement); the
triangular solves are well conditioned for identical and sign-alternating keys at gamma = 1 and
chunk 64; the free arm structurally never reads or writes the states; parameter counts and
initialisation are identical across the equal-parameter arms; the model is causal; the task's
targets always equal a value shown before the query, far/near flags match sliding-window
visibility, events never overlap, and presentations precede their queries.

## Not verified

GPU execution and wall-clock parity (no CUDA in the verification environment); the behaviour of
the forget drive over a real 3,000-step run; the sparse address variant (3.4, not implemented).
