# RESULTS

Version 0.3.0 (CHANGELOG.md). Phase 1-2 experiments and the phase-3 module ladder ran on 4 CPU cores with seeds {0,1,2} unless a table says otherwise; the phase-3 system-level grids (P3-S1, P3-S1b) ran on one RTX 4080 SUPER (16 GB) with seeds 10-14. Every table row = best learning rate for that condition by seed-mean of the primary metric; mean +/- SD over seeds; the comparison column is the seed-paired difference (hypothesis minus control, or as stated) with a 20,000-sample paired bootstrap 95% CI, and the effect in units of the larger seed SD. Differences inside the seed spread are nulls. Run directories: `experiments/<exp>/runs/<condition>/lr*/seed*/` (not committed; per-run tables in `runs_table.csv` where present).

Threshold verdicts below are computed mechanically from summary.json by `scripts/build_results.py`; interpretation is in REPORT.md.

## H5 — connectome graph as recurrent mask (calibration)

PREREG: `experiments/h5_graph_mask/PREREG.md`. Primary metric: validation bits/char (lower is better).

## H5: validation bits/char on TinyShakespeare (lower is better), best LR per condition

### N = 256

| condition | best LR | seeds | val_bpc mean +/- SD | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|
| dense_N_N256 | 0.01 | 3 | 2.4558 +/- 0.0049 | +0.1551 [+0.1516, +0.1580] | -20.8 |
| dense_param_N256 | 0.03 | 3 | 2.6826 +/- 0.0075 | -0.0718 [-0.0853, -0.0618] | +9.6 |
| er_N256 | 0.01 | 3 | 2.6052 +/- 0.0122 | +0.0057 [-0.0041, +0.0164] | -0.5 |
| real_N256 | 0.01 | 3 | 2.6109 +/- 0.0075 | (hypothesis) |  |
| shuffle_N256 | 0.01 | 3 | 2.6000 +/- 0.0187 | +0.0109 [-0.0131, +0.0267] | -0.6 |

### N = 512

| condition | best LR | seeds | val_bpc mean +/- SD | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|
| dense_N_N512 | 0.01 | 3 | 2.4219 +/- 0.0252 | +0.1008 [+0.0765, +0.1234] | -4.0 |
| dense_param_N512 | 0.03 | 3 | 2.6041 +/- 0.0096 | -0.0813 [-0.0899, -0.0698] | +8.5 |
| er_N512 | 0.01 | 3 | 2.5164 +/- 0.0030 | +0.0063 [+0.0027, +0.0113] | -2.1 |
| real_N512 | 0.01 | 3 | 2.5228 +/- 0.0018 | (hypothesis) |  |
| shuffle_N512 | 0.01 | 3 | 2.5208 +/- 0.0080 | +0.0020 [-0.0055, +0.0115] | -0.2 |

### N = 1024

| condition | best LR | seeds | val_bpc mean +/- SD | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|
| dense_N_N1024 | 0.003 | 3 | 2.4116 +/- 0.0453 | +0.0307 [-0.0286, +0.0762] | -0.7 |
| dense_param_N1024 | 0.01 | 3 | 2.5073 +/- 0.0030 | -0.0650 [-0.0825, -0.0503] | +4.4 |
| er_N1024 | 0.01 | 3 | 2.4352 +/- 0.0012 | +0.0071 [-0.0088, +0.0223] | -0.5 |
| real_N1024 | 0.01 | 3 | 2.4423 +/- 0.0146 | (hypothesis) |  |
| shuffle_N1024 | 0.01 | 3 | 2.4369 +/- 0.0075 | +0.0054 [-0.0016, +0.0127] | -0.4 |

### Pre-registered thresholds

- N=256: real vs shuffle effect -0.0109 = -0.6 seed-SD (n=3), CI includes 0 -> no real-vs-shuffle gap beyond threshold; real vs ER effect -0.0057 = -0.5 seed-SD (n=3), CI includes 0; real vs dense_param effect +0.0718 = +9.6 seed-SD (n=3), CI excludes 0; real vs dense_N effect -0.1551 = -20.8 seed-SD (n=3), CI excludes 0.
- N=512: real vs shuffle effect -0.0020 = -0.2 seed-SD (n=3), CI includes 0 -> no real-vs-shuffle gap beyond threshold; real vs ER effect -0.0063 = -2.1 seed-SD (n=3), CI excludes 0; real vs dense_param effect +0.0813 = +8.5 seed-SD (n=3), CI excludes 0; real vs dense_N effect -0.1008 = -4.0 seed-SD (n=3), CI excludes 0.
- N=1024: real vs shuffle effect -0.0054 = -0.4 seed-SD (n=3), CI includes 0 -> no real-vs-shuffle gap beyond threshold; real vs ER effect -0.0071 = -0.5 seed-SD (n=3), CI includes 0; real vs dense_param effect +0.0650 = +4.4 seed-SD (n=3), CI excludes 0; real vs dense_N effect -0.0307 = -0.7 seed-SD (n=3), CI includes 0.
- **Verdict:** real-vs-shuffle gap beyond threshold at 0 of 3 sizes (pre-registered: needs >= 2 of 3). **The graph carries no usable information beyond its degree sequence as a recurrent mask at these sizes.**
- Dense same-N layer beats the real graph at every size: yes (pre-registered expectation: yes; a 'no' unlocks H3/H4 follow-up).

## H1 — mushroom-body memory (sparse expansion, k-WTA, error-gated delta write) on MQAR

PREREG: `experiments/h1_mushroom_body/PREREG.md` (v2; v1 superseded before any cross-condition result, see its change log). Primary metric: MQAR accuracy at T=64 / 8 pairs (chance 0.048).

## H1 primary: MQAR accuracy (T=64, 8 pairs), best LR per condition

| condition | best LR | seeds | acc_in mean +/- SD | acc_pairs16_T128 | acc_T128 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|
| abl_dense_expanded | 0.003 | 3 | 0.9401 +/- 0.0958 | 0.8841 +/- 0.1394 | 0.9577 +/- 0.0666 |  |  |
| abl_hebb | 0.01 | 3 | 0.9417 +/- 0.0435 | 0.9316 +/- 0.0569 | 0.9310 +/- 0.0581 | +0.0397 [-0.0039, +0.0801] | +0.9 |
| abl_learned_gate | 0.01 | 3 | 0.9966 +/- 0.0059 | 0.9980 +/- 0.0034 | 0.9954 +/- 0.0063 | -0.0151 [-0.0215, -0.0054] | -2.6 |
| abl_no_kwta | 0.01 | 3 | 0.9149 +/- 0.0608 | 0.9229 +/- 0.0756 | 0.9583 +/- 0.0226 |  |  |
| attention | 0.003 | 3 | 0.8280 +/- 0.2193 | 0.7881 +/- 0.2342 | 0.8236 +/- 0.2166 | +0.1535 [-0.0156, +0.4004] | +0.7 |
| deltanet_matched | 0.01 | 3 | 1.0000 +/- 0.0000 | 0.9958 +/- 0.0020 | 1.0000 +/- 0.0000 | -0.0186 [-0.0215, -0.0156] | -6.3 |
| gdn_matched | 0.01 | 3 | 0.9979 +/- 0.0016 | 0.9932 +/- 0.0043 | 0.9987 +/- 0.0011 | -0.0164 [-0.0176, -0.0142] | -5.6 |
| linattn_matched | 0.01 | 3 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | -0.0186 [-0.0215, -0.0156] | -6.3 |
| mb | 0.01 | 3 | 0.9814 +/- 0.0029 | 0.9906 +/- 0.0023 | 0.9902 +/- 0.0078 | (hypothesis) |  |
| mb_connectome | 0.01 | 3 | 0.9772 +/- 0.0039 | 0.9876 +/- 0.0089 | 0.9857 +/- 0.0011 | +0.0042 [+0.0010, +0.0083] | +1.1 |
| mb_shuffled | 0.01 | 3 | 0.9782 +/- 0.0073 | 0.9909 +/- 0.0011 | 0.9922 +/- 0.0034 | +0.0033 [-0.0015, +0.0088] | +0.4 |

## H1a expansion ablations (hypothesis row = mb)

| condition | best LR | seeds | acc_in mean +/- SD | acc_pairs16_T128 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|
| abl_dense_expanded | 0.003 | 3 | 0.9401 +/- 0.0958 | 0.8841 +/- 0.1394 | +0.0413 [-0.0186, +0.1489] | +0.4 |
| abl_hebb | 0.01 | 3 | 0.9417 +/- 0.0435 | 0.9316 +/- 0.0569 |  |  |
| abl_learned_gate | 0.01 | 3 | 0.9966 +/- 0.0059 | 0.9980 +/- 0.0034 |  |  |
| abl_no_kwta | 0.01 | 3 | 0.9149 +/- 0.0608 | 0.9229 +/- 0.0756 | +0.0666 [+0.0166, +0.1353] | +1.1 |
| attention | 0.003 | 3 | 0.8280 +/- 0.2193 | 0.7881 +/- 0.2342 |  |  |
| deltanet_matched | 0.01 | 3 | 1.0000 +/- 0.0000 | 0.9958 +/- 0.0020 |  |  |
| gdn_matched | 0.01 | 3 | 0.9979 +/- 0.0016 | 0.9932 +/- 0.0043 |  |  |
| linattn_matched | 0.01 | 3 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |  |  |
| mb | 0.01 | 3 | 0.9814 +/- 0.0029 | 0.9906 +/- 0.0023 | (hypothesis) |  |
| mb_connectome | 0.01 | 3 | 0.9772 +/- 0.0039 | 0.9876 +/- 0.0089 |  |  |
| mb_shuffled | 0.01 | 3 | 0.9782 +/- 0.0073 | 0.9909 +/- 0.0011 |  |  |

## H1d real vs shuffled wiring (hypothesis row = mb_connectome)

| condition | best LR | seeds | acc_in mean +/- SD | acc_pairs16_T128 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|
| abl_dense_expanded | 0.003 | 3 | 0.9401 +/- 0.0958 | 0.8841 +/- 0.1394 |  |  |
| abl_hebb | 0.01 | 3 | 0.9417 +/- 0.0435 | 0.9316 +/- 0.0569 |  |  |
| abl_learned_gate | 0.01 | 3 | 0.9966 +/- 0.0059 | 0.9980 +/- 0.0034 |  |  |
| abl_no_kwta | 0.01 | 3 | 0.9149 +/- 0.0608 | 0.9229 +/- 0.0756 |  |  |
| attention | 0.003 | 3 | 0.8280 +/- 0.2193 | 0.7881 +/- 0.2342 |  |  |
| deltanet_matched | 0.01 | 3 | 1.0000 +/- 0.0000 | 0.9958 +/- 0.0020 |  |  |
| gdn_matched | 0.01 | 3 | 0.9979 +/- 0.0016 | 0.9932 +/- 0.0043 |  |  |
| linattn_matched | 0.01 | 3 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |  |  |
| mb | 0.01 | 3 | 0.9814 +/- 0.0029 | 0.9906 +/- 0.0023 | -0.0042 [-0.0083, -0.0010] | -1.1 |
| mb_connectome | 0.01 | 3 | 0.9772 +/- 0.0039 | 0.9876 +/- 0.0089 | (hypothesis) |  |
| mb_shuffled | 0.01 | 3 | 0.9782 +/- 0.0073 | 0.9909 +/- 0.0011 | -0.0010 [-0.0098, +0.0078] | -0.1 |
### Pre-registered thresholds

- H1 main: mb vs gdn_matched effect -0.0164 = -5.6 seed-SD (n=3), CI excludes 0; abl_dense_expanded vs mb: matches within seed spread. -> **NOT SUPPORTED**.
- H1a sparse expansion (no-kWTA): effect +0.0666 = +1.1 seed-SD (n=3), CI excludes 0 -> not falsified (ablation is worse).
- H1a sparse expansion (dense learned projection): effect +0.0413 = +0.4 seed-SD (n=3), CI includes 0 -> **falsified** (falsified if it matches mb).
- H1b error gate (vs learned gate): effect -0.0151 = -2.6 seed-SD (n=3), CI excludes 0 -> **falsified** (ablation matches or beats mb).
- H1c delta/depression write (vs Hebb): effect +0.0397 = +0.9 seed-SD (n=3), CI includes 0 -> **falsified** (ablation matches or beats mb).
- H1d real PN->KC wiring vs degree-preserving shuffle: effect -0.0010 = -0.1 seed-SD (n=3), CI includes 0 -> **NOT SUPPORTED: the measured pattern carries no extra information beyond its degree sequence here**.

## H2 — central-complex ring attractor for state tracking with length extrapolation

PREREG: `experiments/h2_ring_attractor/PREREG.md`. Primary metric: heading accuracy at 4x training length (T=128; chance 0.125).

## H2: compass state tracking, accuracy at 4x training length (T=128), best LR per condition

| condition | best LR | seeds | acc_T128 mean +/- SD | acc_T32_last2 | acc_T256_last2 | acc_T512_last2 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|
| attention | 0.01 | 3 | 0.3189 +/- 0.0119 | 0.8256 +/- 0.0025 | 0.2199 +/- 0.0061 | 0.1734 +/- 0.0028 | -0.0753 [-0.1232, -0.0428] | -2.1 |
| gdn | 0.03 | 3 | 0.2519 +/- 0.0585 | 0.5732 +/- 0.1323 | 0.1916 +/- 0.0315 | 0.1596 +/- 0.0156 | -0.0083 [-0.0702, +0.0762] | -0.1 |
| ring_canonical | 0.01 | 3 | 0.2156 +/- 0.0161 | 0.5213 +/- 0.0975 | 0.1689 +/- 0.0115 | 0.1485 +/- 0.0105 | +0.0281 [+0.0032, +0.0445] | +0.8 |
| ring_connectome | 0.01 | 3 | 0.2436 +/- 0.0360 | 0.6235 +/- 0.1474 | 0.1850 +/- 0.0166 | 0.1550 +/- 0.0113 | (hypothesis) |  |
| rnn_plain | 0.003 | 3 | 0.3897 +/- 0.0417 | 0.8004 +/- 0.0503 | 0.2609 +/- 0.0204 | 0.2021 +/- 0.0097 | -0.1461 [-0.1863, -0.1093] | -3.5 |
| rnn_unstructured | 0.03 | 3 | 0.6688 +/- 0.0468 | 0.9627 +/- 0.0049 | 0.4483 +/- 0.0337 | 0.3027 +/- 0.0201 | -0.4252 [-0.4466, -0.4084] | -9.1 |
### Pre-registered thresholds

- ring_connectome vs rnn_unstructured (same size, same velocity input): effect -0.4252 = -9.1 seed-SD (n=3), CI excludes 0 -> **NOT SUPPORTED (falsified: an unstructured layer of equal size matches or beats it)**.
- ring_connectome vs ring_canonical (idealised ring, positive control): effect +0.0281 (+0.8 seed-SD). Connectome matrix performs at least as well as the idealised ring.
- ring_connectome vs rnn_plain: effect -0.1461 = -3.5 seed-SD (n=3), CI excludes 0.
- ring_connectome vs gdn: effect -0.0083 = -0.1 seed-SD (n=3), CI includes 0.
- ring_connectome vs attention: effect -0.0753 = -2.1 seed-SD (n=3), CI excludes 0.

## H3 — cell-type dcSBM prior vs Erdos-Renyi

PREREG: `experiments/h3_sbm_prior/PREREG.md`.

_not run_

## H4 — Dale's law / E-I fraction and long-context stability

PREREG: `experiments/h4_dale_stability/PREREG.md`.

_not run_


# Phase 2 — FlyAI active-inference model

PREREG files: `experiments/aif_h*/PREREG.md`. Same reporting rules; every condition got the same 3-point LR sweep (deviation from the >= 5-point baseline rule, see LIMITATIONS.md).

## AIF-H1 — per-token free-energy inference vs amortised-only (val bits/char, lower better)

## AIF-H1: per-token free-energy inference (K steps) vs amortised-only, TinyShakespeare val bits/char

| condition | best LR | seeds | val_bpc mean +/- SD | K_mean | params | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|
| flyai_K0 | 0.01 | 3 | 2.8258 +/- 0.0352 | 0.0000 +/- 0.0000 | 135232.0000 +/- 0.0000 | -0.0259 [-0.0565, -0.0078] | +0.7 |
| flyai_K0_wide | 0.003 | 3 | 2.7931 +/- 0.0107 | 0.0000 +/- 0.0000 | 262080.0000 +/- 0.0000 | +0.0068 [-0.0112, +0.0273] | -0.6 |
| flyai_K3 | 0.01 | 3 | 2.7998 +/- 0.0106 | 2.9994 +/- 0.0011 | 135232.0000 +/- 0.0000 | (hypothesis) |  |
| flyai_K3_fixed | 0.01 | 3 | 2.8017 +/- 0.0125 | 3.0000 +/- 0.0000 | 135232.0000 +/- 0.0000 | -0.0019 [-0.0081, +0.0024] | +0.2 |
| flyai_K3_noattr | 0.01 | 3 | 2.8106 +/- 0.0127 | 0.0261 +/- 0.0042 | 127916.0000 +/- 0.0000 | -0.0108 [-0.0327, +0.0116] | +0.9 |
### Pre-registered thresholds

- flyai_K3 vs flyai_K0 (bits/char): effect +0.0259 = +0.7 seed-SD (n=3), CI excludes 0 -> within seed spread (null).
- flyai_K3 vs flyai_K0_wide (bits/char): effect -0.0068 = -0.6 seed-SD (n=3), CI includes 0 -> within seed spread (null).
- **Verdict:** NOT SUPPORTED (needs every listed control beaten beyond threshold and every clause above true).

Notes recorded by the verification pass (docs/aif/VERIFICATION.md): `flyai_K0_wide` (n = 192) is an over-provisioned control, not a matched one (measured 1.88 vs 1.42 MFLOPs/token and 262k vs 135k parameters against `flyai_K3`); the adaptive-K rule never stopped early (K_mean = 3.00 in every run), so `flyai_K3` and `flyai_K3_fixed` are the same computation; H1 ran on the code before the precision bounds and the dithered gain controller were added (both inert or absent in H1's conditions, see VERIFICATION.md).

## AIF-H2 — inferred-precision gate under non-stationarity (post-rebind recall)

_not run_

## AIF-H3 — arousal from running surprise on topic-switch streams (val bits/char)

_not run_

## AIF-H4 — Lyapunov-regulated core gain vs fixed gains (retention accuracy)

_not run_

## AIF-H6 — Bayesian wind tunnel: KL to the exact HMM posterior predictive (bits, lower better)

_not run_

## AIF-H5 — expected-free-energy decoding on ask-to-learn

PREREG: `experiments/aif_h5_efe/PREREG.md`. Thresholds (a)-(c) are evaluated mechanically by the driver on the held-out seeds; the verdict line is in the summary below.

_not run_


# Phase 3 — principle ladder (module level)

PREREG files: `experiments/p3_*/PREREG.md`; plan: docs/phase3/EXPERIMENT_PLAN.md. The memory module is measured directly on explicit write/query streams; learned controls are trained by backpropagation through the module. Compute is matched by active rows per event; state sizes are in every table.

## P3-E0 — signal availability of candidate per-address states (diagnostic, no training)

PREREG: `experiments/p3_e0_signal/PREREG.md`. Which candidate states carry information about an address's teaching history that a learned gate could not compute from the delta rule's own intermediates. Pre-registered admission rule: incremental 3-way accuracy > 0.05 on sparse keys.

Columns: control logistic 3-way accuracy (novel / clean / contradicted history; the novel class is trivially separable, so the informative contrast is clean-vs-contradicted, CvC); incremental accuracy from adding each state; CvC AUROC of the control's own cos(v, vhat) / ||vhat||; CvC AUROC of the contradiction mass a at the stated floor, at the oracle floor, of the familiarity-gated mass zc, of the confirmation count; dead fraction of a.

| key code | mode | beta | control acc | +a | +zc | +nconf | +log age | +all | AUROC CvC cos / vhat | a (stated floor) | a (oracle floor) | zc | nconf | dead a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sparse | rebind | 0.25 | 0.984 | -0.013 | -0.003 | +0.013 | +0.000 | +0.013 | 0.47 / 0.00 | 0.65 | 0.66 | 0.92 | 0.00 | 0.18 |
| sparse | rebind | 0.5 | 0.974 | -0.013 | -0.005 | +0.021 | +0.003 | +0.021 | 0.47 / 0.00 | 0.81 | 0.81 | 0.92 | 0.00 | 0.19 |
| sparse | rebind | 0.8 | 0.919 | +0.010 | +0.010 | +0.073 | +0.003 | +0.073 | 0.47 / 0.03 | 0.91 | 0.92 | 0.92 | 0.00 | 0.19 |
| sparse | rebind | 1 | 0.677 | +0.232 | +0.208 | +0.315 | +0.013 | +0.315 | 0.47 / 0.45 | 0.94 | 0.94 | 0.93 | 0.00 | 0.19 |
| sparse | cancel | 0.25 | 1.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.47 / 0.00 | 0.77 | 0.77 | 1.00 | 0.00 | 0.17 |
| sparse | cancel | 0.5 | 1.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | 0.48 / 0.00 | 0.91 | 0.91 | 0.98 | 0.00 | 0.17 |
| sparse | cancel | 0.8 | 0.997 | +0.000 | +0.000 | +0.003 | +0.000 | +0.003 | 0.48 / 0.00 | 0.97 | 0.97 | 0.98 | 0.00 | 0.17 |
| sparse | cancel | 1 | 0.635 | +0.307 | +0.307 | +0.365 | +0.034 | +0.365 | 0.48 / 0.42 | 0.98 | 0.98 | 0.98 | 0.00 | 0.17 |
| dense | rebind | 0.25 | 0.812 | +0.000 | +0.010 | +0.122 | +0.010 | +0.130 | 0.48 / 0.16 | 0.50 | 0.52 | 0.66 | 0.07 | 1.00 |
| dense | rebind | 0.5 | 0.797 | +0.000 | +0.013 | +0.148 | +0.000 | +0.156 | 0.48 / 0.21 | 0.50 | 0.55 | 0.66 | 0.06 | 1.00 |
| dense | rebind | 0.8 | 0.727 | +0.000 | +0.021 | +0.185 | +0.008 | +0.177 | 0.48 / 0.34 | 0.50 | 0.57 | 0.66 | 0.12 | 1.00 |
| dense | rebind | 1 | 0.669 | +0.000 | +0.052 | +0.208 | +0.010 | +0.214 | 0.49 / 0.46 | 0.50 | 0.57 | 0.64 | 0.17 | 1.00 |
| dense | cancel | 0.25 | 0.974 | +0.000 | +0.003 | +0.000 | +0.008 | +0.005 | 0.45 / 0.01 | 0.50 | 0.56 | 0.70 | 0.04 | 1.00 |
| dense | cancel | 0.5 | 0.906 | +0.000 | -0.005 | +0.029 | +0.018 | +0.036 | 0.45 / 0.05 | 0.50 | 0.61 | 0.72 | 0.06 | 1.00 |
| dense | cancel | 0.8 | 0.779 | +0.000 | +0.026 | +0.117 | +0.023 | +0.138 | 0.45 / 0.25 | 0.50 | 0.64 | 0.73 | 0.08 | 1.00 |
| dense | cancel | 1 | 0.638 | +0.000 | +0.102 | +0.240 | +0.023 | +0.258 | 0.46 / 0.47 | 0.50 | 0.64 | 0.73 | 0.10 | 1.00 |
| dense_aniso0.5 | rebind | 0.25 | 0.557 | +0.000 | -0.003 | +0.023 | +0.190 | +0.224 | 0.46 / 0.31 | 0.50 | 0.51 | 0.53 | 0.28 | 1.00 |
| dense_aniso0.5 | rebind | 0.5 | 0.529 | +0.000 | +0.003 | +0.047 | +0.195 | +0.253 | 0.46 / 0.34 | 0.50 | 0.51 | 0.53 | 0.27 | 1.00 |
| dense_aniso0.5 | rebind | 0.8 | 0.529 | +0.000 | -0.018 | +0.042 | +0.177 | +0.242 | 0.45 / 0.39 | 0.50 | 0.52 | 0.52 | 0.30 | 1.00 |
| dense_aniso0.5 | rebind | 1 | 0.521 | +0.000 | -0.008 | +0.047 | +0.195 | +0.247 | 0.45 / 0.43 | 0.50 | 0.52 | 0.52 | 0.32 | 1.00 |
| dense_aniso0.5 | cancel | 0.25 | 0.674 | +0.000 | +0.003 | +0.031 | +0.208 | +0.227 | 0.45 / 0.13 | 0.50 | 0.52 | 0.56 | 0.20 | 1.00 |
| dense_aniso0.5 | cancel | 0.5 | 0.625 | +0.000 | +0.005 | +0.042 | +0.216 | +0.232 | 0.46 / 0.18 | 0.50 | 0.53 | 0.57 | 0.21 | 1.00 |
| dense_aniso0.5 | cancel | 0.8 | 0.562 | +0.000 | +0.000 | +0.052 | +0.227 | +0.250 | 0.48 / 0.30 | 0.50 | 0.53 | 0.56 | 0.26 | 1.00 |
| dense_aniso0.5 | cancel | 1 | 0.534 | +0.000 | -0.008 | +0.062 | +0.201 | +0.240 | 0.49 / 0.39 | 0.50 | 0.53 | 0.56 | 0.30 | 1.00 |
| dense_aniso1.0 | rebind | 0.25 | 0.424 | -0.003 | +0.005 | +0.005 | +0.260 | +0.268 | 0.47 / 0.42 | 0.50 | 0.50 | 0.47 | 0.44 | 0.98 |
| dense_aniso1.0 | rebind | 0.5 | 0.440 | +0.003 | +0.008 | +0.008 | +0.247 | +0.258 | 0.46 / 0.42 | 0.50 | 0.50 | 0.47 | 0.47 | 0.98 |
| dense_aniso1.0 | rebind | 0.8 | 0.448 | +0.000 | +0.010 | +0.010 | +0.253 | +0.250 | 0.45 / 0.44 | 0.50 | 0.51 | 0.47 | 0.50 | 0.98 |
| dense_aniso1.0 | rebind | 1 | 0.456 | +0.003 | -0.010 | +0.008 | +0.237 | +0.240 | 0.46 / 0.46 | 0.50 | 0.51 | 0.47 | 0.47 | 0.98 |
| dense_aniso1.0 | cancel | 0.25 | 0.414 | -0.005 | +0.013 | +0.021 | +0.279 | +0.286 | 0.47 / 0.40 | 0.49 | 0.50 | 0.49 | 0.42 | 0.98 |
| dense_aniso1.0 | cancel | 0.5 | 0.424 | +0.000 | +0.000 | +0.016 | +0.260 | +0.294 | 0.49 / 0.40 | 0.49 | 0.50 | 0.49 | 0.45 | 0.98 |
| dense_aniso1.0 | cancel | 0.8 | 0.409 | +0.005 | +0.008 | +0.003 | +0.266 | +0.276 | 0.50 / 0.42 | 0.49 | 0.50 | 0.49 | 0.43 | 0.97 |
| dense_aniso1.0 | cancel | 1 | 0.385 | -0.008 | +0.018 | +0.000 | +0.250 | +0.273 | 0.51 / 0.44 | 0.50 | 0.50 | 0.49 | 0.44 | 0.97 |

H_B test (||S^T k|| after +v, -v; ||v|| = 3.186): beta 0.25: GLA 0.0000 / GDN 0.199, beta 0.5: GLA 0.0000 / GDN 0.797, beta 0.8: GLA 0.0000 / GDN 2.039, beta 1: GLA 0.0000 / GDN 3.186

**Pre-registered kill conditions.** K1 (best single state adds <= 0.02 accuracy) fires in 5 of 32 configurations (4 of 8 sparse); K2 (dense contradiction-mass AUROC < 0.7 at the stated floor) fires in 24 of 24 dense; K3 (dense dead fraction > 0.5) fires in 24; K4 (H_B residual > 0.1 ||v|| under the gated delta rule) fires for betas ['0.5', '0.8', '1'].
**Incremental 3-way accuracy on sparse keys (mean over beta and mode):** a +0.065, zc +0.065, nconf +0.099, logage +0.007; on dense keys: a -0.000, zc +0.010, nconf +0.061, logage +0.157. **States admitted to E2 (> 0.05 on sparse):** ['a', 'zc', 'nconf'].

## P3-D1 (v2) — persistent re-binding of key-like tokens in text (diagnostic; v1 retracted as an instrument defect, kept in `experiments/p3_d1_corpus/`)

PREREG: `experiments/p3_d1_corpus/PREREG.md`. Does text contain the regime the memory principles serve (an address re-bound beyond the 512-token attention window, with the new binding persisting), against a within-key permutation null.

| corpus | stratum | window | keys / 1k | repeats / 1k | consistent fraction | persistent re-bindings / 1k | far (> 512) persistent / 1k | gap median / p90 | persistent vs null | far persistent vs null | consistent vs null |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wikitext2 | cap | 512 | 92.5 | 52.1 | 0.25 | 3.28 | 0.000 | 59 / 295 | 1.32x (z +38.4) | 0.00x (z +0.0) | 1.89x (z +165.2) |
| wikitext2 | cap | 2048 | 92.5 | 63.2 | 0.26 | 4.43 | 0.774 | 91 / 861 | 1.34x (z +46.9) | 1.62x (z +19.9) | 1.91x (z +179.7) |
| wikitext2 | cap | 8192 | 92.5 | 69.6 | 0.25 | 5.26 | 1.497 | 130 / 2632 | 1.41x (z +54.3) | 1.90x (z +39.4) | 1.89x (z +175.3) |
| wikitext2 | rare | 512 | 210.2 | 66.2 | 0.25 | 3.46 | 0.000 | 66 / 324 | 1.20x (z +28.8) | 0.00x (z +0.0) | 1.79x (z +135.7) |
| wikitext2 | rare | 2048 | 210.2 | 87.2 | 0.25 | 5.07 | 1.066 | 116 / 1015 | 1.21x (z +29.0) | 1.36x (z +16.2) | 1.84x (z +159.3) |
| wikitext2 | rare | 8192 | 210.2 | 101.8 | 0.25 | 6.15 | 1.970 | 179 / 2747 | 1.25x (z +31.6) | 1.47x (z +24.0) | 1.80x (z +163.4) |
| ptb | rare | 512 | 199.9 | 49.2 | 0.18 | 1.80 | 0.000 | 48 / 227 | 1.22x (z +9.7) | 0.00x (z +0.0) | 1.82x (z +54.2) |
| ptb | rare | 2048 | 199.9 | 59.9 | 0.17 | 2.20 | 0.282 | 63 / 698 | 1.26x (z +13.1) | 1.86x (z +10.2) | 1.72x (z +51.4) |
| ptb | rare | 8192 | 199.9 | 85.0 | 0.16 | 3.43 | 1.414 | 193 / 5865 | 1.27x (z +21.9) | 1.58x (z +21.3) | 1.50x (z +50.4) |

**Decision rule.** far persistent re-bindings > 1 per 2048 tokens at W=8192 AND persistent count exceeds the permutation null by z > 3. Passes in: ['wikitext2.train.txt|cap|W8192', 'wikitext2.train.txt|rare|W8192', 'ptb.train.txt|rare|W8192']. **Language rung in scope under the pre-registered rule: True.**

## P3-E1 — sparse random addressing vs dense keys at equal active rows (recall accuracy at P = 128)

## P3-E1: one-shot recall accuracy at P = 128 stored pairs (chance 1/64), module level

| condition | best LR | seeds | acc_P128 mean +/- SD | acc_P32 | acc_P64 | acc_P256 | acc_P512 | state_numel | active_rows | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0 | 5 | 0.3683 +/- 0.0000 | 0.9245 +/- 0.0000 | 0.6764 +/- 0.0000 | 0.1899 +/- 0.0000 | 0.1025 +/- 0.0000 | 1024.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | +0.6250 [+0.6239, +0.6262] | +407.7 |
| dense_expanded | 0.001 | 5 | 0.3774 +/- 0.0032 | 0.9314 +/- 0.0031 | 0.6842 +/- 0.0027 | 0.1930 +/- 0.0013 | 0.1050 +/- 0.0003 | 16384.0000 +/- 0.0000 | 1024.0000 +/- 0.0000 | +0.6159 [+0.6132, +0.6178] | +193.8 |
| sparse_hebb | 0 | 5 | 0.7078 +/- 0.0045 | 0.9992 +/- 0.0011 | 0.9735 +/- 0.0005 | 0.2798 +/- 0.0032 | 0.1101 +/- 0.0013 | 16384.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | +0.2854 [+0.2825, +0.2882] | +64.1 |
| sparse_learned | 0.001 | 5 | 0.9976 +/- 0.0004 | 1.0000 +/- 0.0000 | 0.9993 +/- 0.0003 | 0.9617 +/- 0.0016 | 0.7901 +/- 0.0023 | 16384.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | -0.0043 [-0.0054, -0.0029] | -2.8 |
| sparse_nonorm | 0 | 5 | 0.9933 +/- 0.0015 | 0.9999 +/- 0.0003 | 0.9995 +/- 0.0005 | 0.9385 +/- 0.0012 | 0.7418 +/- 0.0013 | 16384.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] | +0.0 |
| sparse_random | 0 | 5 | 0.9933 +/- 0.0015 | 0.9999 +/- 0.0003 | 0.9995 +/- 0.0005 | 0.9385 +/- 0.0012 | 0.7418 +/- 0.0013 | 16384.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | (hypothesis) |  |
### Pre-registered thresholds

- sparse_random vs dense (accuracy at P = 128): effect +0.6250 = +407.7 seed-SD (n=5), CI excludes 0 -> beyond threshold.
- equal-state control `dense_expanded`: effect +0.6159 = +193.8 seed-SD (n=5), CI excludes 0 -> sparse_random better beyond seed spread (the gain is addressing, not state)
- descriptive `sparse_learned`: effect -0.0043 = -2.8 seed-SD (n=5), CI excludes 0 -> the fixed random expansion is NOT necessary (learned expansion better beyond threshold)
- P9 normalisation ablation: sparse_random vs sparse_nonorm effect +0.0000 = +0.0 seed-SD (n=5), CI includes 0
- P2 Hebb row: sparse_random vs sparse_hebb effect +0.2854 = +64.1 seed-SD (n=5), CI excludes 0
- **Verdict:** SUPPORTED at the discovery seeds (n = 5), PENDING the pre-registered six-fresh-seed replication before it may be reported as supported (needs every listed control beaten beyond threshold and every clause above true).

### Replication grid (`runs_replication/`, fresh seeds; reported side by side, never merged)

## P3-E1: one-shot recall accuracy at P = 128 stored pairs (chance 1/64), module level

| condition | best LR | seeds | acc_P128 mean +/- SD | acc_P32 | acc_P64 | acc_P256 | acc_P512 | state_numel | active_rows | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 0 | 6 | 0.3683 +/- 0.0000 | 0.9245 +/- 0.0000 | 0.6764 +/- 0.0000 | 0.1899 +/- 0.0000 | 0.1025 +/- 0.0000 | 1024.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | +0.6251 [+0.6245, +0.6256] | +836.1 |
| dense_expanded | 0.003 | 6 | 0.3784 +/- 0.0012 | 0.9311 +/- 0.0026 | 0.6787 +/- 0.0044 | 0.1919 +/- 0.0013 | 0.1044 +/- 0.0010 | 16384.0000 +/- 0.0000 | 1024.0000 +/- 0.0000 | +0.6150 [+0.6140, +0.6161] | +527.2 |
| sparse_hebb | 0 | 6 | 0.7056 +/- 0.0030 | 0.9990 +/- 0.0004 | 0.9736 +/- 0.0023 | 0.2756 +/- 0.0024 | 0.1110 +/- 0.0012 | 16384.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | +0.2878 [+0.2856, +0.2901] | +94.7 |
| sparse_learned | 0.003 | 6 | 0.9973 +/- 0.0007 | 1.0000 +/- 0.0000 | 0.9997 +/- 0.0002 | 0.9646 +/- 0.0024 | 0.7995 +/- 0.0016 | 16384.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | -0.0039 [-0.0045, -0.0033] | -5.2 |
| sparse_nonorm | 0 | 6 | 0.9934 +/- 0.0007 | 1.0000 +/- 0.0000 | 0.9997 +/- 0.0003 | 0.9366 +/- 0.0029 | 0.7447 +/- 0.0023 | 16384.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | +0.0000 [+0.0000, +0.0000] | +0.0 |
| sparse_random | 0 | 6 | 0.9934 +/- 0.0007 | 1.0000 +/- 0.0000 | 0.9997 +/- 0.0003 | 0.9366 +/- 0.0029 | 0.7447 +/- 0.0023 | 16384.0000 +/- 0.0000 | 64.0000 +/- 0.0000 | (hypothesis) |  |
- replication: sparse_random vs dense: effect +0.6251 = +836.1 seed-SD (n=6), CI excludes 0 -> beyond threshold.
- **Replication verdict:** REPLICATED (discovery seeds [0, 1, 2, 3, 4], replication seeds [5, 6, 7, 8, 9, 10]).

## P3-E2 (v2) — K0: learned write gate with vs without per-address states (overall accuracy, mean of 4 regimes)

## P3-E2 (v2): learned gate with vs without per-address states; fixed rules for reference (overall accuracy, mean over 4 regimes), module level

| condition | best LR | seeds | acc_mean mean +/- SD | acc_s0.3_o0 | acc_s1_o0 | acc_s0.3_o0.1 | acc_s1_o0.1 | acc_tag2_s0.3_o0.1 | acc_tag1_s0.3_o0.1 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| const_0.3 | 0 | 5 | 0.8903 +/- 0.0002 | 0.9042 +/- 0.0002 | 0.8924 +/- 0.0005 | 0.8945 +/- 0.0002 | 0.8701 +/- 0.0002 | 0.7798 +/- 0.0020 | 0.3862 +/- 0.0014 | +0.0069 [+0.0067, +0.0072] | +15.2 |
| const_0.6 | 0 | 5 | 0.8924 +/- 0.0001 | 0.9503 +/- 0.0002 | 0.9218 +/- 0.0003 | 0.8622 +/- 0.0004 | 0.8355 +/- 0.0004 | 0.0431 +/- 0.0023 | 0.6821 +/- 0.0016 | +0.0048 [+0.0044, +0.0051] | +10.5 |
| const_1.0 | 0 | 5 | 0.8384 +/- 0.0002 | 0.9553 +/- 0.0000 | 0.8119 +/- 0.0005 | 0.8610 +/- 0.0000 | 0.7256 +/- 0.0008 | 0.0008 +/- 0.0001 | 0.7009 +/- 0.0001 | +0.0588 [+0.0585, +0.0591] | +129.4 |
| learned_free | 0.003 | 5 | 0.8971 +/- 0.0001 | 0.9196 +/- 0.0012 | 0.9152 +/- 0.0009 | 0.8944 +/- 0.0015 | 0.8591 +/- 0.0011 | 0.5760 +/- 0.0241 | 0.5165 +/- 0.0075 | +0.0001 [-0.0002, +0.0005] | +0.3 |
| learned_state | 0.001 | 5 | 0.8972 +/- 0.0005 | 0.9204 +/- 0.0013 | 0.9146 +/- 0.0009 | 0.8942 +/- 0.0015 | 0.8597 +/- 0.0010 | 0.5812 +/- 0.0232 | 0.5116 +/- 0.0086 | (hypothesis) |  |
| precision | 0 | 5 | 0.8959 +/- 0.0001 | 0.9552 +/- 0.0000 | 0.9306 +/- 0.0003 | 0.8611 +/- 0.0000 | 0.8367 +/- 0.0003 | 0.0060 +/- 0.0003 | 0.6991 +/- 0.0003 | +0.0013 [+0.0010, +0.0016] | +2.8 |
| two_trace | 0 | 5 | 0.8504 +/- 0.0002 | 0.9157 +/- 0.0004 | 0.8191 +/- 0.0006 | 0.8939 +/- 0.0004 | 0.7728 +/- 0.0005 | 0.6642 +/- 0.0043 | 0.4517 +/- 0.0011 | +0.0468 [+0.0464, +0.0472] | +103.2 |
### Pre-registered thresholds

- learned_state vs learned_free (overall accuracy): effect +0.0001 = +0.3 seed-SD (n=5), CI includes 0 -> within seed spread (null).
- learned_state vs precision (reference): effect +0.0013 = +2.8 seed-SD (n=5), CI excludes 0 -> beyond threshold
- learned_state vs two_trace (reference): effect +0.0468 = +103.2 seed-SD (n=5), CI excludes 0 -> beyond threshold
- learned_state vs the best fixed rate (`const_0.6`, reference): effect +0.0048 = +10.5 seed-SD (n=5), CI excludes 0 -> beyond threshold
- P3 precision vs the best fixed rate (`const_0.6`): effect +0.0035 = +30.5 seed-SD (n=5), CI excludes 0 -> SUPPORTED
- P4 two_trace vs precision: effect -0.0456 = -192.5 seed-SD (n=5), CI excludes 0 -> not supported
- seed means: learned_free 0.8971, learned_state 0.8972, best fixed rate 0.8924 (a learned gate below the best fixed rate is reported as an optimisation failure of the arm, not as evidence about the state)
- **Verdict:** NOT SUPPORTED (needs every listed control beaten beyond threshold and every clause above true).

## P3-E3 — K0 on the two-trace substrate: learned readout and gates with vs without the states; state-matched single store (class-balanced accuracy)

## P3-E3: learned readout over two traces with vs without the states; state-matched single store (class-balanced accuracy, mean over 4 regimes), module level

| condition | best LR | seeds | acc_classes mean +/- SD | acc_mean | acc_s0.3_o0 | acc_s1_o0 | acc_s0.3_o0.1 | acc_s1_o0.1 | acc_tag2_s0.3_o0.1 | acc_tag1_s0.3_o0.1 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| learned_free | 0.003 | 5 | 0.6903 +/- 0.0007 | 0.8971 +/- 0.0001 | 0.9196 +/- 0.0012 | 0.9152 +/- 0.0009 | 0.8944 +/- 0.0015 | 0.8591 +/- 0.0011 | 0.5760 +/- 0.0241 | 0.5165 +/- 0.0075 | +0.0004 [-0.0002, +0.0010] | +0.4 |
| single_free_wide | 0.003 | 5 | 0.6919 +/- 0.0009 | 0.8985 +/- 0.0002 | 0.9196 +/- 0.0013 | 0.9181 +/- 0.0010 | 0.8948 +/- 0.0015 | 0.8615 +/- 0.0014 | 0.5747 +/- 0.0262 | 0.5186 +/- 0.0089 | -0.0013 [-0.0019, -0.0005] | -1.2 |
| tt_learned_free | 0.003 | 5 | 0.6896 +/- 0.0010 | 0.8969 +/- 0.0003 | 0.9198 +/- 0.0010 | 0.9154 +/- 0.0009 | 0.8940 +/- 0.0011 | 0.8585 +/- 0.0017 | 0.5706 +/- 0.0213 | 0.5173 +/- 0.0077 | +0.0010 [+0.0003, +0.0017] | +1.0 |
| tt_learned_state | 0.001 | 5 | 0.6906 +/- 0.0010 | 0.8972 +/- 0.0005 | 0.9206 +/- 0.0018 | 0.9152 +/- 0.0012 | 0.8938 +/- 0.0016 | 0.8592 +/- 0.0011 | 0.5668 +/- 0.0288 | 0.5196 +/- 0.0114 | (hypothesis) |  |
| two_trace | 0 | 5 | 0.6596 +/- 0.0005 | 0.8504 +/- 0.0002 | 0.9157 +/- 0.0004 | 0.8191 +/- 0.0006 | 0.8939 +/- 0.0004 | 0.7728 +/- 0.0005 | 0.6642 +/- 0.0043 | 0.4517 +/- 0.0011 | +0.0310 [+0.0299, +0.0321] | +30.2 |
### Pre-registered thresholds

- tt_learned_state vs tt_learned_free (class-balanced accuracy): effect +0.0010 = +1.0 seed-SD (n=5), CI excludes 0 -> within seed spread (null).
- tt_learned_state vs single_free_wide (class-balanced accuracy): effect -0.0013 = -1.2 seed-SD (n=5), CI excludes 0 -> CONTROL better beyond seed spread.
- tt_learned_state vs learned_free (E2 reference): effect +0.0004 = +0.4 seed-SD (n=5), CI includes 0 -> within seed spread (null)
- tt_learned_state vs the fixed two_trace (E2 reference): effect +0.0310 = +30.2 seed-SD (n=5), CI excludes 0 -> beyond threshold
- SUBSTRATE: tt_learned_free vs the state-matched single store: effect -0.0023 = -2.2 seed-SD (n=5), CI excludes 0 -> not beyond threshold (reported, not part of the K0-TT verdict)
- tt_learned_free vs the fixed two_trace: effect +0.0300 = +29.1 seed-SD (n=5), CI excludes 0 -> the substrate with a learned readout wins (reported, not part of the K0-TT verdict)
- post-hoc oracle ceiling (exact Bayesian filtering with the generator's rates, `scripts/e2_oracle_ceiling.py`): class-balanced 0.7810, natural mix 0.9292; the learned arms sit about 9 and 3 points below it, so the stream has headroom and the co-location of every learned mechanism is not an information ceiling
- **Verdict:** NOT SUPPORTED (needs every listed control beaten beyond threshold and every clause above true).

### Exploratory variant `long800` (800 steps, lr 3e-3, seeds 0-2; NOT pre-registered as such, reported beside the grid, never merged)

- single_free_wide: 0.6924, tt_learned_free: 0.6870, tt_learned_state: 0.6993.
- tt_learned_state vs tt_learned_free: effect +0.0123 = +7.8 seed-SD (n=3), CI excludes 0; tt_learned_state vs single_free_wide: effect +0.0070 = +8.2 seed-SD (n=3), CI excludes 0. Motivates the pre-registered E4 (fresh seeds).

## P3-E4 — E3's nested test at 800 training steps, fresh seeds (class-balanced accuracy)

## P3-E4: E3's nested test at 800 training steps, fresh seeds (class-balanced accuracy, mean over 4 regimes), module level

| condition | best LR | seeds | acc_classes mean +/- SD | acc_mean | acc_s0.3_o0 | acc_s1_o0 | acc_s0.3_o0.1 | acc_s1_o0.1 | acc_tag2_s0.3_o0.1 | acc_tag1_s0.3_o0.1 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_free_wide | 0.003 | 5 | 0.6929 +/- 0.0013 | 0.8987 +/- 0.0003 | 0.9183 +/- 0.0008 | 0.9177 +/- 0.0009 | 0.8959 +/- 0.0015 | 0.8630 +/- 0.0015 | 0.5949 +/- 0.0265 | 0.5116 +/- 0.0091 | +0.0058 [+0.0041, +0.0076] | +3.2 |
| tt_learned_free | 0.003 | 5 | 0.6875 +/- 0.0006 | 0.8963 +/- 0.0002 | 0.9225 +/- 0.0010 | 0.9111 +/- 0.0020 | 0.8904 +/- 0.0016 | 0.8612 +/- 0.0010 | 0.5137 +/- 0.0228 | 0.5343 +/- 0.0069 | +0.0112 [+0.0101, +0.0127] | +6.2 |
| tt_learned_state | 0.003 | 5 | 0.6987 +/- 0.0018 | 0.8981 +/- 0.0011 | 0.9274 +/- 0.0017 | 0.9056 +/- 0.0036 | 0.8937 +/- 0.0008 | 0.8659 +/- 0.0007 | 0.6217 +/- 0.0317 | 0.4804 +/- 0.0238 | (hypothesis) |  |
| two_trace | 0 | 5 | 0.6592 +/- 0.0002 | 0.8505 +/- 0.0003 | 0.9159 +/- 0.0002 | 0.8193 +/- 0.0010 | 0.8934 +/- 0.0003 | 0.7734 +/- 0.0009 | 0.6600 +/- 0.0024 | 0.4504 +/- 0.0015 | +0.0395 [+0.0383, +0.0412] | +21.7 |
### Pre-registered thresholds

- tt_learned_state vs tt_learned_free (class-balanced accuracy): effect +0.0112 = +6.2 seed-SD (n=5), CI excludes 0 -> beyond threshold.
- tt_learned_state vs single_free_wide (class-balanced accuracy): effect +0.0058 = +3.2 seed-SD (n=5), CI excludes 0 -> beyond threshold.
- tt_learned_state vs the fixed two_trace (E2 reference): effect +0.0395 = +21.7 seed-SD (n=5), CI excludes 0 -> beyond threshold
- SUBSTRATE: tt_learned_free vs the state-matched single store: effect -0.0054 = -4.2 seed-SD (n=5), CI excludes 0 -> not beyond threshold (reported, not part of the K0-TT verdict)
- tt_learned_free vs the fixed two_trace: effect +0.0282 = +49.2 seed-SD (n=5), CI excludes 0 -> the substrate with a learned readout wins (reported, not part of the K0-TT verdict)
- post-hoc oracle ceiling (exact Bayesian filtering with the generator's rates, `scripts/e2_oracle_ceiling.py`): class-balanced 0.7810, natural mix 0.9292; the learned arms sit about 9 and 3 points below it, so the stream has headroom and the co-location of every learned mechanism is not an information ceiling
- **Verdict:** SUPPORTED at the discovery seeds (n = 5), PENDING the pre-registered six-fresh-seed replication before it may be reported as supported (needs every listed control beaten beyond threshold and every clause above true).

### Replication grid (`runs_replication/`, fresh seeds; reported side by side, never merged)

## P3-E4: E3's nested test at 800 training steps, fresh seeds (class-balanced accuracy, mean over 4 regimes), module level

| condition | best LR | seeds | acc_classes mean +/- SD | acc_mean | acc_s0.3_o0 | acc_s1_o0 | acc_s0.3_o0.1 | acc_s1_o0.1 | acc_tag2_s0.3_o0.1 | acc_tag1_s0.3_o0.1 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_free_wide | 0.003 | 6 | 0.6920 +/- 0.0008 | 0.8985 +/- 0.0002 | 0.9188 +/- 0.0004 | 0.9181 +/- 0.0004 | 0.8953 +/- 0.0011 | 0.8618 +/- 0.0003 | 0.5806 +/- 0.0155 | 0.5174 +/- 0.0044 | +0.0067 [+0.0059, +0.0078] | +8.8 |
| tt_learned_free | 0.003 | 6 | 0.6867 +/- 0.0009 | 0.8962 +/- 0.0003 | 0.9231 +/- 0.0011 | 0.9113 +/- 0.0019 | 0.8895 +/- 0.0014 | 0.8607 +/- 0.0012 | 0.4992 +/- 0.0224 | 0.5391 +/- 0.0095 | +0.0120 [+0.0111, +0.0132] | +13.3 |
| tt_learned_state | 0.003 | 6 | 0.6987 +/- 0.0007 | 0.8984 +/- 0.0003 | 0.9275 +/- 0.0023 | 0.9069 +/- 0.0022 | 0.8940 +/- 0.0016 | 0.8651 +/- 0.0018 | 0.6092 +/- 0.0233 | 0.4909 +/- 0.0065 | (hypothesis) |  |
| two_trace | 0 | 6 | 0.6593 +/- 0.0008 | 0.8502 +/- 0.0004 | 0.9158 +/- 0.0003 | 0.8188 +/- 0.0005 | 0.8937 +/- 0.0006 | 0.7725 +/- 0.0007 | 0.6609 +/- 0.0031 | 0.4530 +/- 0.0023 | +0.0394 [+0.0387, +0.0401] | +48.2 |
- replication: tt_learned_state vs tt_learned_free: effect +0.0120 = +13.3 seed-SD (n=6), CI excludes 0 -> beyond threshold.
- replication: tt_learned_state vs single_free_wide: effect +0.0067 = +8.8 seed-SD (n=6), CI excludes 0 -> beyond threshold.
- **Replication verdict:** REPLICATED (discovery seeds [5, 6, 7, 8, 9], replication seeds [10, 11, 12, 13, 14, 15]).


# Phase 3 — system level (GPU)

## P3-S1 (v3) — the two-store memory layer with vs without per-address states in trained memory-only sequence models (rung 1)

PREREG: `experiments/p3_s1_hybrid/PREREG.md`; runbook: docs/phase3/GPU_RUNBOOK.md. Primary metric: held-out accuracy on the decisive strata (re-bound, after-outlier; equal-weighted). The verdict below is computed by `run.py` from the pre-registered conjuncts over every outlier rate.

## P3-S1 (v3): two-store memory with vs without the per-address states, memory-only models on the dense rung-1 task (primary: accuracy on the decisive strata)

### outlier rate 0.15

| condition | best LR | seeds | acc_decisive mean +/- SD | acc_inert | acc_rebound | acc_after_outlier | acc_cf_all | train_acc_last | steps_to_0.9 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attn_only\|r0.15 | 0.001 | 5 | 0.0430 +/- 0.0070 | 0.0693 +/- 0.0033 | 0.0646 +/- 0.0129 | 0.0214 +/- 0.0014 | 0.0618 +/- 0.0018 | 0.0578 +/- 0.0016 | nan +/- nan | +0.7498 [+0.7106, +0.7885] | +15.3 |
| best_fixed_0.5\|r0.15 | 0.001 | 5 | 0.7031 +/- 0.0355 | 0.9695 +/- 0.0146 | 0.9884 +/- 0.0053 | 0.4178 +/- 0.0731 | 0.9438 +/- 0.0233 | 0.8878 +/- 0.0258 | nan +/- nan | +0.0898 [+0.0500, +0.1329] | +1.8 |
| best_fixed_1.0\|r0.15 | 0.001 | 5 | 0.6340 +/- 0.0527 | 0.9133 +/- 0.0667 | 0.9780 +/- 0.0227 | 0.2901 +/- 0.0886 | 0.8707 +/- 0.0844 | 0.8069 +/- 0.0870 | nan +/- nan | +0.1588 [+0.1109, +0.2255] | +3.0 |
| free\|r0.15 | 0.001 | 5 | 0.7767 +/- 0.0474 | 0.9928 +/- 0.0021 | 0.7229 +/- 0.0853 | 0.8304 +/- 0.0401 | 0.9924 +/- 0.0022 | 0.9665 +/- 0.0068 | 2160.0000 +/- 328.6335 | +0.0162 [-0.0149, +0.0487] | +0.3 |
| gdn_x_gate\|r0.15 | 0.001 | 5 | 0.7099 +/- 0.0425 | 0.9639 +/- 0.0139 | 0.9562 +/- 0.0475 | 0.4636 +/- 0.0713 | 0.9364 +/- 0.0182 | 0.8869 +/- 0.0234 | nan +/- nan | +0.0830 [+0.0506, +0.1392] | +1.7 |
| hebb_column\|r0.15 | 0.001 | 5 | 0.7857 +/- 0.0577 | 0.9941 +/- 0.0030 | 0.7197 +/- 0.0905 | 0.8517 +/- 0.0343 | 0.9939 +/- 0.0035 | 0.9707 +/- 0.0076 | 2080.0000 +/- 286.3564 | +0.0072 [-0.0469, +0.0630] | +0.1 |
| single_state\|r0.15 | 0.001 | 5 | 0.7633 +/- 0.0449 | 0.9925 +/- 0.0043 | 0.7226 +/- 0.0732 | 0.8041 +/- 0.0378 | 0.9928 +/- 0.0039 | 0.9631 +/- 0.0103 | 2460.0000 +/- 219.0890 | +0.0295 [-0.0178, +0.0692] | +0.6 |
| single_wide\|r0.15 | 0.001 | 5 | 0.8328 +/- 0.0263 | 0.9987 +/- 0.0011 | 0.7632 +/- 0.0402 | 0.9024 +/- 0.0335 | 0.9984 +/- 0.0013 | 0.9830 +/- 0.0053 | 1840.0000 +/- 320.9361 | -0.0399 [-0.0820, +0.0021] | -0.8 |
| single\|r0.15 | 0.001 | 5 | 0.7968 +/- 0.0244 | 0.9947 +/- 0.0023 | 0.7832 +/- 0.0743 | 0.8104 +/- 0.0510 | 0.9934 +/- 0.0033 | 0.9669 +/- 0.0090 | 2400.0000 +/- 212.1320 | -0.0039 [-0.0446, +0.0293] | -0.1 |
| state_feats_only\|r0.15 | 0.001 | 5 | 0.7596 +/- 0.0332 | 0.9928 +/- 0.0032 | 0.7052 +/- 0.0999 | 0.8139 +/- 0.0514 | 0.9929 +/- 0.0032 | 0.9647 +/- 0.0096 | 2420.0000 +/- 408.6563 | +0.0333 [-0.0106, +0.0704] | +0.7 |
| state_slots_only\|r0.15 | 0.001 | 5 | 0.7780 +/- 0.0570 | 0.9940 +/- 0.0028 | 0.6991 +/- 0.0787 | 0.8569 +/- 0.0378 | 0.9938 +/- 0.0034 | 0.9711 +/- 0.0084 | 2060.0000 +/- 250.9980 | +0.0148 [-0.0256, +0.0621] | +0.3 |
| state\|r0.15 | 0.001 | 5 | 0.7929 +/- 0.0492 | 0.9934 +/- 0.0037 | 0.7403 +/- 0.0535 | 0.8454 +/- 0.0557 | 0.9944 +/- 0.0033 | 0.9700 +/- 0.0112 | 2400.0000 +/- 234.5208 | (hypothesis) |  |

### outlier rate 0.3

| condition | best LR | seeds | acc_decisive mean +/- SD | acc_inert | acc_rebound | acc_after_outlier | acc_cf_all | train_acc_last | steps_to_0.9 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| best_fixed_0.5\|r0.3 | 0.001 | 5 | 0.7365 +/- 0.0588 | 0.9685 +/- 0.0238 | 0.9732 +/- 0.0145 | 0.4998 +/- 0.1280 | 0.9342 +/- 0.0379 | 0.8699 +/- 0.0527 | nan +/- nan | +0.0211 [-0.0288, +0.0746] | +0.4 |
| best_fixed_1.0\|r0.3 | 0.001 | 5 | 0.6479 +/- 0.0577 | 0.9146 +/- 0.0764 | 0.9793 +/- 0.0163 | 0.3164 +/- 0.1024 | 0.8551 +/- 0.0973 | 0.7741 +/- 0.1030 | nan +/- nan | +0.1097 [+0.0676, +0.1576] | +1.9 |
| free\|r0.3 | 0.001 | 5 | 0.7517 +/- 0.0225 | 0.9926 +/- 0.0045 | 0.6441 +/- 0.0751 | 0.8593 +/- 0.0711 | 0.9927 +/- 0.0045 | 0.9622 +/- 0.0169 | 2420.0000 +/- 531.0367 | +0.0059 [-0.0181, +0.0299] | +0.3 |
| hebb_column\|r0.3 | 0.001 | 5 | 0.7560 +/- 0.0157 | 0.9940 +/- 0.0037 | 0.6381 +/- 0.0440 | 0.8739 +/- 0.0590 | 0.9933 +/- 0.0053 | 0.9658 +/- 0.0156 | 2340.0000 +/- 568.3309 | +0.0016 [-0.0194, +0.0227] | +0.1 |
| single_wide\|r0.3 | 0.001 | 5 | 0.8511 +/- 0.0198 | 0.9976 +/- 0.0015 | 0.8158 +/- 0.0695 | 0.8863 +/- 0.0373 | 0.9950 +/- 0.0034 | 0.9748 +/- 0.0073 | 1980.0000 +/- 238.7467 | -0.0935 [-0.1145, -0.0726] | -4.2 |
| single\|r0.3 | 0.001 | 5 | 0.7808 +/- 0.0611 | 0.9908 +/- 0.0076 | 0.7431 +/- 0.0894 | 0.8186 +/- 0.0537 | 0.9902 +/- 0.0063 | 0.9537 +/- 0.0175 | 2780.0000 +/- 414.7288 | -0.0233 [-0.0531, +0.0182] | -0.4 |
| state\|r0.3 | 0.001 | 5 | 0.7576 +/- 0.0224 | 0.9883 +/- 0.0047 | 0.7132 +/- 0.0344 | 0.8019 +/- 0.0377 | 0.9863 +/- 0.0054 | 0.9467 +/- 0.0114 | 2920.0000 +/- 311.4482 | (hypothesis) |  |

### Pre-registered verdict (computed)

- A_state_beats_free_gt2sd_ci_at_every_rate: False
- A_passing: []
- B_state_beats_single_wide_ci_at_every_rate: False
- B_passing: []
- C_no_regression_gt1sd_on_inert_or_cf: False
- C_regressions: [('r0.3', 'acc_cf_all')]
- settings_analysed: 2
- arms_missing_seeds: []
- voided_runs: []
- grid_complete: True
- verdict: FALSIFIED


## P3-S1b (v2) — the Kenyon-cell sparse address against Gated DeltaNet in trained memory-only models (rung 1b)

PREREG: `experiments/p3_s1b_sparse/PREREG.md`. Primary metric: held-out recall accuracy on the capacity form of the dense recall task (P = 32 / 64 / 128 keys, key width 16, one wiring per seed and layer). A is the verdict (sparse against Gated DeltaNet at equal active rows and parameters); B (equal compute), C (equal memory), D (product keys) and E (this repository's full layer) are classified by fixed rules (BEATS / LOSES when the paired-bootstrap CI excludes 0, MATCHES when it lies within +/- 0.02). Per-run values: `runs_table.csv`.

## P3-S1b (v2): the Kenyon-cell sparse address in trained memory-only models, capacity form of the dense recall task (primary: held-out recall accuracy)

### P32

| condition | best LR | seeds | acc mean +/- SD | acc_single | acc_confirmed | train_acc_last | steps_to_0.9 | step_s_median | gpu_peak_reserved_gb | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_c\|P32 | 0.001 | 5 | 0.9880 +/- 0.0087 | 0.9817 +/- 0.0132 | 0.9995 +/- 0.0005 | 0.9882 +/- 0.0086 | 1720.0000 +/- 192.3538 | 0.0242 +/- 0.0001 | 1.7988 +/- 0.0000 | +0.0116 [+0.0059, +0.0196] | +1.3 |
| dense_wide\|P32 | 0.001 | 5 | 1.0000 +/- 0.0001 | 0.9999 +/- 0.0001 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1200.0000 +/- 122.4745 | 0.0598 +/- 0.0000 | 2.2754 +/- 0.0028 | -0.0004 [-0.0008, -0.0001] | -1.1 |
| dense\|P32 | 0.001 | 5 | 0.7862 +/- 0.3268 | 0.7450 +/- 0.3414 | 0.8620 +/- 0.3009 | 0.7857 +/- 0.3261 | nan +/- nan | 0.0266 +/- 0.0030 | 0.9902 +/- 0.0014 | +0.2133 [+0.0522, +0.5077] | +0.7 |
| product\|P32 | 0.001 | 5 | 0.9999 +/- 0.0002 | 0.9998 +/- 0.0003 | 1.0000 +/- 0.0000 | 0.9999 +/- 0.0000 | 1200.0000 +/- 70.7107 | 0.0580 +/- 0.0000 | 2.2773 +/- 0.0000 | -0.0004 [-0.0006, -0.0001] | -0.9 |
| sparse\|P32 | 0.001 | 5 | 0.9995 +/- 0.0004 | 0.9993 +/- 0.0006 | 1.0000 +/- 0.0000 | 0.9997 +/- 0.0004 | 1300.0000 +/- 70.7107 | 0.0579 +/- 0.0002 | 1.7988 +/- 0.0000 | (hypothesis) |  |

### P64

| condition | best LR | seeds | acc mean +/- SD | acc_single | acc_confirmed | train_acc_last | steps_to_0.9 | step_s_median | gpu_peak_reserved_gb | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attn_only\|P64 | 0.001 | 5 | 0.0397 +/- 0.0027 | 0.0304 +/- 0.0008 | 0.0499 +/- 0.0051 | 0.0403 +/- 0.0020 | nan +/- nan | 0.0393 +/- 0.0081 | 5.3730 +/- 0.0000 | +0.9596 [+0.9571, +0.9618] | +349.3 |
| dense_c\|P64 | 0.001 | 5 | 0.9533 +/- 0.0404 | 0.9158 +/- 0.0701 | 0.9949 +/- 0.0075 | 0.9531 +/- 0.0412 | nan +/- nan | 0.0508 +/- 0.0002 | 4.0371 +/- 0.0000 | +0.0459 [+0.0212, +0.0809] | +1.1 |
| dense_wide\|P64 | 0.001 | 5 | 0.9999 +/- 0.0001 | 0.9999 +/- 0.0001 | 1.0000 +/- 0.0001 | 0.9999 +/- 0.0001 | 1160.0000 +/- 134.1641 | 0.1533 +/- 0.0000 | 5.1082 +/- 0.0043 | -0.0007 [-0.0011, -0.0003] | -1.4 |
| dense\|P64 | 0.001 | 5 | 0.8091 +/- 0.0976 | 0.6873 +/- 0.1262 | 0.9439 +/- 0.0667 | 0.8083 +/- 0.0985 | nan +/- nan | 0.0422 +/- 0.0021 | 2.5945 +/- 0.0011 | +0.1902 [+0.1322, +0.2780] | +1.9 |
| expand_dense\|P64 | 0.001 | 5 | 0.8103 +/- 0.0595 | 0.6870 +/- 0.0802 | 0.9469 +/- 0.0368 | 0.8127 +/- 0.0592 | nan +/- nan | 0.1341 +/- 0.0002 | 5.1191 +/- 0.0014 | +0.1889 [+0.1469, +0.2406] | +3.2 |
| full_dense\|P64 | 0.001 | 5 | 0.9394 +/- 0.0516 | 0.9034 +/- 0.0829 | 0.9793 +/- 0.0230 | 0.9398 +/- 0.0516 | nan +/- nan | 0.1610 +/- 0.0023 | 5.3730 +/- 0.0000 | +0.0598 [+0.0231, +0.1029] | +1.2 |
| full_sparse\|P64 | 0.001 | 5 | 0.9999 +/- 0.0002 | 0.9999 +/- 0.0002 | 1.0000 +/- 0.0001 | 0.9999 +/- 0.0001 | 1220.0000 +/- 216.7948 | 0.5447 +/- 0.1183 | 5.3730 +/- 0.0000 | -0.0007 [-0.0010, -0.0003] | -1.4 |
| product\|P64 | 0.001 | 5 | 0.9998 +/- 0.0003 | 0.9997 +/- 0.0005 | 1.0000 +/- 0.0000 | 0.9998 +/- 0.0002 | 1160.0000 +/- 89.4427 | 0.1492 +/- 0.0001 | 5.1152 +/- 0.0000 | -0.0006 [-0.0009, -0.0003] | -1.2 |
| sparse_learned\|P64 | 0.001 | 5 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0001 | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 1120.0000 +/- 44.7214 | 0.1483 +/- 0.0000 | 5.1160 +/- 0.0011 | -0.0007 [-0.0011, -0.0004] | -1.5 |
| sparse\|P64 | 0.001 | 5 | 0.9993 +/- 0.0005 | 0.9986 +/- 0.0009 | 1.0000 +/- 0.0000 | 0.9993 +/- 0.0005 | 1240.0000 +/- 54.7723 | 0.1490 +/- 0.0002 | 4.0371 +/- 0.0000 | (hypothesis) |  |

### P128

| condition | best LR | seeds | acc mean +/- SD | acc_single | acc_confirmed | train_acc_last | steps_to_0.9 | step_s_median | gpu_peak_reserved_gb | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_c\|P128 | 0.001 | 5 | 0.9422 +/- 0.0258 | 0.8634 +/- 0.0554 | 0.9864 +/- 0.0095 | 0.9424 +/- 0.0258 | 2600.0000 +/- 689.2024 | 0.1778 +/- 0.0845 | 6.8750 +/- 0.0000 | +0.0550 [+0.0353, +0.0746] | +2.1 |
| dense_wide\|P128 | 0.001 | 5 | 1.0000 +/- 0.0000 | 0.9999 +/- 0.0001 | 1.0000 +/- 0.0000 | 0.9999 +/- 0.0000 | 1180.0000 +/- 130.3840 | 0.4469 +/- 0.0003 | 8.2609 +/- 0.0017 | -0.0027 [-0.0044, -0.0011] | -1.3 |
| dense\|P128 | 0.001 | 5 | 0.7891 +/- 0.0339 | 0.6212 +/- 0.0410 | 0.8833 +/- 0.0303 | 0.7888 +/- 0.0353 | nan +/- nan | 0.1923 +/- 0.0026 | 6.0020 +/- 0.0000 | +0.2081 [+0.1786, +0.2312] | +6.1 |
| product\|P128 | 0.001 | 5 | 0.9990 +/- 0.0020 | 0.9975 +/- 0.0052 | 0.9999 +/- 0.0003 | 0.9990 +/- 0.0021 | 1240.0000 +/- 207.3644 | 0.4411 +/- 0.0054 | 8.2617 +/- 0.0000 | -0.0018 [-0.0036, -0.0004] | -0.8 |
| sparse\|P128 | 0.001 | 5 | 0.9972 +/- 0.0022 | 0.9933 +/- 0.0049 | 0.9994 +/- 0.0006 | 0.9973 +/- 0.0020 | 1260.0000 +/- 89.4427 | 1.0256 +/- 0.0043 | 6.8746 +/- 0.0009 | (hypothesis) |  |

### Pre-registered verdict (computed)

- A_sparse_beats_dense_gt2sd_ci_at: ['P128']
- A_needed: 2
- B_sparse_vs_dense_c_equal_compute: {'P32': 'BEATS', 'P64': 'BEATS', 'P128': 'BEATS'}
- C_sparse_vs_dense_wide_equal_memory: {'P32': 'LOSES', 'P64': 'LOSES', 'P128': 'LOSES'}
- D_sparse_vs_product_keys: {'P32': 'LOSES', 'P64': 'LOSES', 'P128': 'LOSES'}
- E_full_layer_sparse_vs_dense: {'P64': 'BEATS'}
- arms_missing_seeds: []
- voided_runs: []
- grid_complete: True
- verdict: FALSIFIED
