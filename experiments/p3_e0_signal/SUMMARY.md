## P3-E0: signal availability of candidate per-address states (no training; 8 seeds, even train / odd test)

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
