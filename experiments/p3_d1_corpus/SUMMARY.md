## P3-D1 (v2): persistent re-binding of key-like tokens in text, against a within-key permutation null

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
