## P3-D1: re-binding of key-like tokens in text (continuation-change proxy)

| corpus | window | keys / 1k tokens | repeats / 1k | re-binding fraction (keys) | re-binding fraction (freq-matched control) | re-bindings / 1k | re-bindings with gap > 512 / 1k | gap median / p90 |
|---|---|---|---|---|---|---|---|---|
| wikitext2.train.txt | 512 | 356.8 | 131.4 | 0.92 | 0.94 | 121.47 | 0.00 | 70 / 337 |
| wikitext2.train.txt | 2048 | 356.8 | 173.4 | 0.92 | 0.95 | 159.90 | 38.43 | 127 / 1119 |
| wikitext2.train.txt | 8192 | 356.8 | 207.4 | 0.93 | 0.96 | 192.38 | 70.91 | 217 / 3545 |
| ptb.train.txt | 512 | 233.8 | 85.0 | 0.97 | 0.00 | 82.18 | 0.00 | 28 / 209 |
| ptb.train.txt | 2048 | 233.8 | 95.0 | 0.97 | 0.00 | 91.90 | 9.72 | 35 / 550 |
| ptb.train.txt | 8192 | 233.8 | 118.0 | 0.97 | 0.00 | 114.50 | 32.32 | 64 / 4850 |

**Decision rule.** Re-bindings beyond a 512-token gap exceed 1 per 2,048 tokens at W = 8192: True. Key-like re-binding fraction exceeds the frequency-matched control by > 5 points: True. **Language rung in scope under the pre-registered rule: True.**
