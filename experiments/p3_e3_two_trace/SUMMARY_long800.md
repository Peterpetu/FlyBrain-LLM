## P3-E3: learned readout over two traces with vs without the states; state-matched single store (class-balanced accuracy, mean over 4 regimes), module level

| condition | best LR | seeds | acc_classes mean +/- SD | acc_mean | acc_s0.3_o0 | acc_s1_o0 | acc_s0.3_o0.1 | acc_s1_o0.1 | acc_tag2_s0.3_o0.1 | acc_tag1_s0.3_o0.1 | vs hypothesis: diff [95% CI] | effect / seed-SD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_free_wide | 0.003 | 3 | 0.6924 +/- 0.0008 | 0.8986 +/- 0.0002 | 0.9184 +/- 0.0011 | 0.9178 +/- 0.0007 | 0.8958 +/- 0.0015 | 0.8623 +/- 0.0010 | 0.5923 +/- 0.0244 | 0.5125 +/- 0.0077 | +0.0070 [+0.0062, +0.0076] | +8.2 |
| tt_learned_free | 0.003 | 3 | 0.6870 +/- 0.0016 | 0.8962 +/- 0.0001 | 0.9227 +/- 0.0032 | 0.9107 +/- 0.0029 | 0.8898 +/- 0.0037 | 0.8617 +/- 0.0027 | 0.5079 +/- 0.0651 | 0.5339 +/- 0.0220 | +0.0123 [+0.0104, +0.0136] | +7.8 |
| tt_learned_state | 0.003 | 3 | 0.6993 +/- 0.0007 | 0.8983 +/- 0.0003 | 0.9271 +/- 0.0040 | 0.9053 +/- 0.0037 | 0.8945 +/- 0.0025 | 0.8662 +/- 0.0044 | 0.6207 +/- 0.0479 | 0.4868 +/- 0.0163 | (hypothesis) |  |