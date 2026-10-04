# Consolidation toy: a negative result to keep

*Pure-numpy study (`results/consolidation_numpy.json`), run before the phase-3 design.*

A fast sparse delta-rule store (64 of 1,024 rows, multiplicative decay 0.995 per write, so its
horizon is about 200 items) received 600 associations in sequence. A slow dense store (rank 512,
random fixed key projection, normalised delta rule, rate 0.3) learned only during "sleep" every
50 items, by replaying 64 random addresses through the fast store and consolidating whatever was
still readable. At test, an item is read from the fast store if readable, else from the slow one.

| items by age at test | fast store only | fast + naive consolidation |
|---|---|---|
| newest 50 | 0.76 | 0.80 |
| 51-150 | 0.64 | 0.62 |
| 151-300 | 0.26 | 0.26 |
| older than 300 | 0.02 | 0.06 |

**Reading.** A naive replay scheme (one replay per item, random selection, single-pass delta rule
into a rank-512 store loaded with up to 600 items) barely retains anything beyond the fast
horizon. This is the known hard part of complementary-learning-systems theory: consolidation
needs (i) selection of what to replay (novelty / value, which is what the fly's dopamine and
MBON valence signals provide), (ii) enough replay passes for the slow learner to converge, and
(iii) a slow store whose capacity exceeds the stream. The toy establishes that "add sleep" is
not a free ingredient; the phase-3 design must specify the selection rule and the replay budget
and measure retention against a store with the same total capacity and no consolidation.
No further tuning of this toy was attempted, because a toy tuned until it works demonstrates
nothing.
