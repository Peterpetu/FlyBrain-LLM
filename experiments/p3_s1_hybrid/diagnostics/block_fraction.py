"""Fraction of key presentations whose previous presentation of the same key lies in the same b-token block
(the gate senses the block-start state, so it does not yet see that earlier presentation's effect)."""
import sys
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from flyprinc.lmtasks import LMTaskConfig, make_batch

cfg = LMTaskConfig(p_rebind=0.2, p_outlier=0.15)
toks, tgt, strat, far = make_batch(cfg, 256, np.random.default_rng(0))
for b in (1, 16, 32, 64):
    same = total = 0
    for s in range(toks.shape[0]):
        last = {}
        for p in range(toks.shape[1] - 1):
            t = int(toks[s, p])
            if t < cfg.n_keys and (p == 0 or toks[s, p - 1] != cfg.Q):          # a presentation (not a query's key)
                if t in last:
                    total += 1; same += (last[t] // b) == (p // b)
                last[t] = p
    print(f"block {b}: {same / total:.3f} of re-presentations fall in the same block as the previous presentation ({same} of {total})")
