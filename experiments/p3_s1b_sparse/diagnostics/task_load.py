"""Rung-1b task statistics (reported in PREREG.md, Task): for the capacity form of the dense recall generator at
P = 32, 64, 128 (and 256, which the placement cannot fill) with T = 8P, on 64 sequences per P: queries and
presentations per sequence, the memory load at a query (distinct keys already presented), and the distance from
a query to its key's last presentation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from run import task_for, query_load, FULL
from flyprinc.lmtasks import make_batch

for P in (32, 64, 128, 256):
    task = task_for(dict(FULL, keys=P)); toks, tgt, _, _ = make_batch(task, 64, np.random.default_rng(0)); m = tgt >= 0
    load = query_load(toks, tgt, task.n_keys, task.Q)[m]
    pres = np.mean([((toks[b] < task.n_keys) & np.r_[True, toks[b][:-1] != task.Q]).sum() for b in range(64)])
    dist = []
    for b in range(64):
        last = {}
        for p in range(task.T):
            t = toks[b, p]
            if t < task.n_keys:
                if p > 0 and toks[b, p - 1] == task.Q: dist.append(p - last.get(t, p))
                else: last[t] = p
    q = np.percentile(load, [10, 50, 90])
    print(f"P={P:3d} T={task.T:4d}: queries/seq {m.sum(1).mean():5.1f}, presentations/seq {pres:5.1f} ({pres / P:.2f} per key), "
          f"load at query p10/p50/p90 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}, query distance median {np.median(dist):.0f} (p90 {np.percentile(dist, 90):.0f}), "
          f"filler {(toks >= task.n_keys + task.n_vals).mean():.0%}")
