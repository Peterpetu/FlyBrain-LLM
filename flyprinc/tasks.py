"""Stream tasks for the module-level probe. A stream is a batch of B independent event
sequences sharing one schedule: `kinds` (T,) in {0 = write, 1 = query, 2 = sleep}, `keys` (B, T)
key ids, `vals` (B, T) value ids (the value written, or the correct answer for a query), `tags`
(T,) an integer per event for stratified metrics (meaning per task)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

WRITE, QUERY, SLEEP = 0, 1, 2


@dataclass
class Vocab:
    """Fixed random embeddings: keys Gaussian (d_key), values unit-norm (d_val)."""
    n_keys: int = 2048
    n_vals: int = 64
    d_key: int = 64
    d_val: int = 16
    seed: int = 1234

    def tensors(self):
        g = torch.Generator().manual_seed(self.seed)
        K = torch.randn(self.n_keys, self.d_key, generator=g)
        V = torch.nn.functional.normalize(torch.randn(self.n_vals, self.d_val, generator=g), dim=-1)
        return K, V


def _stream(kinds, keys, vals, tags, meta=None):
    return {"kinds": np.asarray(kinds, np.int64), "keys": np.asarray(keys, np.int64), "vals": np.asarray(vals, np.int64),
            "tags": np.asarray(tags, np.int64), "meta": meta or {}}


def capacity_stream(B: int, P: int, voc: Vocab, rng: np.random.Generator):
    """P distinct keys written once each with random values, then every key queried (shuffled). tags = 0."""
    kinds = [WRITE] * P + [QUERY] * P; keys = np.zeros((B, 2 * P), np.int64); vals = np.zeros((B, 2 * P), np.int64)
    for b in range(B):
        ks = rng.choice(voc.n_keys, size=P, replace=False); vs = rng.integers(0, voc.n_vals, size=P)
        keys[b, :P], vals[b, :P] = ks, vs
        q = rng.permutation(P); keys[b, P:], vals[b, P:] = ks[q], vs[q]
    return _stream(kinds, keys, vals, [0] * (2 * P), {"P": P})


def rebind_stream(B: int, P: int, T: int, voc: Vocab, rng: np.random.Generator, p_change: float = 0.001, p_outlier: float = 0.0,
                  probe_every: int = 50, recent: int = 150):
    """Noisy non-stationary stream (noise is added by the probe): each step presents one random pair
    (a write); with probability p_change * P one key is silently re-bound (a genuine change); with
    probability p_outlier a presentation shows a wrong random value ONCE (an outlier, not a change).
    Every `probe_every` steps all P keys are queried; tags: 0 stable, 1 re-bound within `recent` steps,
    2 the key's last presentation was an outlier."""
    kinds, tags = [], []; keys = [[] for _ in range(B)]; vals = [[] for _ in range(B)]
    ks = [rng.choice(voc.n_keys, size=P, replace=False) for _ in range(B)]
    vs = [rng.integers(0, voc.n_vals, size=P) for _ in range(B)]
    changed = np.full(P, -10 ** 9); last_outlier = np.zeros(P, bool)
    for t in range(T):                       # the schedule is shared across the batch so kinds/tags align
        if rng.random() < p_change * P:
            q = rng.integers(P); changed[q] = t
            for b in range(B): vs[b][q] = (vs[b][q] + rng.integers(1, voc.n_vals)) % voc.n_vals
        p = rng.integers(P); outlier = rng.random() < p_outlier
        kinds.append(WRITE); tags.append(0); last_outlier[p] = outlier
        for b in range(B):
            keys[b].append(ks[b][p]); vals[b].append((vs[b][p] + rng.integers(1, voc.n_vals)) % voc.n_vals if outlier else vs[b][p])
        if (t + 1) % probe_every == 0 and t > recent:
            for q in range(P):
                kinds.append(QUERY); tags.append(2 if last_outlier[q] else (1 if t - changed[q] < recent else 0))
                for b in range(B): keys[b].append(ks[b][q]); vals[b].append(vs[b][q])
    return _stream(kinds, keys, vals, tags, {"P": P, "T": T, "p_change": p_change, "p_outlier": p_outlier})


def load_stream(B: int, voc: Vocab, rng: np.random.Generator, n_long: int = 16, n_short: int = 16, interval_long: int = 256,
                interval_short: int = 8, T: int = 2048, p_distractor: float = 0.5):
    """Items with fixed values recur at a long interval (tag 0) or a short one (tag 1); each recurrence is
    preceded by a query of the item. Meanwhile one-shot distractors (never queried) arrive with probability
    p_distractor per step, so a store without forgetting fills up and a store that forgets fast loses the
    long-interval items: the regime where time scales matter."""
    items = [(0, interval_long, rng.integers(interval_long)) for _ in range(n_long)] + [(1, interval_short, rng.integers(interval_short)) for _ in range(n_short)]
    n = len(items); n_d = int(T * p_distractor) + 8
    ks = [rng.choice(voc.n_keys, size=n + n_d, replace=False) for _ in range(B)]; vs = [rng.integers(0, voc.n_vals, size=n + n_d) for _ in range(B)]
    kinds, tags = [], []; keys = [[] for _ in range(B)]; vals = [[] for _ in range(B)]
    seen = np.zeros(n, bool); d = 0
    for t in range(T):
        if rng.random() < p_distractor and d < n_d:
            kinds.append(WRITE); tags.append(9)
            for b in range(B): keys[b].append(ks[b][n + d]); vals[b].append(vs[b][n + d])
            d += 1
        for i, (cls, iv, ph) in enumerate(items):
            if (t - ph) % iv == 0:
                if seen[i]:
                    kinds.append(QUERY); tags.append(cls)
                    for b in range(B): keys[b].append(ks[b][i]); vals[b].append(vs[b][i])
                kinds.append(WRITE); tags.append(cls); seen[i] = True
                for b in range(B): keys[b].append(ks[b][i]); vals[b].append(vs[b][i])
    return _stream(kinds, keys, vals, tags, {"n_long": n_long, "n_short": n_short, "distractors": d})


def consolidation_stream(B: int, voc: Vocab, rng: np.random.Generator, P_total: int = 600, sleep_every: int = 50, age_edges=(50, 150, 300)):
    """New items arrive in sequence (write once); a sleep event every `sleep_every` items; at the end every
    item is queried with tag = age bucket (0 newest .. len(age_edges) oldest)."""
    kinds, tags = [], []; keys = [[] for _ in range(B)]; vals = [[] for _ in range(B)]
    ks = [rng.choice(voc.n_keys, size=P_total, replace=False) for _ in range(B)]; vs = [rng.integers(0, voc.n_vals, size=P_total) for _ in range(B)]
    for p in range(P_total):
        kinds.append(WRITE); tags.append(0)
        for b in range(B): keys[b].append(ks[b][p]); vals[b].append(vs[b][p])
        if (p + 1) % sleep_every == 0:
            kinds.append(SLEEP); tags.append(0)
            for b in range(B): keys[b].append(0); vals[b].append(0)
    for p in range(P_total):
        age = P_total - p; bucket = int(np.searchsorted(np.asarray(age_edges), age, side="right"))
        kinds.append(QUERY); tags.append(bucket)
        for b in range(B): keys[b].append(ks[b][p]); vals[b].append(vs[b][p])
    return _stream(kinds, keys, vals, tags, {"P_total": P_total, "sleep_every": sleep_every, "age_edges": list(age_edges)})


def reversal_stream(B: int, voc: Vocab, rng: np.random.Generator, P: int = 32, n_rev: int = 8, reps: int = 3):
    """A -> B -> A reversal. Phase A: every pair presented `reps` times (values v1). Phase B: n_rev keys
    re-bound to v2, presented `reps` times each (interleaved with unreversed pairs). Phase C: the original
    v1 returns; after each of the first `reps` presentations of a reversed key it is queried
    (tag = presentation index 1..reps); unreversed keys are queried at the end (tag 9)."""
    kinds, tags = [], []; keys = [[] for _ in range(B)]; vals = [[] for _ in range(B)]
    ks = [rng.choice(voc.n_keys, size=P, replace=False) for _ in range(B)]
    v1 = [rng.integers(0, voc.n_vals, size=P) for _ in range(B)]
    v2 = [(v1[b] + rng.integers(1, voc.n_vals, size=P)) % voc.n_vals for b in range(B)]
    rev = rng.choice(P, size=n_rev, replace=False)
    def present(idx_list, which):
        for i in idx_list:
            kinds.append(WRITE); tags.append(0)
            for b in range(B): keys[b].append(ks[b][i]); vals[b].append((v2 if which == 2 else v1)[b][i])
    for _ in range(reps): present(rng.permutation(P), 1)                    # phase A
    # phase B: reversed keys carry v2, the others v1 (interleaved)
    for _ in range(reps):
        for i in rng.permutation(P):
            kinds.append(WRITE); tags.append(0)
            for b in range(B): keys[b].append(ks[b][i]); vals[b].append(v2[b][i] if i in rev else v1[b][i])
    # phase C: v1 returns; query each reversed key right after each of its first `reps` presentations
    for r in range(1, reps + 1):
        for i in rng.permutation(P):
            kinds.append(WRITE); tags.append(0)
            for b in range(B): keys[b].append(ks[b][i]); vals[b].append(v1[b][i])
            if i in rev:
                kinds.append(QUERY); tags.append(r)
                for b in range(B): keys[b].append(ks[b][i]); vals[b].append(v1[b][i])
    for i in range(P):
        if i not in rev:
            kinds.append(QUERY); tags.append(9)
            for b in range(B): keys[b].append(ks[b][i]); vals[b].append(v1[b][i])
    return _stream(kinds, keys, vals, tags, {"P": P, "n_rev": n_rev, "reps": reps})
