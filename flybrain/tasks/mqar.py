"""Multi-Query Associative Recall (Arora et al. 2023, "Zoology").

Sequence: n_pairs (key, value) pairs, then a stream of queries (keys seen earlier) mixed
with filler tokens up to seq_len. The model must output the value of a queried key *at the
query position* (it never sees the value again). Loss only on query positions.
Keys, values and filler come from disjoint vocabularies.
"""
from __future__ import annotations

import numpy as np
import torch


def mqar_batch(batch: int, seq_len: int, n_pairs: int, vocab: int, rng: np.random.Generator,
               n_queries: int | None = None):
    assert vocab >= 3 and 2 * n_pairs < seq_len
    nk = vocab // 3
    keys_v = np.arange(0, nk); vals_v = np.arange(nk, 2 * nk); fill_v = np.arange(2 * nk, vocab)
    x = np.zeros((batch, seq_len), dtype=np.int64); y = np.full((batch, seq_len), -100, dtype=np.int64)
    nq = n_queries or n_pairs
    for b in range(batch):
        ks = rng.choice(keys_v, size=n_pairs, replace=False); vs = rng.choice(vals_v, size=n_pairs, replace=True)
        x[b, 0:2 * n_pairs:2] = ks; x[b, 1:2 * n_pairs:2] = vs
        rest = seq_len - 2 * n_pairs
        x[b, 2 * n_pairs:] = rng.choice(fill_v, size=rest)
        qpos = rng.choice(np.arange(2 * n_pairs, seq_len), size=min(nq, rest), replace=False)
        qi = rng.integers(0, n_pairs, size=len(qpos))
        x[b, qpos] = ks[qi]; y[b, qpos] = vs[qi]
    return torch.from_numpy(x), torch.from_numpy(y)


def mqar_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    m = y != -100
    return float((logits.argmax(-1)[m] == y[m]).float().mean())
