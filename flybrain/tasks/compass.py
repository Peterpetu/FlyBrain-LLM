"""Heading-integration ("compass") state tracking with length extrapolation.

Tokens: 0 = forward (no turn), 1 = turn left by 1 sector, 2 = turn right by 1,
3 = turn left by 2, 4 = turn right by 2, 5.. = distractors (no effect).
Target at every position = current heading in {0..n_sectors-1} (cyclic group Z_n).
This is the S_n-word-problem-style state tracking that finite-precision transformers and
diagonal linear RNNs cannot do beyond training length (Merrill et al. 2024; Grazzi et al.
2025); a ring attractor integrates it natively. Train at one length, test at 4-16x.
"""
from __future__ import annotations

import numpy as np
import torch

N_ACTION = 5
STEP = np.array([0, 1, -1, 2, -2])


def compass_batch(batch: int, seq_len: int, rng: np.random.Generator, n_sectors: int = 8,
                  n_distractors: int = 4, p_turn: float = 0.5):
    vocab = N_ACTION + n_distractors
    x = np.where(rng.random((batch, seq_len)) < p_turn, rng.integers(1, N_ACTION, size=(batch, seq_len)),
                 rng.integers(N_ACTION, vocab, size=(batch, seq_len)))
    x[rng.random((batch, seq_len)) < 0.1] = 0
    step = np.where(x < N_ACTION, STEP[np.minimum(x, N_ACTION - 1)], 0)
    head = np.cumsum(step, 1) % n_sectors
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(head.astype(np.int64))


def compass_vocab(n_distractors: int = 4) -> int:
    return N_ACTION + n_distractors
