"""Connectome-masked recurrent layer with Dale's law (H5 calibration experiment).

h_t = tanh( x_t W_in + h_{t-1} (mask * |W| * sign[:, None]) + b )

* mask : fixed adjacency (row = presynaptic), from the real core graph or a control graph
* sign : fixed per-source-neuron sign from the neurotransmitter prediction
* |W|  : learned non-negative magnitudes, initialised from log contact counts

Dense baselines use the same class with a full mask and free signs.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def select_core(c, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Top-n neurons by total contact-weighted degree inside the largest strongly connected
    component of the whole-CNS graph (deterministic)."""
    from scipy.sparse.csgraph import connected_components
    ncomp, lab = connected_components(c.W, directed=True, connection="strong")
    big = np.argmax(np.bincount(lab)); scc = np.flatnonzero(lab == big)
    deg = np.asarray(c.W.sum(1)).ravel() + np.asarray(c.W.sum(0)).ravel()
    order = scc[np.argsort(-deg[scc], kind="stable")]
    return np.sort(order[:n])


class MaskedDaleRNN(nn.Module):
    def __init__(self, d_in: int, W_adj: sp.csr_matrix | None, sign: np.ndarray | None, n_hidden: int,
                 dale: bool = True, init_from_counts: bool = True):
        super().__init__()
        self.n = n_hidden
        if W_adj is None:  # dense, unconstrained
            self.register_buffer("mask", torch.ones(n_hidden, n_hidden))
            self.register_buffer("sign", torch.ones(n_hidden))
            self.dale = False
            self.W = nn.Parameter(torch.randn(n_hidden, n_hidden) / np.sqrt(n_hidden))
        else:
            # structural mask = every stored edge (the mirror keeps 3.2 % of edges with an
            # unrecorded weight of 0; they are present edges, see LIMITATIONS.md), and
            # unrecorded weights are initialised at the 3-contact inclusion threshold.
            W_adj = W_adj.tocsr(); M = W_adj.copy(); M.data = np.ones_like(M.data)
            A = W_adj.toarray().astype(np.float32); A[(M.toarray() > 0) & (A <= 0)] = 3.0
            self.register_buffer("mask", torch.tensor(M.toarray() > 0, dtype=torch.float32))
            self.register_buffer("sign", torch.tensor(sign, dtype=torch.float32) if dale else torch.ones(n_hidden))
            self.dale = dale
            if init_from_counts:
                mag = np.log1p(A); mag = mag / (mag.sum(0, keepdims=True) + 1e-6)   # column-normalised fan-in
                W0 = torch.tensor(mag, dtype=torch.float32)
            else:
                W0 = torch.rand(n_hidden, n_hidden) * self.mask
                W0 = W0 / (W0.sum(0, keepdim=True) + 1e-6)
            self.W = nn.Parameter(W0 if dale else W0 * torch.tensor(sign, dtype=torch.float32)[:, None])
        self.W_in = nn.Linear(d_in, n_hidden)
        self.b = nn.Parameter(torch.zeros(n_hidden))
        self.gain = nn.Parameter(torch.ones(1) * 1.0)

    def effective_W(self):
        if self.dale:
            return self.mask * self.W.abs() * self.sign[:, None] * self.gain
        return self.mask * self.W * self.gain

    def forward(self, x, state=None):
        B, T, _ = x.shape
        W = self.effective_W(); xin = self.W_in(x)
        h = state if state is not None else x.new_zeros(B, self.n)
        outs = []
        for t in range(T):
            h = torch.tanh(xin[:, t] + h @ W + self.b)
            outs.append(h)
        return torch.stack(outs, 1), h

    def n_recurrent_params(self) -> int:
        return int(self.mask.sum().item())
