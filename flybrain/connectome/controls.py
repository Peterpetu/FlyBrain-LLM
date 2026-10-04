"""Graph controls required by the protocol.

* `degree_preserving_shuffle`: directed edge swaps that preserve every node's in- and
  out-degree (and the multiset of edge weights); >= 10 |E| swap attempts by default.
* `erdos_renyi`: uniformly random directed graph with the same N and E.
* `fit_dcsbm` / `sample_dcsbm`: degree-corrected stochastic block model on cell-type
  blocks, and a sampler at arbitrary N that preserves block structure, degree
  heterogeneity and per-type sign.

All functions take/return scipy CSR matrices with row = presynaptic source. Signs are a
property of *nodes* (Dale's law) and are handled by the caller, so every control
preserves the sign fraction by construction.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _edges(W: sp.csr_matrix):
    W = W.tocoo()
    return W.row.astype(np.int64), W.col.astype(np.int64), W.data.copy()


def degree_preserving_shuffle(W: sp.csr_matrix, rng: np.random.Generator, attempts_per_edge: int = 10) -> sp.csr_matrix:
    """Directed degree-preserving rewiring (Maslov-Sneppen edge swaps).

    Picks two edges (a->b), (c->d) and rewires to (a->d), (c->b) if neither creates a
    self-loop or duplicate. Weights travel with the *source* endpoint's edge, so the
    out-weight multiset of each node is preserved as well as in/out degree.
    """
    n = W.shape[0]
    src, dst, w = _edges(W)
    E = len(src)
    key = src * n + dst
    present = set(key.tolist())
    attempts = attempts_per_edge * E
    accepted = 0
    # batched proposals for speed; conflicts within a batch are resolved sequentially
    batch = 1 << 16
    done = 0
    while done < attempts:
        m = min(batch, attempts - done); done += m
        I = rng.integers(0, E, size=m); J = rng.integers(0, E, size=m)
        for i, j in zip(I.tolist(), J.tolist()):
            if i == j:
                continue
            a, b, c_, d = src[i], dst[i], src[j], dst[j]
            if a == d or c_ == b:
                continue
            k1 = a * n + d; k2 = c_ * n + b
            if k1 in present or k2 in present:
                continue
            present.remove(a * n + b); present.remove(c_ * n + d)
            present.add(k1); present.add(k2)
            dst[i], dst[j] = d, b
            accepted += 1
    out = sp.csr_matrix((w, (src, dst)), shape=W.shape)
    out.sum_duplicates()
    assert out.nnz == W.nnz, "shuffle created duplicates"
    out.accepted_swaps = accepted  # type: ignore[attr-defined]
    return out


def erdos_renyi(n: int, n_edges: int, rng: np.random.Generator, weights: np.ndarray | None = None) -> sp.csr_matrix:
    """G(n, E): E distinct directed edges without self-loops, weights permuted from `weights`."""
    chosen = set()
    keys = []
    while len(keys) < n_edges:
        need = n_edges - len(keys)
        s = rng.integers(0, n, size=int(need * 1.2) + 16); d = rng.integers(0, n, size=len(s))
        ok = s != d
        for k in (s[ok] * n + d[ok]).tolist():
            if k not in chosen:
                chosen.add(k); keys.append(k)
                if len(keys) == n_edges:
                    break
    keys = np.array(keys, dtype=np.int64)
    src, dst = keys // n, keys % n
    w = rng.permutation(weights) if weights is not None else np.ones(n_edges, dtype=np.float32)
    return sp.csr_matrix((w, (src, dst)), shape=(n, n))


# ---------------------------------------------------------------------------------------
# Degree-corrected stochastic block model on cell-type blocks
# ---------------------------------------------------------------------------------------
def fit_dcsbm(W: sp.csr_matrix, blocks: np.ndarray, sign: np.ndarray) -> dict:
    """Fit a directed degree-corrected SBM (Karrer & Newman 2011) with blocks = cell types.

    Returns block sizes, block edge-count matrix M[r,s] (number of edges r->s), node degree
    propensities theta_out/theta_in normalised within block, and inhibitory fraction per block.
    Edges are counted, weights are kept as an empirical per-block distribution.
    """
    labels, b = np.unique(blocks, return_inverse=True)
    B = len(labels)
    A = (W > 0).astype(np.float64).tocsr()
    P = sp.csr_matrix((np.ones(len(b)), (b, np.arange(len(b)))), shape=(B, len(b)))
    M = (P @ A @ P.T).toarray()                       # edges from block r to block s
    kout = np.asarray(A.sum(1)).ravel(); kin = np.asarray(A.sum(0)).ravel()
    kappa_out = np.bincount(b, weights=kout, minlength=B); kappa_in = np.bincount(b, weights=kin, minlength=B)
    theta_out = kout / np.maximum(kappa_out[b], 1); theta_in = kin / np.maximum(kappa_in[b], 1)
    inh = np.bincount(b, weights=sign < 0, minlength=B) / np.maximum(np.bincount(b, minlength=B), 1)
    return {"labels": labels, "block_of": b, "sizes": np.bincount(b, minlength=B), "M": M,
            "theta_out": theta_out, "theta_in": theta_in, "inh_frac": inh,
            "weights": W.data.astype(np.float32), "n": W.shape[0], "E": int(A.nnz)}


def sample_dcsbm(fit: dict, n_target: int, rng: np.random.Generator, mean_degree: float | None = None) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Sample a graph with `n_target` nodes preserving block proportions, block-pair edge
    propensities, within-block degree heterogeneity (theta resampled from the empirical
    distribution) and per-block sign fraction.

    Edges are Bernoulli(min(1, lambda_ij)) with lambda_ij = M_rs * theta_i * theta_j
    (the standard dcSBM likelihood; Karrer & Newman 2011), rescaled once so the expected
    edge count hits the target. Mean degree defaults to the fitted graph's, so density
    falls as 1/N like real neural graphs, rather than staying constant."""
    B = len(fit["sizes"]); n0 = fit["n"]
    sizes = np.maximum(1, np.round(fit["sizes"] / n0 * n_target)).astype(int)
    block_of = np.repeat(np.arange(B), sizes); n = len(block_of)
    th_out = np.empty(n); th_in = np.empty(n)
    starts = np.r_[0, np.cumsum(sizes)]
    for r in range(B):
        idx0 = np.flatnonzero(fit["block_of"] == r); idx = np.arange(starts[r], starts[r + 1])
        th_out[idx] = rng.choice(fit["theta_out"][idx0], size=len(idx)); th_in[idx] = rng.choice(fit["theta_in"][idx0], size=len(idx))
        th_out[idx] /= max(th_out[idx].sum(), 1e-12); th_in[idx] /= max(th_in[idx].sum(), 1e-12)
    md = mean_degree if mean_degree is not None else fit["E"] / n0
    target_E = md * n
    Mscaled = fit["M"] * (target_E / fit["E"])
    # expected edges under clipping, one correction pass
    def expected(scale):
        tot = 0.0
        for r in range(B):
            for s_ in range(B):
                if Mscaled[r, s_] == 0: continue
                lam = scale * Mscaled[r, s_] * np.outer(th_out[starts[r]:starts[r + 1]], th_in[starts[s_]:starts[s_ + 1]])
                tot += np.minimum(lam, 1.0).sum()
        return tot
    scale = 1.0
    for _ in range(3):
        e = expected(scale)
        if e > 0: scale *= target_E / e
    src, dst = [], []
    for r in range(B):
        for s_ in range(B):
            if Mscaled[r, s_] == 0: continue
            lam = np.minimum(scale * Mscaled[r, s_] * np.outer(th_out[starts[r]:starts[r + 1]], th_in[starts[s_]:starts[s_ + 1]]), 1.0)
            hit = rng.random(lam.shape) < lam
            i, j = np.nonzero(hit)
            src.append(i + starts[r]); dst.append(j + starts[s_])
    src = np.concatenate(src); dst = np.concatenate(dst)
    ok = src != dst; src, dst = src[ok], dst[ok]
    w = rng.choice(fit["weights"], size=len(src))
    W = sp.csr_matrix((w, (src, dst)), shape=(n, n))
    sign = np.ones(n, dtype=np.int8)
    for r in range(B):
        idx = np.arange(starts[r], starts[r + 1]); k = int(round(fit["inh_frac"][r] * len(idx)))
        if k: sign[rng.choice(idx, size=k, replace=False)] = -1
    return W, block_of, sign


def graph_stats(W: sp.csr_matrix) -> dict:
    A = (W > 0).astype(np.int64)
    ko = np.asarray(A.sum(1)).ravel(); ki = np.asarray(A.sum(0)).ravel()
    n = W.shape[0]
    recip = int(A.multiply(A.T).nnz)
    return {"n": n, "E": int(A.nnz), "density": float(A.nnz / (n * (n - 1))), "outdeg_mean": float(ko.mean()),
            "outdeg_max": int(ko.max()), "indeg_max": int(ki.max()), "outdeg_cv": float(ko.std() / max(ko.mean(), 1e-9)),
            "reciprocity": float(recip / max(A.nnz, 1))}
