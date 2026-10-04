"""An exact chunked step for k-sparse address codes (block B of the 12-hour plan, 3 October 2026; nothing frozen is touched).

The memory layer (flyprinc.hybrid.StateMemoryLayer) stores its state as rows x d_v per head and, in its chunked form, touches
every row for every chunk: K K^T and Q K^T over all rows, the reads K S and Q S, and the rank-n update S += K^T U. With a
k-sparse code (k active cells of `rows`, as the Kenyon-cell address produces) the same arithmetic can be done over the active
rows only: gathers of S at the active cells for the reads, a scatter-add into them for the update, and the two n x n products
from the active cells' values. This module does exactly that for the Gated DeltaNet-gated single-store layer (one store,
gates from the input only, erase tied to write, no history states): `sparse_chunk` is one chunk, `sparse_forward` runs a
layer end to end with it, and tests/test_sparse_kernel.py checks both against the layer's own output to 1e-4.

Cost per chunk and head (n tokens, k active of `rows` cells, values d_v): O(n^2 k + n k d_v + rows d_v) here against
O(n^2 rows + n rows d_v) in the dense form; the per-chunk decay of the whole state (rows x d_v) remains."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyprinc.hybrid import StateMemoryLayer  # noqa: E402


def compact(code: torch.Tensor, k: int):
    """A k-sparse code (B,H,n,rows) as (indices (B,H,n,k), values (B,H,n,k)): the k largest entries (every other entry is 0)."""
    val, idx = code.topk(k, dim=-1)
    return idx, val


def sparse_chunk(idx_q, val_q, idx_k, val_k, vc, S, beta, lam, tau, rows: int):
    """One chunk. idx_*/val_* (B,H,n,k): the active cells and their values for queries and keys; vc (B,H,n,dv); S (B,H,rows,dv);
    beta, lam (B,H,n): write strength (erase = write) and forget drive; tau (H,). Returns the reads y (B,H,n,dv) and the new S."""
    B, H, n, k = idx_k.shape; dv = vc.shape[-1]; dev = vc.device; dt = vc.dtype
    kd = torch.zeros(B, H, n, rows, device=dev, dtype=dt).scatter(-1, idx_k, val_k)       # the keys, dense over cells (n x rows: small)
    # K K^T and Q K^T from the active cells: KK[t,s] = sum_j val_k[t,j] kd[s, idx_k[t,j]]  -> n^2 k multiply-adds
    kd_s = kd[:, :, None].expand(B, H, n, n, rows)
    KK = (kd_s.gather(-1, idx_k[:, :, :, None, :].expand(B, H, n, n, k)) * val_k[:, :, :, None, :]).sum(-1)
    QK = (kd_s.gather(-1, idx_q[:, :, :, None, :].expand(B, H, n, n, k)) * val_q[:, :, :, None, :]).sum(-1)
    # reads of the chunk-start state at the active cells: n k dv each
    S_t = S[:, :, None].expand(B, H, n, rows, dv)
    KS0 = (S_t.gather(3, idx_k[..., None].expand(B, H, n, k, dv)) * val_k[..., None]).sum(3)
    QS0 = (S_t.gather(3, idx_q[..., None].expand(B, H, n, k, dv)) * val_q[..., None]).sum(3)
    # the intra-chunk solve: independent of the number of rows (as in the dense form)
    lg = torch.cumsum(-lam / tau[None, :, None], dim=-1); G = torch.exp(lg)
    D = torch.exp(torch.tril(lg[..., :, None] - lg[..., None, :], -1)); L = torch.tril(D * KK, -1)
    A = torch.eye(n, device=dev, dtype=dt) + beta[..., None] * L
    u = torch.linalg.solve_triangular(A, beta[..., None] * vc - beta[..., None] * G[..., None] * KS0, upper=False, unitriangular=True)
    Dq = torch.exp(torch.tril(lg[..., :, None] - lg[..., None, :], 0))
    y = G[..., None] * QS0 + torch.tril(Dq * QK, 0) @ u
    # the state update over the active cells: S <- G_n S + sum_t k_t (decay_t u_t)^T, a scatter-add of n k rows
    w = torch.exp(lg[..., -1:] - lg)[..., None] * u                                            # (B,H,n,dv)
    src = (val_k[..., None] * w[:, :, :, None, :]).reshape(B, H, n * k, dv)
    S_new = (G[..., -1:, None] * S).scatter_add(2, idx_k.reshape(B, H, n * k, 1).expand(-1, -1, -1, dv), src)
    return y, S_new


@torch.no_grad()
def layer_gates(layer: StateMemoryLayer, xg, vc):
    """The layer's own gates for the input-only case (they see only xg): beta, gamma, lam as (B,H,n)."""
    B, H, n, _ = vc.shape; c = layer.c; z = vc.new_zeros
    beta, gamma, lam = layer._gates(xg, vc, [vc * 0], [vc * 0], z(B, H, n), z(B, H, n, c.n_state_slots))
    return beta[..., 0], gamma[..., 0], lam[..., 0]


def sparse_forward(layer: StateMemoryLayer, x: torch.Tensor, chunk: int | None = None):
    """The layer's forward computed with sparse_chunk (Gated DeltaNet-gated single store with a k-sparse address)."""
    c = layer.c; assert c.n_traces == 1 and c.x_only_gate and c.tie_gamma and not c.states and c.address in ("sparse", "sparse_learned", "product")
    B, T, _ = x.shape; H = c.n_heads; n = chunk or c.chunk; k = c.k_active; rows = c.rows
    q = layer.code(F.normalize(layer.cq(layer.wq(x)).view(B, T, H, c.d_k).transpose(1, 2), dim=-1))
    kk = layer.code(F.normalize(layer.ck(layer.wk(x)).view(B, T, H, c.d_k).transpose(1, 2), dim=-1))
    v = layer.cv(layer.wv(x)).view(B, T, H, c.d_v).transpose(1, 2); xg = layer.wg(x).view(B, T, H, c.d_g).transpose(1, 2)
    S = x.new_zeros(B, H, rows, c.d_v); ys = []
    for s0 in range(0, T, n):
        sl = slice(s0, min(T, s0 + n)); iq, vq = compact(q[:, :, sl], k); ik, vk = compact(kk[:, :, sl], k)
        beta, _, lam = layer_gates(layer, xg[:, :, sl], v[:, :, sl])
        y, S = sparse_chunk(iq, vq, ik, vk, v[:, :, sl], S, beta, lam, layer.tau, rows); ys.append(y)
    y = torch.cat(ys, 2); y = layer.norm_o(y).transpose(1, 2).reshape(B, T, H * c.d_v)
    return layer.wo(torch.cat([y, y.new_zeros(B, T, 2 * H)], -1)), S
