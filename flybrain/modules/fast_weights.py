"""One fast-weight associative-memory layer, many instances.

Following the Miras view (Behrouz et al. 2025) every fast-weight sequence layer is an
online optimiser of an associative loss with a retention term:

    M_t = alpha_t * M_{t-1}  -  beta_t * grad_M || read(M_{t-1}, k_t) - v_t ||^2      (delta)
    M_t = alpha_t * M_{t-1}  +  beta_t * k_t v_t^T                                    (hebb)

Choices that turn this into a named model:

    linear attention : dense keys, hebb write, alpha = 1, beta = 1
    DeltaNet         : dense keys, delta write, alpha = 1, beta learned
    Gated DeltaNet   : dense keys, delta write, alpha learned, beta learned
    Mushroom body    : keys = fixed sparse expansion + k-WTA (PN->KC->APL), delta write
                       (dopamine-gated depression of active KC->MBON synapses), beta =
                       function of the prediction error itself (DAN = MBON feedback minus
                       reinforcement), alpha learned, one head per compartment.

Everything shares one code path so the comparison is implementation-fair. FLOPs per token
are computed analytically by `fast_weight_flops` for compute matching.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FWConfig:
    d_model: int = 64
    n_heads: int = 1                   # MB: compartments
    d_value: int = 32                  # per head
    # key space
    key_mode: str = "dense"            # dense | sparse_random | sparse_connectome | sparse_shuffled | dense_expanded
    d_key: int = 32                    # dense key dim per head
    n_pn: int = 60                     # PN types feeding the expansion
    n_kc: int = 512                    # KCs per head (expansion width)
    fan_in: int = 5                    # PN inputs per KC (connectome: ~5 types)
    kwta_frac: float = 0.05            # APL-enforced active fraction (Lin 2014: ~5%)
    kwta: bool = True
    shared_query: bool = True          # MB: same KC code for write and read
    # write rule
    write: str = "delta"               # delta | hebb
    gate: str = "learned"              # learned | error | const
    beta_const: float = 0.5
    decay: str = "learned"             # learned | none
    connectome_matrix: Optional[np.ndarray] = None   # (n_kc, n_pn) real PN->KC contacts (for sparse_connectome/shuffled)
    seed: int = 0
    out_norm: bool = True
    conv_kernel: int = 0               # depthwise causal conv over time before the projections (GDN/Mamba style); 0 = off


class SparseExpansion(nn.Module):
    """Fixed PN->KC expansion followed by APL-like k-winner-take-all."""

    def __init__(self, cfg: FWConfig):
        super().__init__()
        g = torch.Generator().manual_seed(cfg.seed)
        n_kc, n_pn = cfg.n_kc, cfg.n_pn
        if cfg.key_mode == "sparse_connectome":
            E = torch.tensor(cfg.connectome_matrix, dtype=torch.float32)
            assert E.shape[1] == n_pn, "connectome matrix width must equal n_pn"
            E = _subsample_rows(E, n_kc, g)
        elif cfg.key_mode == "sparse_shuffled":
            # degree-preserving shuffle of the real bipartite matrix (binary), weights permuted
            E0 = torch.tensor(cfg.connectome_matrix, dtype=torch.float32)
            E0 = _subsample_rows(E0, n_kc, g)
            E = _shuffle_bipartite(E0, g)
        else:  # sparse_random: uniform PN choice, same fan-in for every KC
            E = torch.zeros(n_kc, n_pn)
            for i in range(n_kc):
                idx = torch.randperm(n_pn, generator=g)[: cfg.fan_in]
                E[i, idx] = 1.0
        # row-normalise so each KC sums its inputs with unit total weight
        E = E / E.sum(1, keepdim=True).clamp_min(1e-6)
        self.register_buffer("E", E)
        self.k = max(1, int(round(cfg.kwta_frac * n_kc)))
        self.kwta = cfg.kwta
        self.theta = nn.Parameter(torch.zeros(1))  # KC threshold (learned scalar)

    def forward(self, p: torch.Tensor) -> torch.Tensor:  # p: (..., n_pn) -> dense (..., n_kc)
        h = F.relu(p @ self.E.t() - self.theta)
        if self.kwta:
            top, idx = h.topk(self.k, dim=-1)
            mask = torch.zeros_like(h).scatter_(-1, idx, 1.0)
            h = h * mask
        return F.normalize(h, dim=-1)

    def forward_sparse(self, p: torch.Tensor):
        """k-WTA code as (values, indices) of the k active KCs; values L2-normalised."""
        h = F.relu(p @ self.E.t() - self.theta)
        val, idx = h.topk(self.k, dim=-1)
        val = F.normalize(val, dim=-1)
        return val, idx


def _subsample_rows(E: torch.Tensor, n: int, g: torch.Generator) -> torch.Tensor:
    """Seeded random subset of KCs (rows) when the model uses fewer KCs than the hemisphere has."""
    if E.shape[0] == n:
        return E
    idx = torch.randperm(E.shape[0], generator=g)[:n] if E.shape[0] > n else torch.randint(0, E.shape[0], (n,), generator=g)
    return E[idx]


def _shuffle_bipartite(E: torch.Tensor, g: torch.Generator, attempts_per_edge: int = 10) -> torch.Tensor:
    A = (E > 0)
    rows, cols = A.nonzero(as_tuple=True)
    rows, cols = rows.clone(), cols.clone()
    n_edges = len(rows); present = set((rows * 10_000 + cols).tolist())
    ii = torch.randint(0, n_edges, (attempts_per_edge * n_edges,), generator=g).tolist()
    jj = torch.randint(0, n_edges, (attempts_per_edge * n_edges,), generator=g).tolist()
    for i, j in zip(ii, jj):
        if i == j:
            continue
        a, b, c, d = rows[i].item(), cols[i].item(), rows[j].item(), cols[j].item()
        k1, k2 = a * 10_000 + d, c * 10_000 + b
        if k1 in present or k2 in present or b == d or a == c:
            continue
        present.discard(a * 10_000 + b); present.discard(c * 10_000 + d); present.add(k1); present.add(k2)
        cols[i], cols[j] = d, b
    out = torch.zeros_like(E)
    w = E[A][torch.randperm(n_edges, generator=g)]
    out[rows, cols] = w
    return out


class FastWeightMemory(nn.Module):
    def __init__(self, cfg: FWConfig):
        super().__init__()
        self.cfg = cfg
        H, dv, d = cfg.n_heads, cfg.d_value, cfg.d_model
        self.sparse = cfg.key_mode.startswith("sparse")
        if self.sparse:
            self.dk = cfg.n_kc
            self.pn = nn.Linear(d, H * cfg.n_pn)                  # learned "antennal lobe" -> PN types
            self.pn_q = None if cfg.shared_query else nn.Linear(d, H * cfg.n_pn)
            self.expand = nn.ModuleList([SparseExpansion(_with_seed(cfg, cfg.seed + h)) for h in range(H)])
        elif cfg.key_mode == "dense_expanded":  # dense learned projection to n_kc, no k-WTA (ablation)
            self.dk = cfg.n_kc
            self.k_proj = nn.Linear(d, H * self.dk); self.q_proj = nn.Linear(d, H * self.dk)
        else:
            self.dk = cfg.d_key
            self.k_proj = nn.Linear(d, H * self.dk); self.q_proj = nn.Linear(d, H * self.dk)
        self.v_proj = nn.Linear(d, H * dv)
        self.o_proj = nn.Linear(H * dv, d)
        if cfg.gate == "learned":
            self.beta_proj = nn.Linear(d, H)
        elif cfg.gate == "error":
            self.err_a = nn.Parameter(torch.zeros(H)); self.err_b = nn.Parameter(torch.zeros(H))
        if cfg.decay == "learned":
            self.alpha_proj = nn.Linear(d, H)
            nn.init.constant_(self.alpha_proj.bias, 4.0)   # start near alpha = 1 (no forgetting)
        self.norm = nn.RMSNorm(dv) if cfg.out_norm else nn.Identity()
        self.conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d, padding=cfg.conv_kernel - 1) if cfg.conv_kernel > 1 else None

    def _pre(self, x):
        if self.conv is None:
            return x
        T = x.shape[1]
        return F.silu(self.conv(x.transpose(1, 2))[:, :, :T].transpose(1, 2))

    # --- key / query codes -------------------------------------------------------------
    def _codes(self, x):
        B, T, _ = x.shape; H = self.cfg.n_heads
        if self.sparse:
            p = self.pn(x).view(B, T, H, self.cfg.n_pn)
            k = torch.stack([self.expand[h](p[:, :, h]) for h in range(H)], 2)        # (B,T,H,n_kc)
            if self.pn_q is None:
                q = k
            else:
                pq = self.pn_q(x).view(B, T, H, self.cfg.n_pn)
                q = torch.stack([self.expand[h](pq[:, :, h]) for h in range(H)], 2)
        else:
            k = F.normalize(F.silu(self.k_proj(x)).view(B, T, H, self.dk), dim=-1)
            q = F.normalize(F.silu(self.q_proj(x)).view(B, T, H, self.dk), dim=-1)
        return k, q

    def _sparse_codes(self, x):
        B, T, _ = x.shape; H = self.cfg.n_heads
        p = self.pn(x).view(B, T, H, self.cfg.n_pn)
        kv, ki = zip(*[self.expand[h].forward_sparse(p[:, :, h]) for h in range(H)])
        kv, ki = torch.stack(kv, 2), torch.stack(ki, 2)                         # (B,T,H,k)
        if self.pn_q is None:
            return kv, ki, kv, ki
        pq = self.pn_q(x).view(B, T, H, self.cfg.n_pn)
        qv, qi = zip(*[self.expand[h].forward_sparse(pq[:, :, h]) for h in range(H)])
        return kv, ki, torch.stack(qv, 2), torch.stack(qi, 2)

    def _forward_sparse(self, x, state):
        """Same recurrence as the dense path, but memory rows are touched only for the k
        active KCs (gather / scatter_add), and per-head decay is carried as a cumulative
        scalar so no O(n_kc * d_v) work happens per token. This is the O(k * d_v) cost that
        `fast_weight_flops` calls 'effective'."""
        cfg = self.cfg; B, T, _ = x.shape; H, dv = cfg.n_heads, cfg.d_value
        x = self._pre(x)
        kv, ki, qv, qi = self._sparse_codes(x)
        v = self.v_proj(x).view(B, T, H, dv)
        beta = torch.sigmoid(self.beta_proj(x)) if cfg.gate == "learned" else None
        alpha = torch.sigmoid(self.alpha_proj(x)) if cfg.decay == "learned" else None
        M = state if state is not None else x.new_zeros(B, H, self.dk, dv, dtype=torch.float32)       # stored unscaled; fp32 under autocast
        c = x.new_ones(B, H)                                                       # M_true = c * M
        outs = []
        for t in range(T):
            kvt, kit, qvt, qit, vt = kv[:, t], ki[:, t], qv[:, t], qi[:, t], v[:, t]
            rows_k = M.gather(2, kit[..., None].expand(-1, -1, -1, dv))            # (B,H,k,dv)
            pred = c[..., None] * (kvt[..., None] * rows_k).sum(2)
            err = pred - vt
            if cfg.gate == "learned":
                bt = beta[:, t]
            elif cfg.gate == "error":
                bt = torch.sigmoid(self.err_a * torch.log(err.pow(2).mean(-1) + 1e-6) + self.err_b)
            else:
                bt = x.new_full((B, H), cfg.beta_const)
            if self.pn_q is None:
                o = pred
            else:
                rows_q = M.gather(2, qit[..., None].expand(-1, -1, -1, dv))
                o = c[..., None] * (qvt[..., None] * rows_q).sum(2)
            if alpha is not None:
                c = c * alpha[:, t]
            upd = kvt[..., None] * (err if cfg.write == "delta" else -vt)[:, :, None, :]   # (B,H,k,dv)
            upd = -(bt / c)[..., None, None] * upd
            M = M.scatter_add(2, kit[..., None].expand(-1, -1, -1, dv), upd)
            if (t + 1) % 64 == 0:                                                  # renormalise stored scale
                M = M * c[..., None, None]; c = torch.ones_like(c)
            outs.append(o)
        M = M * c[..., None, None]
        o = torch.stack(outs, 1)
        o = self.norm(o).reshape(B, T, H * dv)
        return self.o_proj(o), M

    def _forward_lowrank(self, x):
        """Exact re-formulation used for training from an empty memory.

        M_t = sum_{s<=t} (c_t / c_s) k_s u_s^T with u_s = -beta_s * err_s (delta) or
        beta_s * v_s (hebb), c_t = prod alpha. Reads and predictions become T x T
        overlap-weighted sums over past writes (the fast-weight / linear-attention
        duality), so the sequential loop only touches O(t * d_v) tensors per step instead
        of copying the whole (n_kc x d_v) memory. Numerically identical to the recurrent
        paths (tested), including the error-dependent gate, which keeps the loop sequential.
        Cost accounting for compute matching still uses `fast_weight_flops`, i.e. the
        recurrent form that would be used at inference."""
        cfg = self.cfg; B, T, _ = x.shape; H, dv = cfg.n_heads, cfg.d_value
        x = self._pre(x)
        k, q = self._codes(x)                                                       # (B,T,H,dk) dense codes
        v = self.v_proj(x).view(B, T, H, dv)
        G = torch.einsum("bthk,bshk->bhts", k, k)                                    # key-key overlaps
        Gq = G if (self.sparse and self.pn_q is None) else torch.einsum("bthk,bshk->bhts", q, k)
        if cfg.decay == "learned":
            logc = torch.log(torch.sigmoid(self.alpha_proj(x)).clamp_min(1e-4)).cumsum(1).transpose(1, 2)   # (B,H,T), c_t = prod_{r<=t} alpha_r
            logc_prev = torch.cat([torch.zeros_like(logc[:, :, :1]), logc[:, :, :-1]], 2)                  # reads at t use M_{t-1}
            decay = torch.exp(logc_prev[:, :, :, None] - logc[:, :, None, :])                             # (B,H,t,s) = c_{t-1}/c_s
            G = G * decay; Gq = Gq * decay
        past = torch.tril(torch.ones(T, T, device=x.device), diagonal=-1)             # s < t strictly
        G = G * past; Gq = Gq * past
        beta = torch.sigmoid(self.beta_proj(x)) if cfg.gate == "learned" else None
        us, outs = [], []
        for t in range(T):
            if t == 0:
                pred = x.new_zeros(B, H, dv); o = pred
            else:
                U = torch.stack(us, 2)                                                 # (B,H,t,dv)
                pred = torch.einsum("bhs,bhsv->bhv", G[:, :, t, :t], U)
                o = torch.einsum("bhs,bhsv->bhv", Gq[:, :, t, :t], U)
            vt = v[:, t]; err = pred - vt
            if cfg.gate == "learned":
                bt = beta[:, t]
            elif cfg.gate == "error":
                bt = torch.sigmoid(self.err_a * torch.log(err.pow(2).mean(-1) + 1e-6) + self.err_b)
            else:
                bt = x.new_full((B, H), cfg.beta_const)
            us.append(-bt[..., None] * (err if cfg.write == "delta" else -vt))
            outs.append(o)
        U = torch.stack(us, 2)
        # final memory (for callers that want the state): sum_s (c_T/c_s) k_s u_s^T
        w_last = (torch.exp(logc[:, :, T - 1, None] - logc) if cfg.decay == "learned" else x.new_ones(B, H, T))
        M = torch.einsum("bhs,bshk,bhsv->bhkv", w_last, k, U)
        o = torch.stack(outs, 1)
        o = self.norm(o).reshape(B, T, H * dv)
        return self.o_proj(o), M

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        cfg = self.cfg; B, T, _ = x.shape; H, dv = cfg.n_heads, cfg.d_value
        if state is None and getattr(self, "use_lowrank", False):   # exact alternative; slower on CPU
            return self._forward_lowrank(x)
        if self.sparse and cfg.kwta:
            return self._forward_sparse(x, state)
        x = self._pre(x)
        k, q = self._codes(x)
        v = self.v_proj(x).view(B, T, H, dv)
        beta = torch.sigmoid(self.beta_proj(x)) if cfg.gate == "learned" else None          # (B,T,H)
        alpha = torch.sigmoid(self.alpha_proj(x)) if cfg.decay == "learned" else None       # (B,T,H)
        M = state if state is not None else x.new_zeros(B, H, self.dk, dv, dtype=torch.float32)         # fp32 state under autocast (as FlyAI's)
        outs = []
        for t in range(T):
            kt, qt, vt = k[:, t], q[:, t], v[:, t]                                          # (B,H,dk),(B,H,dv)
            pred = torch.einsum("bhk,bhkv->bhv", kt, M)                                     # memory's prediction for this key
            err = pred - vt
            if cfg.gate == "learned":
                bt = beta[:, t]
            elif cfg.gate == "error":                                                       # DAN-like: gate from prediction error
                bt = torch.sigmoid(self.err_a * torch.log(err.pow(2).mean(-1) + 1e-6) + self.err_b)
            else:
                bt = x.new_full((B, H), cfg.beta_const)
            at = alpha[:, t] if alpha is not None else None
            # read (with query code) BEFORE write, so a token cannot read its own value
            o = torch.einsum("bhk,bhkv->bhv", qt, M)
            if at is not None:
                M = M * at[..., None, None]
            if cfg.write == "delta":
                M = M - bt[..., None, None] * torch.einsum("bhk,bhv->bhkv", kt, err)
            else:
                M = M + bt[..., None, None] * torch.einsum("bhk,bhv->bhkv", kt, vt)
            outs.append(o)
        o = torch.stack(outs, 1)                                                            # (B,T,H,dv)
        o = self.norm(o).reshape(B, T, H * dv)
        return self.o_proj(o), M


def _with_seed(cfg: FWConfig, seed: int) -> FWConfig:
    c = FWConfig(**{**cfg.__dict__}); c.seed = seed; return c


def fast_weight_flops(cfg: FWConfig) -> dict:
    """Analytic multiply-adds per token (x2 for FLOPs) for the recurrent part, counting the
    sparse code at its *effective* cost (only k active KCs touch memory) and at its dense
    cost (what a naive implementation pays). Projections are counted separately."""
    H, dv, d = cfg.n_heads, cfg.d_value, cfg.d_model
    if cfg.key_mode.startswith("sparse"):
        k_act = max(1, int(round(cfg.kwta_frac * cfg.n_kc)))
        expansion = cfg.n_kc * cfg.fan_in + cfg.n_pn * d          # E is fan_in-sparse; PN projection dense
        mem_eff = 3 * k_act * dv                                    # pred, read, write over active rows
        mem_dense = 3 * cfg.n_kc * dv
        proj = d * H * dv + H * dv * d + d * cfg.conv_kernel
        return {"recurrent_effective": H * (expansion + mem_eff), "recurrent_dense": H * (expansion + mem_dense),
                "projections": proj, "state_size": H * cfg.n_kc * dv}
    dk = cfg.n_kc if cfg.key_mode == "dense_expanded" else cfg.d_key
    mem = 3 * dk * dv
    proj = 2 * d * H * dk + d * H * dv + H * dv * d + d * cfg.conv_kernel
    return {"recurrent_effective": H * mem, "recurrent_dense": H * mem, "projections": proj, "state_size": H * dk * dv}


def matched_dense_dkey(mb_cfg: FWConfig, match: str = "effective") -> int:
    """d_key for a dense fast-weight head whose per-token cost (recurrent + projections)
    equals the MB layer's, at the same d_model, n_heads, d_value."""
    f = fast_weight_flops(mb_cfg); target = f[f"recurrent_{match}"] + f["projections"]
    H, dv, d = mb_cfg.n_heads, mb_cfg.d_value, mb_cfg.d_model
    # cost(dk) = H*(3*dk*dv) + 2*d*H*dk + d*H*dv + H*dv*d  ->  solve for dk
    dk = (target - d * H * dv - H * dv * d - d * mb_cfg.conv_kernel) / (H * (3 * dv + 2 * d))
    return max(4, int(round(dk)))
