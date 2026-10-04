"""System-level model for phase 3 (docs/phase3/DESIGN.md section 3): a hybrid decoder whose memory
layers are Gated DeltaNet plus per-address STATES that record teaching history (a confirmation
vector, a familiarity-gated contradiction mass, ATTEST's evidence mass), read at the same address as
the content and shown to a learned gate through fixed input slots.

The control is the same layer with the slots zeroed (`states=()`): equal parameters, same kernels.
With the slots zeroed and gamma tied to beta the layer is Gated DeltaNet with a learned gate on the
delta rule's own intermediates.

Two execution modes, both differentiable:
  sequential  the per-token reference (exact; slow; the verification oracle)
  chunked     chunkwise UT-transform for the content and confirmation stores (exact given the
              gates); the gate INPUTS are computed from the chunk-start states (the stated
              approximation, measured by tests/test_hybrid.py against the reference)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

STATES = ("nconf", "zc", "a")


@dataclass
class LayerConfig:
    d_model: int = 256
    n_heads: int = 4
    d_k: int = 64
    d_v: int = 64
    d_c: int = 16                                 # confirmation-vector width (P: d_v -> d_c)
    d_g: int = 16                                 # per-head projection of the input shown to the gate
    n_traces: int = 1                             # 1 = one content store; 2 = a slow and a labile store with a learned readout (the form E4 supported)
    readout_hidden: int = 16                      # the learned readout's hidden width (n_traces = 2)
    gate_hidden: int = 32
    states: Sequence[str] = ("nconf", "zc")       # () = the free control (slots read 0)
    n_state_slots: int = 4
    tie_gamma: bool = False                       # True: erase gate = write gate (Gated DeltaNet's coupling)
    taus: Optional[Sequence[float]] = None        # per-head decay ladder; None = (8, 32, 128, 512, ...) cycled
    conv_kernel: int = 4
    confirm_tol: float = 1.5
    familiar_norm: float = 0.25
    rho_track: float = 0.9
    chunk: int = 0                                # 0 = sequential reference; > 0 = chunk size
    refine: Optional[int] = None                  # chunked: maximum gate-input refinement passes; None = chunk - 1, which makes the iteration EXACT
                                                  # (the intra-chunk dependency is causal, so after p passes the first p + 1 tokens' gates are exact)
    refine_tol: float = 0.0                       # chunked: > 0 stops refining early when the gates change by less than this (costs a device sync per pass); 0 = fixed pass budget
    x_only_gate: bool = False                     # the gate sees only the input features (Gated DeltaNet's own gate): no state dependence, no refinement needed
    address: str = "dense"                        # "dense": L2-normalised keys (Gated DeltaNet); "sparse": the Kenyon-cell code (a FIXED random expansion of the
                                                  # key, each cell sampling fan_in key dims, then k-WTA: k_active of n_cells on, unit norm); "sparse_learned": a
                                                  # learned expansion + k-WTA; "expand_dense": the fixed random expansion WITHOUT k-WTA (is it the sparsity?);
                                                  # "product": product-key addressing (learned sub-key codebooks for the two halves of the key, a cell per pair,
                                                  # exact top-k_active of the summed scores, softmax weights, unit norm: the addressing of product-key memories)
    n_cells: int = 1024                           # expanded address cells per head (sparse addresses)
    pk_n1: int = 32                               # product keys: size of the first sub-key codebook (the second has n_cells // pk_n1)
    k_active: int = 64                            # cells on per token (6.25% coding level at 64 of 1,024; the fly's is ~5%)
    fan_in: int = 7                               # key dims sampled by each cell in the fixed expansion (the fly's claw count is ~6-7)
    sense_block: int = 0                          # 0: the gate senses the store per token (exact, needs refinement in the chunked form); b > 0: the gate senses the
                                                  # store and the states as of the start of each block of b tokens (block-delayed sensing; with chunk = b the chunked
                                                  # form is exact in ONE round, like Gated DeltaNet's own gates). Everything else stays per token.
    compile_chunk: bool = False                   # torch.compile the per-chunk step once and reuse it for every chunk (a runtime speed option; identical arithmetic)
    compile_mode: str = "default"                 # torch.compile mode for the chunk step ("default", "reduce-overhead", "max-autotune")
    slot_path: bool = True                        # show the states to the gate (False: slots read 0 although states are tracked)
    feature_path: bool = True                     # show the states to the output projection (False: feature columns read 0)
    a_floor: bool = True                          # False: the evidence mass without its crosstalk floor (the "one uncorrected Hebbian column" control)
    const_gate: Optional[Sequence] = None         # fixed gates: (beta, gamma, lambda) for every store, or one triple per store ((b0,g0,l0),(b1,g1,l1))

    def __post_init__(self):
        self.states = tuple(self.states)
        if any(s not in STATES for s in self.states): raise ValueError(f"states must be a subset of {STATES}")
        if len(self.states) > self.n_state_slots: raise ValueError("more states than slots")
        if self.taus is None:                       # architectural ladder, log-spaced from 8 to 2048 tokens across the heads (fixed, not learned)
            self.taus = tuple(float(t) for t in (8.0 * (256.0 ** (i / max(1, self.n_heads - 1))) for i in range(self.n_heads)))
        if len(self.taus) != self.n_heads: raise ValueError("one tau per head")
        if self.n_traces not in (1, 2): raise ValueError("n_traces must be 1 or 2")
        if self.address not in ADDRESSES: raise ValueError("unknown address")
        if self.address != "dense" and not (0 < self.k_active <= self.n_cells and 0 < self.fan_in <= self.d_k): raise ValueError("sparse address sizes")
        if self.address == "product" and (self.d_k % 2 or self.pk_n1 <= 0 or self.n_cells % self.pk_n1): raise ValueError("product keys: even d_k, pk_n1 dividing n_cells")
        if self.sense_block < 0: raise ValueError("sense_block must be >= 0")
        if self.sense_block > 0 and self.chunk > 0 and self.chunk != self.sense_block:
            raise ValueError("block-delayed sensing: the chunked form needs chunk == sense_block (the sequential form, chunk = 0, takes any block)")

    @property
    def passes_max(self) -> int:                  # refinement passes actually run in the chunked form (refine=None resolves per chunk size, so a config copied to another chunk size stays exact)
        if self.sense_block > 0: return 0         # block-delayed sensing: the chunk-start gate inputs ARE the model's gate inputs, one round is exact
        if self.refine is not None: return self.refine
        return 0 if self.x_only_gate else max(0, self.chunk - 1)

    @property
    def tracks_conf(self): return "nconf" in self.states or "zc" in self.states
    @property
    def tracks_a(self): return "a" in self.states
    @property
    def rows(self) -> int:                        # the memory's address dimension: d_k (dense keys) or n_cells (expanded codes)
        return self.d_k if self.address == "dense" else self.n_cells
    @property
    def active_rows(self) -> int:                 # rows a read or write touches per token: all of them for dense codes, k_active for k-WTA codes
        return self.k_active if self.address in SPARSE_CODES else self.rows


ADDRESSES = ("dense", "sparse", "sparse_learned", "expand_dense", "product")
SPARSE_CODES = ("sparse", "sparse_learned", "product")


def _raise_recompile_limit(n: int = 256):
    """torch.compile keeps at most `recompile_limit` (default 8) graphs per function and then silently runs it UNCOMPILED.
    Every layer configuration (and evaluation without gradients) is its own graph of the one `_chunk` function, so a
    process that trains many arms in turn (a grid) would fall back to eager after a few runs, about 5 times slower."""
    cfg = torch._dynamo.config
    for name in ("recompile_limit", "cache_size_limit"):                      # the name changed across PyTorch versions
        if hasattr(cfg, name): setattr(cfg, name, max(getattr(cfg, name), n))
    for name in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(cfg, name): setattr(cfg, name, max(getattr(cfg, name), 4 * n))


class ShortConv(nn.Module):
    """Causal depthwise conv (kernel 4) followed by SiLU, as in the fla linear-attention layers."""
    def __init__(self, d, k):
        super().__init__(); self.k = k; self.conv = nn.Conv1d(d, d, k, groups=d, bias=True)
    def forward(self, x):                                             # (B,T,d)
        y = self.conv(F.pad(x.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2)
        return F.silu(y)


class StateMemoryLayer(nn.Module):
    """One memory layer with R = n_traces content stores (R = 1: the single-store form; R = 2: the
    two-store form the module ladder supported, a slow and a labile trace with a learned readout)."""
    def __init__(self, c: LayerConfig):
        super().__init__(); self.c = c; H, dk, dv, dc, R = c.n_heads, c.d_k, c.d_v, c.d_c, c.n_traces
        self.wq = nn.Linear(c.d_model, H * dk, bias=False); self.wk = nn.Linear(c.d_model, H * dk, bias=False); self.wv = nn.Linear(c.d_model, H * dv, bias=False)
        self.cq, self.ck, self.cv = ShortConv(H * dk, c.conv_kernel), ShortConv(H * dk, c.conv_kernel), ShortConv(H * dv, c.conv_kernel)
        self.wg = nn.Linear(c.d_model, H * c.d_g, bias=False)                                   # the input features the gate sees
        d_in = c.d_g + dv + R * dv + 2 * R + 1 + c.n_state_slots                                # x-features, value, error per trace, log e2 and log ||vhat|| per trace, log(pi e2), slots
        self.gate = nn.Sequential(nn.Linear(d_in, c.gate_hidden), nn.Tanh(), nn.Linear(c.gate_hidden, 3 * R))
        with torch.no_grad():
            bias = torch.tensor([0.0, 0.0, -2.0] * R)                                           # forget drive starts small (long memory)
            if R == 2: bias[0], bias[3] = -1.0, 2.0                                             # architectural asymmetry: a slow trace and a labile trace at initialisation
            self.gate[2].bias.copy_(bias)
        self.P = nn.Parameter(torch.randn(H, dc, dv) / math.sqrt(dv))                          # confirmation projection per head
        self.wo = nn.Linear(H * dv + 2 * H, c.d_model, bias=False)                              # reads + (nconf, zc) features per head
        self.register_buffer("tau", torch.tensor(list(c.taus), dtype=torch.float32))
        self.norm_o = nn.RMSNorm(dv) if hasattr(nn, "RMSNorm") else nn.LayerNorm(dv)
        self.rho = ((c.k_active / c.n_cells) if c.address in SPARSE_CODES else 1.0 / math.sqrt(c.rows)) if c.a_floor else 0.0
        if R == 2:                                                                              # learned readout: mix the two reads at the query address
            self.readout = nn.Sequential(nn.Linear(3 + c.n_state_slots, c.readout_hidden), nn.Tanh(), nn.Linear(c.readout_hidden, 1))
        if c.address in ("sparse", "expand_dense"):                                               # the fixed random expansion (a buffer: no parameters)
            g = torch.Generator().manual_seed(1234); E = torch.zeros(c.n_cells, dk)
            for i in range(c.n_cells): E[i, torch.randperm(dk, generator=g)[:c.fan_in]] = 1.0 / c.fan_in
            self.register_buffer("E", E)
        elif c.address == "sparse_learned":
            self.E = nn.Parameter(torch.randn(c.n_cells, dk) / math.sqrt(dk))
        elif c.address == "product":                                                              # sub-key codebooks for the two halves of the key
            h = dk // 2; self.C1 = nn.Parameter(torch.randn(c.pk_n1, h) / math.sqrt(h)); self.C2 = nn.Parameter(torch.randn(c.n_cells // c.pk_n1, h) / math.sqrt(h))
        self.monitor = False; self.last_feats = None; self.last_passes = 0; self.last_passes_mean = 0.0   # dead-feature monitor and solve-round counters of the last forward

    # ---- state ------------------------------------------------------------------------
    def code(self, x):
        """Address code of normalised (B,H,T,d_k) keys or queries: the identity (dense), or the expansion followed by
        k-WTA (sparse): k_active non-negative entries of n_cells, unit norm (gradient flows through the winners)."""
        c = self.c
        if c.address == "dense": return x
        if c.address == "product":                   # the score of cell (i, j) is <x1, C1_i> + <x2, C2_j>; the exact top-k over all cells equals product-key
            h = c.d_k // 2                           # memory's two-stage search (every top-k pair has both halves in their own top-k), so the grid is scored directly
            z = ((x[..., :h] @ self.C1.t())[..., :, None] + (x[..., h:] @ self.C2.t())[..., None, :]).flatten(-2)
            top, idx = z.topk(c.k_active, dim=-1)
            return F.normalize(torch.zeros_like(z).scatter(-1, idx, F.softmax(top, dim=-1)), dim=-1)
        z = x @ self.E.t()
        if c.address == "expand_dense": return F.normalize(z, dim=-1)
        top, idx = z.topk(c.k_active, dim=-1)
        return F.normalize(torch.zeros_like(z).scatter(-1, idx, F.relu(top) + 1e-6), dim=-1)

    def init_state(self, B, like):
        c = self.c; H, dk, dv, dc, R = c.n_heads, c.rows, c.d_v, c.d_c, c.n_traces; z = lambda *s: like.new_zeros(B, H, *s)
        S = z(dk, dv) if R == 1 else z(R, dk, dv)                                                # R = 1 keeps the single-store layout
        return {"S": S, "C": z(dk, dc), "Zc": z(dk), "Z": z(dk), "sig": z(), "ta": like.new_ones(B, H), "tb": like.new_full((B, H), 1.0 / dv)}

    def _S_list(self, S): return [S] if self.c.n_traces == 1 else [S[:, :, r] for r in range(self.c.n_traces)]
    def _S_pack(self, Ss): return Ss[0] if self.c.n_traces == 1 else torch.stack(Ss, 2)

    def _reads(self, st, k, vhat):
        """Per-token state reads {name: (B,H)} at address k from a state dict and (B,H,dv) predictions of the labile trace."""
        c = self.c; out = {}
        if c.tracks_conf:
            cread = torch.einsum("bhkc,bhk->bhc", st["C"], k); pv = torch.einsum("hcv,bhv->bhc", self.P, vhat)
            out["nconf"] = (cread * F.normalize(pv, dim=-1, eps=1e-3)).sum(-1)
            out["zc"] = torch.log1p(F.relu(torch.einsum("bhk,bhk->bh", st["Zc"], k)))
        if c.tracks_a: out["a"] = torch.log1p(F.softplus(torch.einsum("bhk,bhk->bh", st["Z"], k) - self.rho * st["sig"]))
        return out

    def _slots_from(self, reads, like):
        c = self.c; slots = like.new_zeros(*like.shape[:-1], c.n_state_slots)
        for j, name in enumerate(c.states): slots[..., j] = reads[name]
        return slots

    def _gates(self, xg, v, es, vhats, pi, slots):
        """Per-trace (beta, gamma, lam), each (B,H,...,R), from the x-features (with gradient) and detached sensing features."""
        c = self.c; R = c.n_traces
        e2s = [e.pow(2).mean(-1) for e in es]; rns = [vh.norm(dim=-1) for vh in vhats]
        feats = torch.cat([xg, v.detach()] + [e.detach() for e in es] + [torch.log(e2.detach() + 1e-6)[..., None] for e2 in e2s]
                          + [torch.log(rn.detach() + 1e-6)[..., None] for rn in rns] + [torch.log((pi * e2s[-1]).detach() + 1e-6)[..., None], slots if c.slot_path else slots * 0], -1)
        if c.x_only_gate: feats = torch.cat([xg, torch.zeros_like(feats[..., xg.shape[-1]:])], -1)
        if c.const_gate is not None:
            cg = c.const_gate; per = [tuple(cg)] * R if not isinstance(cg[0], (tuple, list)) else [tuple(t) for t in cg]
            if len(per) != R: raise ValueError("const_gate: one triple, or one per store")
            mk = lambda i: torch.stack([e2s[0].new_full(e2s[0].shape, float(t[i])) for t in per], -1)
            return mk(0), mk(1), mk(2)
        g = self.gate(feats).view(*feats.shape[:-1], R, 3)
        beta = torch.sigmoid(g[..., 0]); gamma = beta if c.tie_gamma else torch.sigmoid(g[..., 1]); lam = F.softplus(g[..., 2])
        return beta, gamma, lam

    def _mix(self, ys, reads_q):
        """Readout: the single read (R = 1) or a learned mixture of the two reads (R = 2) from their agreement, norms and the state slots at the query."""
        if self.c.n_traces == 1: return ys[0]
        y0, y1 = ys; slots = self._slots_from(reads_q, y1) if self.c.states else y1.new_zeros(*y1.shape[:-1], self.c.n_state_slots)
        feats = torch.cat([F.cosine_similarity(y0, y1, dim=-1)[..., None], torch.log(y0.norm(dim=-1) + 1e-6)[..., None], torch.log(y1.norm(dim=-1) + 1e-6)[..., None],
                           (slots if self.c.slot_path else slots * 0).detach()], -1)
        lam = torch.sigmoid(self.readout(feats)); return y0 * (1 - lam) + y1 * lam

    # ---- sequential reference ---------------------------------------------------------
    def _forward_sequential(self, q, k, v, xg, st):
        c = self.c; T = q.shape[2]; R = c.n_traces; ys, f_nc, f_zc = [], [], []
        Ss = self._S_list(st["S"]); C, Zc, Z, sig, ta, tb = (st[n] for n in ("C", "Zc", "Z", "sig", "ta", "tb")); tau = self.tau[None, :]
        snap = None
        for t in range(T):
            qt, kt, vt, xt = q[:, :, t], k[:, :, t], v[:, :, t], xg[:, :, t]
            vhats = [torch.einsum("bhkv,bhk->bhv", S, kt) for S in Ss]; es = [vt - vh for vh in vhats]; vhat = vhats[-1]; e2 = es[-1].pow(2).mean(-1); rn = vhat.norm(dim=-1)
            reads = self._reads({"C": C, "Zc": Zc, "Z": Z, "sig": sig}, kt, vhat)
            if c.sense_block > 0:                                                                   # block-delayed sensing: the gate reads the store and the states as of the block start
                if t % c.sense_block == 0: snap = (Ss, {"C": C, "Zc": Zc, "Z": Z, "sig": sig}, ta, tb)
                g_vh = [torch.einsum("bhkv,bhk->bhv", S, kt) for S in snap[0]]; g_es = [vt - vh for vh in g_vh]
                slots = self._slots_from(self._reads(snap[1], kt, g_vh[-1]), kt); g_pi = snap[2] / snap[3].clamp_min(1e-8)
            else:
                g_vh, g_es, slots, g_pi = vhats, es, self._slots_from(reads, kt), ta / tb.clamp_min(1e-8)
            beta, gamma, lam = self._gates(xt, vt, g_es, g_vh, g_pi, slots.detach()); alpha = torch.exp(-lam / tau[..., None])   # (B,H,R)
            rs = [alpha[..., r, None] * vhats[r] for r in range(R)]                                 # the decayed predictions alpha S^T k
            Ss = [alpha[..., r, None, None] * Ss[r] + torch.einsum("bhk,bhv->bhkv", kt, beta[..., r, None] * vt - gamma[..., r, None] * rs[r]) for r in range(R)]
            yr = [torch.einsum("bhkv,bhk->bhv", S, qt) for S in Ss]; r = rs[-1]; b_l = beta[..., -1]
            expected = (tb / ta.clamp_min(1e-8)).clamp_min(1e-4); agree = ((vt - r).pow(2).mean(-1) < c.confirm_tol * expected).float()
            ta = c.rho_track * ta + 0.5; tb = c.rho_track * tb + 0.5 * (vt - r).detach().pow(2).mean(-1)     # tracks the labile trace's decayed error
            if c.tracks_conf:
                cread = torch.einsum("bhkc,bhk->bhc", C, kt); target = agree[..., None] * (cread + F.normalize(torch.einsum("hcv,bhv->bhc", self.P, vt), dim=-1))
                C = C + torch.einsum("bhk,bhc->bhkc", kt, target - cread)
                ev = (vt - r).detach().abs().mean(-1); contradict = (1 - agree) * (rn.detach() >= c.familiar_norm).float()
                Zc = Zc + (b_l.detach() * ev * contradict)[..., None] * kt
            if c.tracks_a:
                ev = (vt - r).detach().abs().mean(-1); Z = Z + (b_l.detach() * ev)[..., None] * kt; sig = sig + b_l.detach() * ev
            reads_q = self._reads({"C": C, "Zc": Zc, "Z": Z, "sig": sig}, qt, yr[-1]) if R == 2 and c.states else {}
            ys.append(self._mix(yr, reads_q))
            zero = e2 * 0; f_nc.append(reads["nconf"] if "nconf" in c.states else zero); f_zc.append(reads["zc"] if "zc" in c.states else zero)
        st = {"S": self._S_pack(Ss), "C": C, "Zc": Zc, "Z": Z, "sig": sig, "ta": ta, "tb": tb}
        return torch.stack(ys, 2), torch.stack(f_nc, 2), torch.stack(f_zc, 2), st

    # ---- chunked ------------------------------------------------------------------------
    def _reads_chunk(self, C, Zc, Z, sig, kc, vhat, uC=None, wzc=None, wz=None, at=None, inclusive=False):
        """Per-token state reads {name: (B,H,n)} for a chunk at addresses `at` (default: the keys), from the
        chunk-start states plus, when the intra-chunk write coefficients are given, the exact intra-chunk contributions
        (exclusive of the token's own write unless `inclusive`)."""
        c = self.c; B, H, n, _ = kc.shape; at = kc if at is None else at
        L0 = torch.tril(torch.einsum("bhtk,bhsk->bhts", at, kc), 0 if inclusive else -1); out = {}
        if c.tracks_conf:
            cread = torch.einsum("bhkc,bhtk->bhtc", C, at)
            if uC is not None: cread = cread + torch.einsum("bhts,bhsc->bhtc", L0, uC)
            pv = torch.einsum("hcv,bhtv->bhtc", self.P, vhat); out["nconf"] = (cread * F.normalize(pv, dim=-1, eps=1e-3)).sum(-1)
            zr = torch.einsum("bhk,bhtk->bht", Zc, at)
            if wzc is not None: zr = zr + torch.einsum("bhts,bhs->bht", L0, wzc)
            out["zc"] = torch.log1p(F.relu(zr))
        if c.tracks_a:
            zr = torch.einsum("bhk,bhtk->bht", Z, at); sg = sig[..., None].expand(-1, -1, n)
            if wz is not None:
                zr = zr + torch.einsum("bhts,bhs->bht", L0, wz); sg = (sg + torch.cumsum(wz, -1)) if inclusive else (sg + torch.cumsum(wz, -1) - wz)
            out["a"] = torch.log1p(F.softplus(zr - self.rho * sg))
        return out

    def _forward_chunked(self, q, k, v, xg, st):
        """Chunkwise form. The R content stores are solved together in one batched triangular solve per
        refinement pass (a second solve handles the confirmation store); with refine_tol = 0 the pass
        count is fixed (no data-dependent break, hence no device synchronisation per pass)."""
        c = self.c; T = k.shape[2]; Cn = c.chunk; R = c.n_traces; ys, f_nc, f_zc = [], [], []
        S = st["S"] if R > 1 else st["S"][:, :, None]                                                  # (B,H,R,dk,dv) internally
        C, Zc, Z, sig, ta, tb = (st[n] for n in ("C", "Zc", "Z", "sig", "ta", "tb")); passes = []
        step = self._chunk
        if c.compile_chunk:                                                                              # one compiled graph per chunk shape, reused for every chunk
            if getattr(self, "_chunk_compiled", None) is None:
                _raise_recompile_limit(); self._chunk_compiled = torch.compile(self._chunk, dynamic=False, mode=c.compile_mode)
            step = self._chunk_compiled
        for s0 in range(0, T, Cn):
            y, fnc, fzc, S, C, Zc, Z, sig, ta, tb, rounds = step(q[:, :, s0:s0 + Cn], k[:, :, s0:s0 + Cn], v[:, :, s0:s0 + Cn], xg[:, :, s0:s0 + Cn], S, C, Zc, Z, sig, ta, tb)
            ys.append(y); f_nc.append(fnc); f_zc.append(fzc); passes.append(rounds)
        self.last_passes = max(passes); self.last_passes_mean = sum(passes) / len(passes)              # solve rounds per chunk: max and mean over the forward
        st = {"S": S if R > 1 else S[:, :, 0], "C": C, "Zc": Zc, "Z": Z, "sig": sig, "ta": ta, "tb": tb}
        return torch.cat(ys, 2), torch.cat(f_nc, 2), torch.cat(f_zc, 2), st

    def _chunk(self, qc, kc, vc, xc, S, C, Zc, Z, sig, ta, tb):
        """One chunk of the chunkwise form: a pure function of the chunk inputs and the chunk-start states
        (so it can be compiled once and reused), returning the chunk outputs, the chunk-end states and the solve rounds."""
        c = self.c; B, H, n, _ = kc.shape; R = c.n_traces; tau = self.tau[None, :, None, None]; rho = c.rho_track
        eye = torch.eye(n, device=kc.device, dtype=kc.dtype)[None, None, None]
        if True:
            KK = torch.einsum("bhtk,bhsk->bhts", kc, kc); L0 = torch.tril(KK, -1); QK0 = torch.einsum("bhtk,bhsk->bhts", qc, kc)
            KS0 = torch.einsum("bhtk,bhrkv->bhrtv", kc, S); QS0 = torch.einsum("bhtk,bhrkv->bhrtv", qc, S)                         # (B,H,R,n,dv)
            ar = torch.arange(n, device=kc.device, dtype=kc.dtype)
            Wr = torch.tril(rho ** (ar[:, None] - ar[None, :] - 1), -1)                                 # Wr[t,i] = rho^(t-1-i), i < t
            ta_t = rho ** ar * ta[..., None] + 0.5 * Wr.sum(-1)                                          # tracker BEFORE token t (exact)
            vhat_in = KS0; reads = self._reads_chunk(C, Zc, Z, sig, kc, vhat_in[:, :, -1]); pi_in = (ta / tb.clamp_min(1e-8))[..., None].expand(-1, -1, n)
            prev = None; rounds = 0
            for p in range(1 + c.passes_max):
                slots = self._slots_from(reads, kc) if c.states else kc.new_zeros(B, H, n, c.n_state_slots)
                beta, gamma, lam = self._gates(xc, vc, [vc - vhat_in[:, :, r_] for r_ in range(R)], [vhat_in[:, :, r_] for r_ in range(R)], pi_in, slots.detach())   # (B,H,n,R)
                if c.refine_tol > 0:                                                                     # optional early stop (costs one device sync per pass)
                    gates = torch.stack([beta, gamma, lam], -1).detach()
                    if prev is not None and (gates - prev).abs().max() < c.refine_tol: break
                    prev = gates
                rounds += 1
                bt, gt, lt = beta.transpose(-1, -2), gamma.transpose(-1, -2), lam.transpose(-1, -2)    # (B,H,R,n)
                lg = torch.cumsum(-lt / tau, dim=-1); G = torch.exp(lg)                                  # log-space cumulative decay per store
                D = torch.exp(torch.tril(lg[..., :, None] - lg[..., None, :], -1))                        # masked BEFORE exp: no overflow for t < s
                L = torch.tril(D * KK[:, :, None], -1); A = eye + gt[..., None] * L                       # (B,H,R,n,n)
                u = torch.linalg.solve_triangular(A, bt[..., None] * vc[:, :, None] - gt[..., None] * G[..., None] * KS0, upper=False, unitriangular=True)   # one batched solve for all stores
                rr = G[..., None] * KS0 + torch.einsum("bhrts,bhrsv->bhrtv", L, u)                        # exact alpha_t S_{t-1}^T k_t per store
                alpha = torch.exp(-lt / tau)                                                             # (B,H,R,n)
                e_ex = vc - rr[:, :, -1]; e2_ex = e_ex.pow(2).mean(-1); vhat_ex = rr / alpha[..., None]
                tb_t = rho ** ar * tb[..., None] + 0.5 * torch.einsum("ti,bhi->bht", Wr, e2_ex.detach())
                expected = (tb_t / ta_t.clamp_min(1e-8)).clamp_min(1e-4); agree = (e2_ex < c.confirm_tol * expected).float()
                ev = e_ex.detach().abs().mean(-1); b_l = bt[:, :, -1].detach(); uC = wzc = wz = None
                if c.tracks_conf:
                    b = 1 - agree; W = agree[..., None] * F.normalize(torch.einsum("hcv,bhtv->bhtc", self.P, vc), dim=-1); A0 = eye[:, :, 0] + b[..., None] * L0
                    KC0 = torch.einsum("bhtk,bhkc->bhtc", kc, C); uC = torch.linalg.solve_triangular(A0, W - b[..., None] * KC0, upper=False, unitriangular=True)
                    contradict = (1 - agree) * (vhat_ex[:, :, -1].norm(dim=-1).detach() >= c.familiar_norm).float(); wzc = b_l * ev * contradict
                if c.tracks_a: wz = b_l * ev
                reads = self._reads_chunk(C, Zc, Z, sig, kc, vhat_ex[:, :, -1], uC, wzc, wz); vhat_in = vhat_ex; pi_in = ta_t / tb_t.clamp_min(1e-8)
            Dq = torch.exp(torch.tril(lg[..., :, None] - lg[..., None, :], 0))
            yr = G[..., None] * QS0 + torch.einsum("bhrts,bhrsv->bhrtv", torch.tril(Dq * QK0[:, :, None], 0), u)                     # reads after the write, (B,H,R,n,dv)
            S = G[..., -1:, None] * S + torch.einsum("bhtk,bhrtv->bhrkv", kc, torch.exp(lg[..., -1:] - lg)[..., None] * u)
            ta = ta * rho ** n + 0.5 * (rho ** ar).sum(); tb = tb * rho ** n + 0.5 * torch.einsum("i,bhi->bh", rho ** (n - 1 - ar), e2_ex.detach())
            reads_q = self._reads_chunk(C, Zc, Z, sig, kc, yr[:, :, -1], uC, wzc, wz, at=qc, inclusive=True) if R == 2 and c.states else {}
            if c.tracks_conf:
                C = C + torch.einsum("bhtk,bhtc->bhkc", kc, uC); Zc = Zc + torch.einsum("bht,bhtk->bhk", wzc, kc)
            if c.tracks_a: Z = Z + torch.einsum("bht,bhtk->bhk", wz, kc); sig = sig + wz.sum(-1)
            y = self._mix([yr[:, :, r_] for r_ in range(R)], reads_q); zero = yr[:, :, 0, :, 0] * 0
            fnc = reads["nconf"] if "nconf" in c.states else zero; fzc = reads["zc"] if "zc" in c.states else zero
        return y, fnc, fzc, S, C, Zc, Z, sig, ta, tb, rounds

    # ---- forward ------------------------------------------------------------------------
    def forward(self, x, state=None):
        c = self.c; B, T, _ = x.shape; H = c.n_heads
        q = self.code(F.normalize(self.cq(self.wq(x)).view(B, T, H, c.d_k).transpose(1, 2), dim=-1))
        k = self.code(F.normalize(self.ck(self.wk(x)).view(B, T, H, c.d_k).transpose(1, 2), dim=-1))     # (B,H,T,rows)
        v = self.cv(self.wv(x)).view(B, T, H, c.d_v).transpose(1, 2)
        xg = self.wg(x).view(B, T, H, c.d_g).transpose(1, 2)
        st = self.init_state(B, x) if state is None else state
        y, fnc, fzc, st = (self._forward_chunked if c.chunk > 0 else self._forward_sequential)(q, k, v, xg, st)
        if self.monitor: self.last_feats = (fnc.detach(), fzc.detach())                          # (B,H,T) each
        y = self.norm_o(y).transpose(1, 2).reshape(B, T, H * c.d_v)
        feats = torch.cat([fnc.transpose(1, 2), fzc.transpose(1, 2)], -1)                      # (B,T,2H); zero in the free arm
        if not c.feature_path: feats = feats * 0
        return self.wo(torch.cat([y, feats], -1)), st


class SlidingWindowAttention(nn.Module):
    def __init__(self, d, n_heads, window, rope=True):
        super().__init__(); self.h = n_heads; self.w = window; self.rope = rope; self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, d = x.shape; q, k, v = self.qkv(x).view(B, T, 3, self.h, d // self.h).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v)); pos = torch.arange(T, device=x.device)
        if self.rope: q, k = _rope(q, pos), _rope(k, pos)
        i = pos[:, None]; j = pos[None, :]; mask = (j <= i) & (i - j < self.w)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(o.transpose(1, 2).reshape(B, T, d))


def _rope(x, pos):
    d = x.shape[-1]; half = d // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    ang = pos[:, None].float() * freqs[None]; cs, sn = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cs - x2 * sn, x1 * sn + x2 * cs], -1)


class SwiGLU(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__(); h = int(mult * d * 2 / 3); self.w1 = nn.Linear(d, h, bias=False); self.w2 = nn.Linear(d, h, bias=False); self.w3 = nn.Linear(h, d, bias=False)
    def forward(self, x): return self.w3(F.silu(self.w1(x)) * self.w2(x))


@dataclass
class HybridConfig:
    vocab: int = 256
    d_model: int = 256
    n_layers: int = 4
    attn_every: int = 4                          # every attn_every-th layer is sliding-window attention (3:1)
    window: int = 512
    n_heads: int = 4
    layer: Optional[LayerConfig] = None
    mlp_mult: int = 4
    checkpoint: bool = False                     # recompute each memory layer's forward in the backward pass (memory for compute; 16 GB GPUs)
    pos: str = "rope"                            # "rope": rotary positions in the attention layers; "learned": learned absolute positions added to the input
    max_len: int = 4096                          # learned absolute positions: table size
    tied: bool = True                            # output head = input embedding (True) or a separate linear head (False)
    emb_std: float = 0.02                        # input embedding init std (1.0 = PyTorch's default)
    # The defaults are the original configuration (kept for the frozen reference test). Measured on the textbook MQAR task
    # (16 pairs, T = 64, 2 attention layers): the defaults plateau at 22%; pos="learned", tied=False, emb_std=1.0 learns it
    # to 100% at step 2,000, like a standard reference transformer. Recall experiments use the latter.

    def __post_init__(self):
        if self.layer is None: self.layer = LayerConfig(d_model=self.d_model, n_heads=self.n_heads)
        if self.pos not in ("rope", "learned"): raise ValueError("pos must be rope or learned")


class HybridLM(nn.Module):
    """Decoder: [memory, memory, memory, SWA] x ..., pre-norm, SwiGLU MLP, tied embeddings."""
    def __init__(self, c: HybridConfig):
        super().__init__(); self.c = c; d = c.d_model; N = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.emb = nn.Embedding(c.vocab, d); nn.init.normal_(self.emb.weight, std=c.emb_std); self.mix = nn.ModuleList(); self.kind = []
        for i in range(c.n_layers):
            attn = (i + 1) % c.attn_every == 0
            self.mix.append(SlidingWindowAttention(d, c.n_heads, c.window, rope=(c.pos == "rope")) if attn else StateMemoryLayer(c.layer)); self.kind.append("attn" if attn else "mem")
        self.n1 = nn.ModuleList([N(d) for _ in range(c.n_layers)]); self.n2 = nn.ModuleList([N(d) for _ in range(c.n_layers)])
        self.mlp = nn.ModuleList([SwiGLU(d, c.mlp_mult) for _ in range(c.n_layers)]); self.nf = N(d)
        if c.pos == "learned": self.pos_emb = nn.Embedding(c.max_len, d)                      # created after the original modules: the default layout is unchanged
        if not c.tied: self.head = nn.Linear(d, c.vocab, bias=False)

    def forward(self, tokens, return_states=False):
        x = self.emb(tokens); states = []
        if self.c.pos == "learned": x = x + self.pos_emb(torch.arange(tokens.shape[1], device=tokens.device))[None]
        for i, m in enumerate(self.mix):
            if self.kind[i] == "attn": h = m(self.n1[i](x))
            elif self.c.checkpoint and torch.is_grad_enabled():
                h, st = torch.utils.checkpoint.checkpoint(m, self.n1[i](x), use_reentrant=False); states.append(st)
            else: h, st = m(self.n1[i](x)); states.append(st)
            x = x + h; x = x + self.mlp[i](self.n2[i](x))
        logits = self.head(self.nf(x)) if not self.c.tied else self.nf(x) @ self.emb.weight.t()
        return (logits, states) if return_states else logits

    def memory_layers(self): return [m for m in self.mix if isinstance(m, StateMemoryLayer)]


def count_params(m: nn.Module) -> int: return sum(p.numel() for p in m.parameters() if p.requires_grad)


def textbook_gdn(q, k, v, beta, gamma, alpha):
    """Reference Gated DeltaNet recurrence with decoupled erase (gamma) and write (beta) gates,
    read after write: S_t = alpha_t S_{t-1} (I - gamma_t k k^T) + beta_t k v^T ; y_t = S_t^T q_t."""
    B, H, T, dk = k.shape; S = k.new_zeros(B, H, dk, v.shape[-1]); ys = []
    for t in range(T):
        kt = k[:, :, t]; I_minus = torch.eye(dk, device=k.device, dtype=k.dtype)[None, None] - gamma[:, :, t, None, None] * kt[..., :, None] * kt[..., None, :]
        S = alpha[:, :, t, None, None] * torch.einsum("bhkv,bhkj->bhjv", S, I_minus) + beta[:, :, t, None, None] * kt[..., :, None] * v[:, :, t, None, :]
        ys.append(torch.einsum("bhkv,bhk->bhv", S, q[:, :, t]))
    return torch.stack(ys, 2), S


def layer_cost(c: LayerConfig) -> dict:
    """Per head per token: MACs of the memory recurrence and reads, state values, and the layer's
    parameters (the design's cost table, docs/phase3/DESIGN.md 3.5). Gate MACs count the MLP."""
    H, dv, dc, R = c.n_heads, c.d_v, c.d_c, c.n_traces
    def macs_at(dk, kernel=False):                                       # dk: the rows a read or write touches (all rows as implemented here; the active rows for a sparse kernel)
        m = 3 * dk * dv * R                                              # per trace: read S^T q, predict S^T k, rank-1 write (GDN: ~49k at 128x128)
        if R == 2:
            m += (3 + c.n_state_slots) * c.readout_hidden + c.readout_hidden + 4 * dv          # the learned readout and its cosine / mixture arithmetic
            if c.tracks_conf: m += dk * dc + dc * dv + dk                                       # the state reads at the query address for the readout
            if c.tracks_a: m += dk
        if c.tracks_conf: m += 2 * dk * dc + 2 * dc * dv + 2 * dk                               # C read/write, P vhat and P v, Zc read/write
        if c.tracks_a: m += 2 * dk
        d_in = c.d_g + dv + R * dv + 2 * R + 1 + c.n_state_slots; m += d_in * c.gate_hidden + c.gate_hidden * 3 * R
        if c.address == "product": m += 2 * ((c.d_k // 2) * (c.pk_n1 + c.n_cells // c.pk_n1) + c.n_cells)   # sub-key scores and the score grid, query and key
        elif c.address in ("sparse", "expand_dense") and kernel: m += 2 * c.fan_in * c.n_cells   # the fixed expansion as a gather (fan_in inputs per cell), query and key
        elif c.address != "dense": m += 2 * c.d_k * c.n_cells          # the expansion of the query and the key as a dense product (as implemented; always for a learned expansion)
        return m                                                         # k-WTA selection (a top-k over n_cells per token) is comparisons, not MACs, and is not counted
    dk = c.rows; state = dk * dv * R
    if c.tracks_conf: state += dk * dc + dk
    if c.tracks_a: state += dk + 1
    params = sum(p.numel() for p in StateMemoryLayer(c).parameters())
    return {"macs_per_head_token": macs_at(dk), "macs_active_per_head_token": macs_at(c.active_rows, kernel=True), "active_rows": c.active_rows,
            "state_per_head": state, "params": params, "state_per_layer": H * state}


@torch.no_grad()
def chunk_error(layer: StateMemoryLayer, x: torch.Tensor) -> float:
    """PREREG correctness gate (ii): relative error of the layer's chunked output against its sequential reference on a batch."""
    c = layer.c; seq = StateMemoryLayer(LayerConfig(**{**c.__dict__, "chunk": 0})).to(x.device); seq.load_state_dict(layer.state_dict())
    yc, _ = layer(x); ys, _ = seq(x)
    return float((yc - ys).norm() / (ys.norm() + 1e-12))
