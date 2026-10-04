"""FlyAIBlock (v0.2): the predictive-coding filter as a residual block for deep decoders.

Inside a stack the block's observation at time t is its input x_t, so with a Gaussian
level-0 likelihood the free energy is exactly quadratic in the latent and its minimiser is
the precision-weighted fusion of the evidence and the top-down priors (closed form, no
inner loop; the design panel's central finding):

    s*_t = w0 x_t + w1 mu1_t + w3 mu3_t,   w = softmax over levels of (log pi_l + c_l), per group

Precisions (bounded ratios, flyai.primitives.GammaPrecision): pi0 = Gamma posterior of the
level-0 residual (x - s*) x arousal; pi1 = Gamma posterior of (s* - mu1); pi3 = the attractor
bank's amplitude (or a Gamma posterior as the control). Arousal gamma_t (bounded (0.25, 4),
from running Mahalanobis surprise) scales the sensory precision and the memory write gate;
it never multiplies the free energy.

Fusion arms: 'closed' (hypothesis), 'learned' (a softmax gate over [x, mu1, mu3] computed per
precision group from that group's own channels and shared across groups: n_lv^2 d/G + n_lv
parameters, i.e. within 0.1 % of the closed form), 'loop' (the v0 K-step gradient loop with an
amortised init; the EBT-class ablation). Chunked activation checkpointing (`chunk` tokens)
keeps GPU memory at O(T / chunk) fast-weight states instead of O(T). The core gain a chunk
was computed with is passed into the checkpointed function as a tensor, so a recompute in
the backward pass sees the same gain even though the homeostatic controller (`regulate`,
once per chunk at the boundary, outside the checkpointed region) has moved it since."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .primitives import DaleCore, GammaPrecision
from .memory import MBMemory, GATES
from .attractor import AttractorBank

FUSIONS = ("closed", "learned", "loop")
AROUSALS = ("both", "pi0", "beta", "none")
PI3_SOURCES = ("amplitude", "gamma")


@dataclass
class BlockConfig:
    d: int = 256
    fusion: str = "closed"            # closed | learned | loop
    K_max: int = 2                    # only used by fusion='loop'
    use_memory: bool = True
    use_attractor: bool = True
    memory_gate: str = "precision"    # precision | surprise | learned | const
    shared_code: bool = True
    arousal: str = "both"             # both | pi0 | beta | none
    pi3_source: str = "amplitude"     # amplitude | gamma
    n_groups: int = 8
    regulate_core: bool = True          # homeostatic gain regulation once per chunk (see `regulate`)
    target_lambda: float = -0.02
    inh_frac: float = 0.308
    core_fan_in: int = 32
    n_kc: int = 2048
    n_pn: int = 64
    fan_in: int = 8
    active_frac: float = 0.05
    d_v: int = 32
    n_heads: int = 8
    n_rings: int = 8
    n_ring_units: int = 8
    rho_slow: float = 0.98
    chunk: int = 64
    delayed_write: bool = False       # LaCT-style: memory reads/gates use the chunk-start M; updates applied once per chunk
    seed: int = 0

    def __post_init__(self):
        for name, val, allowed in (("fusion", self.fusion, FUSIONS), ("arousal", self.arousal, AROUSALS),
                                   ("memory_gate", self.memory_gate, GATES), ("pi3_source", self.pi3_source, PI3_SOURCES)):
            if not isinstance(val, str) or val not in allowed:
                raise ValueError(f"BlockConfig.{name} must be one of {allowed}, got {val!r}")
        if self.d % self.n_groups:
            raise ValueError(f"d={self.d} must be divisible by n_groups={self.n_groups}")
        if self.chunk < 1 or self.K_max < 0:
            raise ValueError("chunk must be >= 1 and K_max >= 0")


class FlyAIBlock(nn.Module):
    def __init__(self, c: BlockConfig):
        super().__init__()
        self.c = c; d, G = c.d, c.n_groups
        self.G = G
        self.core = DaleCore(d, d_in=d, inh_frac=c.inh_frac, fan_in=c.core_fan_in, seed=c.seed, target_lambda=c.target_lambda)
        self.td = nn.Linear(2 * d, d)
        self.memory = MBMemory(d, n_pn=c.n_pn, n_kc=c.n_kc, fan_in=c.fan_in, active_frac=c.active_frac, d_v=c.d_v, n_heads=c.n_heads,
                               gate=c.memory_gate, rho=c.rho_slow, seed=c.seed, shared_code=c.shared_code) if c.use_memory else None
        self.attractor = AttractorBank(d, n_rings=c.n_rings, n_units=c.n_ring_units) if c.use_attractor else None
        self.g3 = nn.Linear(self.attractor.out_dim, d) if c.use_attractor else None
        n_lv = 3 if c.use_attractor else 2
        self.n_lv = n_lv
        if c.fusion == "learned":
            self.gate = nn.Linear(n_lv * (d // G), n_lv)        # per-group learned gate, shared across groups
        else:
            self.c_level = nn.Parameter(torch.zeros(n_lv, G))   # fusion offsets (only the closed / loop arms use them)
        if c.fusion == "loop":
            self.amort = nn.Linear(2 * d, d); self.g0 = nn.Linear(d, d); self.log_eta = nn.Parameter(torch.full((1,), -1.0))
        if c.arousal != "none":
            self.ar_w = nn.Parameter(torch.zeros(1)); self.ar_b = nn.Parameter(torch.zeros(1))
        self.prec = GammaPrecision(rho=c.rho_slow)
        self.out_norm = nn.RMSNorm(d)

    # ---------------------------------------------------------------------------------
    def init_state(self, B, like):
        d = self.c.d
        st = {"s1": like.new_zeros(B, d), "pi0": self.prec.init(like.new_zeros(B, self.G)), "pi1": self.prec.init(like.new_zeros(B, self.G)),
              "pi3": self.prec.init(like.new_zeros(B, self.G)), "Sbar": like.new_ones(B), "t": like.new_zeros(())}
        if self.memory is not None: st["mem"] = self.memory.init_state(B, like)
        if self.attractor is not None: st["s3"] = self.attractor.init_state(B, like)
        return st

    def state_numel(self, B: int = 1) -> int:
        """Recurrent state size (elements per batch row) that a token step carries."""
        st = self.init_state(B, torch.zeros(1))
        tot = 0
        def rec(v):
            nonlocal tot
            if torch.is_tensor(v): tot += v.numel()
            elif isinstance(v, dict): [rec(u) for u in v.values()]
            elif isinstance(v, (tuple, list)): [rec(u) for u in v]
        rec(st); return tot // B

    def _grouped(self, v):
        return v.repeat_interleave(self.c.d // self.G, dim=-1)

    def _arousal(self, st):
        if self.c.arousal == "none":
            return st["Sbar"].new_ones(st["Sbar"].shape[0])
        z = self.ar_w * torch.log(st["Sbar"] + 1e-3) + self.ar_b
        return torch.exp(z.clamp(float(np.log(0.25)), float(np.log(4.0))))

    def _priors(self, st, M_ref=None, gain=None):
        c = self.c; s1p = st["s1"]; B, d = s1p.shape
        if self.memory is not None:
            mem_st = st["mem"] if M_ref is None else {**st["mem"], "M": M_ref}
            mu2 = self.memory.read(mem_st, s1p)
        else:
            mu2 = s1p.new_zeros(B, d)
        if self.attractor is not None:
            s3 = self.attractor.step(st["s3"], s1p); mu3 = self.g3(self.attractor.readout(s3))
            pi3 = self.attractor.precision(s3).mean(-1, keepdim=True).expand(-1, self.G) if c.pi3_source == "amplitude" else self.prec.ratio(st["pi3"])
        else:
            s3, mu3, pi3 = None, None, None
        td = self.td(torch.cat([mu2, mu3 if mu3 is not None else torch.zeros_like(mu2)], -1))
        mu1 = self.core.step(s1p, td, gain=gain)
        return mu1, mu3, pi3, s3, td

    def _fuse(self, x, mu1, mu3, pi0, pi1, pi3):
        levels = [x, mu1] + ([mu3] if mu3 is not None else [])
        if self.c.fusion == "learned":
            B = x.shape[0]; dg = self.c.d // self.G
            z = torch.cat([v.view(B, self.G, dg) for v in levels], -1)                    # (B, G, n_lv * dg)
            logw = self.gate(z).permute(2, 0, 1)                                          # (L, B, G)
        else:
            pis = [pi0, pi1] + ([pi3] if pi3 is not None else [])
            logw = torch.stack([torch.log(p + 1e-6) + self.c_level[l] for l, p in enumerate(pis)], 0)   # (L, B, G)
        w = torch.softmax(logw, 0)
        return sum(self._grouped(w[l]) * v for l, v in enumerate(levels))

    def _loop(self, x, mu1, mu3, pi0, pi1, pi3):
        """v0 ablation arm: K gradient steps on the quadratic free energy from an amortised init."""
        s1 = mu1 + self.amort(torch.cat([x, mu1], -1)); P0, P1 = self._grouped(pi0), self._grouped(pi1)
        P3 = self._grouped(pi3) if pi3 is not None else None
        step = torch.exp(self.log_eta) / ((pi0 + pi1 + (pi3 if pi3 is not None else 0)).mean(-1) + 1.0)
        create_graph = self.training and torch.is_grad_enabled()
        with torch.enable_grad():
            for _ in range(self.c.K_max):
                if not s1.requires_grad: s1 = s1.detach().requires_grad_(True)
                Fc = 0.5 * (P0 * (x - self.g0(s1)).pow(2)).sum(-1) + 0.5 * (P1 * (s1 - mu1).pow(2)).sum(-1)
                if mu3 is not None: Fc = Fc + 0.5 * (P3 * (s1 - mu3).pow(2)).sum(-1)
                g, = torch.autograd.grad(Fc.sum(), s1, create_graph=create_graph)
                s1 = s1 - step[:, None] * g
        return s1 if create_graph else s1.detach()

    def step(self, st, x, M_ref=None, gain=None):
        """One token. Returns (output, new_state, proposal); `proposal` is the delayed-write
        ingredient tuple when `M_ref` is given (LaCT-style chunk), else None."""
        c = self.c; B = x.shape[0]
        mu1, mu3, pi3, s3, td = self._priors(st, M_ref, gain)
        gamma = self._arousal(st)
        pi0 = self.prec.ratio(st["pi0"]) * (gamma[:, None] if c.arousal in ("both", "pi0") else 1.0)
        pi1 = self.prec.ratio(st["pi1"])
        s1 = self._loop(x, mu1, mu3, pi0, pi1, pi3) if c.fusion == "loop" else self._fuse(x, mu1, mu3, pi0, pi1, pi3)
        with torch.no_grad():
            x_hat = self.g0(s1) if c.fusion == "loop" else s1                          # the level-0 prediction the likelihood used
            e0 = (x - x_hat).detach().view(B, self.G, -1); e1 = (s1 - mu1).detach().view(B, self.G, -1)
            new_pi0 = self.prec.update(st["pi0"], e0)[0]; new_pi1 = self.prec.update(st["pi1"], e1)[0]
            new_pi3 = (self.prec.update(st["pi3"], (s1 - mu3).detach().view(B, self.G, -1))[0]
                       if (mu3 is not None and c.pi3_source == "gamma") else st["pi3"])
            surprise = (self._grouped(self.prec.ratio(st["pi0"])) * (x - x_hat).detach().pow(2)).mean(-1)   # Mahalanobis, scale-free
            Sbar = 0.9 * st["Sbar"] + 0.1 * surprise
        new = {"s1": s1, "pi0": new_pi0, "pi1": new_pi1, "pi3": new_pi3, "Sbar": Sbar, "t": st["t"] + 1}
        prop = None
        if self.memory is not None:
            g_ = gamma if c.arousal in ("both", "beta") else None
            if M_ref is None:
                new["mem"] = self.memory.write(st["mem"], st["s1"], s1, gamma=g_)
            else:
                new["mem"], prop = self.memory.propose(st["mem"], M_ref, st["s1"], s1, gamma=g_)
        if self.attractor is not None: new["s3"] = s3
        return self.out_norm(s1), new, prop

    @torch.no_grad()
    def regulate(self, st):
        """Lyapunov estimate (every condition) + homeostatic gain step (only when regulating) at a
        chunk boundary: outside the checkpointed region, so it runs once per chunk and never
        during recompute."""
        _, _, _, _, td = self._priors(st)
        self.core.lyapunov_step(st["s1"], td)
        if self.c.regulate_core: self.core.homeostatic_step()

    # ---- chunked forward with activation checkpointing ---------------------------------
    def _run_chunk(self, x_chunk, spec, gain, *flat):
        st = _unflatten(flat, spec); ys = []
        if self.c.delayed_write and self.memory is not None:
            M_ref, proposals = st["mem"]["M"], []
            for t in range(x_chunk.shape[1]):
                y, st, prop = self.step(st, x_chunk[:, t], M_ref=M_ref, gain=gain); ys.append(y); proposals.append(prop)
            st["mem"] = {**st["mem"], "M": self.memory.apply_proposals(M_ref, proposals)}
        else:
            for t in range(x_chunk.shape[1]):
                y, st, _ = self.step(st, x_chunk[:, t], gain=gain); ys.append(y)
        flat_new, spec_new = _flatten(st)
        assert spec_new == spec, "state layout must be fixed across tokens"
        return (torch.stack(ys, 1), *flat_new)

    def forward(self, x, state=None):
        B, T, _ = x.shape
        st = state if state is not None else self.init_state(B, x)
        flat, spec = _flatten(st); ys = []
        use_ckpt = self.training and torch.is_grad_enabled() and T > self.c.chunk
        for s0 in range(0, T, self.c.chunk):
            xc = x[:, s0:s0 + self.c.chunk]
            gain = self.core.gain.detach().clone()                # the gain this chunk is computed AND recomputed with
            out = checkpoint(self._run_chunk, xc, spec, gain, *flat, use_reentrant=False) if use_ckpt else self._run_chunk(xc, spec, gain, *flat)
            ys.append(out[0]); flat = list(out[1:])
            if self.training:
                self.regulate(_unflatten(flat, spec))
        return torch.cat(ys, 1), _unflatten(flat, spec)


def _flatten(st):
    """State dict (nested dicts / tuples / tensors / None) -> flat tensor list + spec."""
    flat, spec = [], []
    def rec(v, path):
        if torch.is_tensor(v): spec.append((path, "t")); flat.append(v)
        elif v is None: spec.append((path, "none"))
        elif isinstance(v, dict):
            for k in v: rec(v[k], path + (k,))
        elif isinstance(v, (tuple, list)):
            for i, u in enumerate(v): rec(u, path + (i,))
        else: raise TypeError(type(v))
    rec(st, ()); return flat, spec


def _unflatten(flat, spec):
    out = {}; it = iter(flat)
    for path, kind in spec:
        v = next(it) if kind == "t" else None
        node = out
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = v
    def fix(node):                                    # dicts with int keys 0..k -> tuples (Gamma states)
        if isinstance(node, dict):
            for k in list(node): node[k] = fix(node[k])
            if node and all(isinstance(k, int) for k in node): return tuple(node[i] for i in range(len(node)))
        return node
    return fix(out)
