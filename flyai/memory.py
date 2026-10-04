"""Mushroom-body episodic memory, v0.2.

* ONE PN projection, ONE fixed sparse expansion and ONE k-WTA give the KC address shared by
  all H compartments (as in the fly); compartments differ in their fast-weight matrix M_h,
  target content u_h, DAN gate beta_h and a log-spaced decay prior tau_h in {4,16,64,256} (alpha_h = 1 - 1/tau_h).
* Hetero-associative, delayed key: at step t the write pairs the PREVIOUS key code with the
  CURRENT content (key = code(s_{t-1}), value = u(s_t)); the read at step t uses code(s_{t-1})
  and therefore predicts the content of s_t before it is inferred.
* Gate ladder (`gate`):
    'precision' : variance-normalised surprise  beta = sigmoid(a * log(pi_slow * ||eps||^2 / n) + b [+ log gamma])
                  -- writes rise when the current error is large relative to the expected error
                  (the DAN-as-precision-weighted-prediction-error hypothesis)
    'surprise'  : raw surprise  beta = sigmoid(a * log(||eps||^2 / n) + b [+ log gamma])  (Titans-style control)
    'learned'   : beta = sigmoid(w . s + c [+ log gamma])   (Gated-DeltaNet-style control)
    'const'     : beta = 0.5 exactly (arousal does not touch it: it is the constant control)
  pi_slow is the bounded Gamma-posterior precision of the compartment's own memory error
  (flyai.primitives.GammaPrecision, rho = `rho`).
* A novelty MBON (Bloom-filter familiarity of the address) is read every step; exposed in the
  state as `nov` for arousal / epistemic use.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import GammaPrecision, NoveltyMBON

DECAY_PRIORS = (4.0, 16.0, 64.0, 256.0)
GATES = ("precision", "surprise", "learned", "const")


class MBMemory(nn.Module):
    def __init__(self, d: int, n_pn: int = 64, n_kc: int = 1024, fan_in: int = 7, active_frac: float = 0.05,
                 d_v: int = 32, n_heads: int = 2, gate: str = "precision", rho: float = 0.98, seed: int = 0,
                 shared_code: bool = True, E: Optional[torch.Tensor] = None):
        super().__init__()
        if gate not in GATES:
            raise ValueError(f"memory gate must be one of {GATES}, got {gate!r}")
        self.H, self.dv, self.gate, self.n_kc, self.n_pn, self.shared = n_heads, d_v, gate, n_kc, n_pn, shared_code
        g = torch.Generator().manual_seed(seed)
        n_codes = 1 if shared_code else n_heads
        if E is None:
            E = torch.zeros(n_codes, n_kc, n_pn)
            for c in range(n_codes):
                for i in range(n_kc):
                    E[c, i, torch.randperm(n_pn, generator=g)[:fan_in]] = 1.0
        else:
            E = E[None].expand(n_codes, -1, -1).clone()
        self.register_buffer("E", E / E.sum(-1, keepdim=True).clamp_min(1e-6))
        self.k = max(1, int(round(active_frac * n_kc)))
        self.theta = nn.Parameter(torch.zeros(n_codes, 1))
        self.pn = nn.Linear(d, n_codes * n_pn)
        self.u_proj = nn.Linear(d, n_heads * d_v); self.o_proj = nn.Linear(n_heads * d_v, d)
        self.alpha_proj = nn.Linear(d, n_heads)
        with torch.no_grad():                                                   # log-spaced decay priors per compartment
            taus = torch.tensor([DECAY_PRIORS[h % len(DECAY_PRIORS)] for h in range(n_heads)])
            a0 = 1 - 1.0 / taus; self.alpha_proj.bias.copy_(torch.log(a0 / (1 - a0)))      # alpha = 1 - 1/tau
        if gate in ("precision", "surprise"):
            self.ga = nn.Parameter(torch.full((n_heads,), 0.5)); self.gb = nn.Parameter(torch.zeros(n_heads))
        elif gate == "learned":
            self.beta_proj = nn.Linear(d, n_heads)
        self.prec = GammaPrecision(rho=rho)
        self.novelty = NoveltyMBON()
        self.norm = nn.RMSNorm(d_v)

    # ---- address --------------------------------------------------------------------
    def code(self, x: torch.Tensor):                          # -> values (B,H,k), indices (B,H,k)
        """PN projection -> fixed sparse expansion -> k-WTA (the only k-WTA in the model), L2-normalised."""
        B = x.shape[0]; n_codes = self.E.shape[0]
        p = self.pn(x).view(B, n_codes, self.n_pn)
        z = torch.einsum("bcp,ckp->bck", p, self.E) - self.theta
        top, idx = z.topk(self.k, dim=-1)
        v = F.normalize(F.relu(top), dim=-1)
        if n_codes == 1:
            v, idx = v.expand(-1, self.H, -1), idx.expand(-1, self.H, -1)
        return v, idx

    def _rows(self, M, idx):
        return M.gather(2, idx[..., None].expand(-1, -1, -1, self.dv))

    def init_state(self, B: int, like: torch.Tensor):
        return {"M": like.new_zeros(B, self.H, self.n_kc, self.dv), "prec": self.prec.init(like.new_zeros(B, self.H)),
                "beta": like.new_zeros(B, self.H), "nov_w": self.novelty.init(B, self.n_kc, like), "nov": like.new_ones(B)}

    # ---- read: prediction for the content of the CURRENT step from the previous key ------
    def read(self, st, x_prev: torch.Tensor) -> torch.Tensor:
        kv, ki = self.code(x_prev)
        o = (kv[..., None] * self._rows(st["M"], ki)).sum(2)
        return self.o_proj(self.norm(o).reshape(x_prev.shape[0], -1))

    # ---- shared write maths ------------------------------------------------------------
    def _terms(self, st, M_rows_src: torch.Tensor, x_prev: torch.Tensor, x_now: torch.Tensor, gamma: Optional[torch.Tensor]):
        """Key code, prediction error against `M_rows_src`, gate, decay, rank-1 update and the
        updated bookkeeping (precision, novelty)."""
        kv, ki = self.code(x_prev); B = x_now.shape[0]
        u = self.u_proj(x_now).view(B, self.H, self.dv)
        eps = (kv[..., None] * self._rows(M_rows_src, ki)).sum(2) - u          # (B,H,dv): what the memory predicted minus what arrived
        e2 = eps.pow(2).mean(-1)                                             # per-head mean squared error (||eps||^2 / n)
        lg = torch.log(gamma)[:, None] if gamma is not None else 0.0         # arousal: gain on the gate logit
        if self.gate == "precision":
            beta = torch.sigmoid(self.ga * torch.log(GammaPrecision.ratio(st["prec"]) * e2 + 1e-6) + self.gb + lg)
        elif self.gate == "surprise":
            beta = torch.sigmoid(self.ga * torch.log(e2 + 1e-6) + self.gb + lg)
        elif self.gate == "learned":
            beta = torch.sigmoid(self.beta_proj(x_now) + lg)
        else:                                                                # const: exactly 0.5, never modulated
            beta = x_now.new_full((B, self.H), 0.5)
        prec, _ = self.prec.update(st["prec"], eps)
        alpha = torch.sigmoid(self.alpha_proj(x_now))
        upd = -(beta[..., None, None]) * kv[..., None] * eps[:, :, None, :]
        nov = self.novelty.read(st["nov_w"], ki[:, 0]); nov_w = self.novelty.update(st["nov_w"], ki[:, 0])
        return kv, ki, upd, alpha, {"prec": prec, "beta": beta, "nov_w": nov_w, "nov": nov}

    # ---- chunk-delayed writes (LaCT-style): propose against a frozen M, apply once per chunk ---
    def propose(self, st, M_ref: torch.Tensor, x_prev: torch.Tensor, x_now: torch.Tensor, gamma: Optional[torch.Tensor] = None):
        """Same maths as `write` but the prediction error, gate and precision use `M_ref`
        (the chunk-start memory). Returns the updated bookkeeping state and the rank-1
        update ingredients (ki, upd, alpha) without touching M."""
        kv, ki, upd, alpha, book = self._terms(st, M_ref, x_prev, x_now, gamma)
        return {"M": st["M"], **book}, (ki, upd, alpha)

    def apply_proposals(self, M: torch.Tensor, proposals) -> torch.Tensor:   # was `apply`: that name shadows nn.Module.apply, which transformers calls when it initialises weights
        """Apply a chunk of proposals [(ki, upd, alpha), ...] in order with cumulative decay:
        M <- (prod alpha) M + sum_t (prod_{s>t} alpha_s) upd_t. Equals sequential writes when
        the proposals were computed against the sequential M (exact for chunk size 1)."""
        if not proposals: return M
        alphas = torch.stack([a for _, _, a in proposals], 0)                       # (C,B,H)
        logc = torch.log(alphas.clamp_min(1e-6)).flip(0).cumsum(0).flip(0)         # log prod_{s>=t} alpha_s
        total = torch.exp(logc[0])
        out = M * total[..., None, None]
        for t, (ki, upd, _) in enumerate(proposals):
            w = torch.exp(logc[t] - torch.log(alphas[t].clamp_min(1e-6)))         # prod_{s>t} alpha_s
            out = out.scatter_add(2, ki[..., None].expand(-1, -1, -1, self.dv), upd * w[..., None, None])
        return out

    # ---- write: pair the previous key with the current content ------------------------
    def write(self, st, x_prev: torch.Tensor, x_now: torch.Tensor, gamma: Optional[torch.Tensor] = None):
        kv, ki, upd, alpha, book = self._terms(st, st["M"], x_prev, x_now, gamma)
        M = (st["M"] * alpha[..., None, None]).scatter_add(2, ki[..., None].expand(-1, -1, -1, self.dv), upd)
        return {"M": M, **book}
