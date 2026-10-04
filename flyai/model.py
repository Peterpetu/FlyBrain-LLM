"""FlyAI (v0.2): a predictive-coding sequence filter with connectome-derived numerical priors.

Per token:  predict (top-down prior = precision-weighted fusion of the core's and the
attractor bank's predictions) -> observe -> infer (K free-energy steps on the CATEGORICAL
likelihood; the only non-quadratic term) -> update precisions and arousal -> write the
episodic memory (delayed key) -> regulate the core.

Levels: 0 = token (categorical NLL, precision = learned temperature x arousal),
1 = core prior (Gamma-tracked precision, grouped), 3 = attractor prior (precision from the
rings' amplitude, or a Gamma tracker as the control). The old level-2 tracker measured a
target, not an error, and was removed. The language-model prediction is made BEFORE the
token is observed; with K = 0 the model is an amortised recurrent network (the ablation)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import DaleCore, GammaPrecision
from .memory import MBMemory, GATES
from .attractor import AttractorBank


@dataclass
class FlyAIConfig:
    vocab: int = 65
    d: int = 64
    n: int = 128
    K_max: int = 3
    tol: float = 0.02
    adaptive_K: bool = True
    use_memory: bool = True
    use_attractor: bool = True
    memory_gate: str = "precision"      # precision | surprise | learned | const
    shared_code: bool = True
    arousal: str = "both"               # both | pi0 | beta | none | learned
    pi3_source: str = "amplitude"       # amplitude | gamma
    n_groups: int = 2                   # precision groups over the core width (1 = scalar per level)
    regulate_core: bool = True
    regulate_every: int = 8
    target_lambda: float = -0.02
    inh_frac: float = 0.308
    core_fan_in: int = 32
    n_kc: int = 512
    n_pn: int = 61
    fan_in: int = 7
    active_frac: float = 0.05
    d_v: int = 32
    n_heads: int = 2
    n_rings: int = 4
    n_ring_units: int = 8
    rho_slow: float = 0.98
    seed: int = 0

    def __post_init__(self):
        for name, val, allowed in (("memory_gate", self.memory_gate, GATES), ("arousal", self.arousal, ("both", "pi0", "beta", "none", "learned")),
                                   ("pi3_source", self.pi3_source, ("amplitude", "gamma"))):
            if not isinstance(val, str) or val not in allowed:
                raise ValueError(f"FlyAIConfig.{name} must be one of {allowed}, got {val!r}")
        if self.n % self.n_groups or self.K_max < 0:
            raise ValueError("n must be divisible by n_groups and K_max >= 0")


class FlyAI(nn.Module):
    def __init__(self, cfg: FlyAIConfig):
        super().__init__()
        self.cfg = cfg; d, n, G = cfg.d, cfg.n, cfg.n_groups
        assert n % G == 0; self.G = G
        self.emb = nn.Embedding(cfg.vocab, d)
        self.core = DaleCore(n, d_in=n, inh_frac=cfg.inh_frac, fan_in=cfg.core_fan_in, seed=cfg.seed, target_lambda=cfg.target_lambda)
        self.td = nn.Linear(2 * n, n)
        self.out_norm = nn.RMSNorm(n); self.out = nn.Linear(n, cfg.vocab)
        self.amort = nn.Linear(d + n, n)
        self.memory = MBMemory(n, n_pn=cfg.n_pn, n_kc=cfg.n_kc, fan_in=cfg.fan_in, active_frac=cfg.active_frac, d_v=cfg.d_v, n_heads=cfg.n_heads,
                               gate=cfg.memory_gate, rho=cfg.rho_slow, seed=cfg.seed, shared_code=cfg.shared_code) if cfg.use_memory else None
        self.attractor = AttractorBank(n, n_rings=cfg.n_rings, n_units=cfg.n_ring_units) if cfg.use_attractor else None
        self.g3 = nn.Linear(self.attractor.out_dim, n) if cfg.use_attractor else None
        self.log_eta = nn.Parameter(torch.full((1,), -1.0))
        self.log_temp = nn.Parameter(torch.zeros(1))                 # pi_0 = exp(log_temp): learned temperature
        self.c_level = nn.Parameter(torch.zeros(2))                  # ratio offsets for levels 1 and 3
        if cfg.arousal == "learned":
            self.ar_proj = nn.Linear(n, 1)
        elif cfg.arousal != "none":
            self.ar_w = nn.Parameter(torch.zeros(1)); self.ar_b = nn.Parameter(torch.zeros(1))
        self.prec = GammaPrecision(rho=cfg.rho_slow)

    # ---------------------------------------------------------------------------------
    def init_state(self, B: int, like: torch.Tensor):
        n = self.cfg.n
        st = {"s1": like.new_zeros(B, n), "pi1": self.prec.init(like.new_zeros(B, self.G)), "pi3": self.prec.init(like.new_zeros(B, self.G)),
              "Sbar": like.new_full((B,), float(np.log(self.cfg.vocab))), "t": 0}
        if self.memory is not None: st["mem"] = self.memory.init_state(B, like)
        if self.attractor is not None: st["s3"] = self.attractor.init_state(B, like)
        return st

    def _grouped(self, v):                       # (B, G) -> (B, n)
        return v.repeat_interleave(self.cfg.n // self.G, dim=-1)

    def _arousal(self, st):
        c = self.cfg; B = st["s1"].shape[0]
        if c.arousal == "none":
            return st["s1"].new_ones(B)
        if c.arousal == "learned":
            z = self.ar_proj(st["s1"]).squeeze(-1)
        else:
            z = self.ar_w * (torch.log(st["Sbar"] + 1e-3) - float(np.log(np.log(c.vocab)))) + self.ar_b     # scale-free running surprise
        return torch.exp(z.clamp(float(np.log(0.25)), float(np.log(4.0))))

    def _priors(self, st):
        """Prior mean for the fast latent (fusion of core and attractor predictions) and the LM prediction."""
        c = self.cfg; s1p = st["s1"]; B = s1p.shape[0]
        mu2 = self.memory.read(st["mem"], s1p) if self.memory is not None else s1p.new_zeros(B, c.n)
        if self.attractor is not None:
            s3 = self.attractor.step(st["s3"], s1p); mu3 = self.g3(self.attractor.readout(s3))
            pi3 = (self.attractor.precision(s3).mean(-1, keepdim=True).expand(-1, self.G) if c.pi3_source == "amplitude" else self.prec.ratio(st["pi3"]))
        else:
            s3, mu3, pi3 = None, None, None
        self._td_input = self.td(torch.cat([mu2, mu3 if mu3 is not None else torch.zeros_like(mu2)], -1))
        mu1 = self.core.step(s1p, self._td_input)
        pi1 = self.prec.ratio(st["pi1"])
        if mu3 is not None:
            logw = torch.stack([torch.log(pi1 + 1e-6) + self.c_level[0], torch.log(pi3 + 1e-6) + self.c_level[1]], 0)   # (2, B, G)
            w = torch.softmax(logw, 0)
            mu = self._grouped(w[0]) * mu1 + self._grouped(w[1]) * mu3
            pi_prior = pi1 + pi3
        else:
            mu, pi_prior = mu1, pi1
        return mu, mu1, mu3, s3, pi_prior

    def _F(self, s1, mu, o, pi0, pi_prior):
        nll = F.cross_entropy(self.out(self.out_norm(s1)), o, reduction="none")
        return pi0 * nll + 0.5 * (self._grouped(pi_prior) * (s1 - mu).pow(2)).sum(-1)

    def step(self, st, o_t: torch.Tensor):
        c = self.cfg; B = o_t.shape[0]
        mu, mu1, mu3, s3, pi_prior = self._priors(st)
        logits = self.out(self.out_norm(mu))                                     # the LM prediction (before observing)
        gamma = self._arousal(st)
        pi0 = torch.exp(self.log_temp) * (gamma if c.arousal in ("both", "pi0", "learned") else 1.0)
        e = self.emb(o_t)
        s1 = mu + self.amort(torch.cat([e, mu], -1))
        K_used = st["s1"].new_zeros(B)
        if c.K_max > 0:
            with torch.no_grad():
                c_nll = 0.5 * torch.linalg.matrix_norm(self.out.weight, 2) ** 2        # curvature bound of the NLL term
            step = (torch.exp(self.log_eta) / (pi0 * c_nll + pi_prior.mean(-1) + 1.0))[:, None]
            create_graph = self.training and torch.is_grad_enabled()          # gradients of the loss flow through inference when training
            with torch.enable_grad():
                active = torch.ones(B, dtype=torch.bool, device=s1.device)
                for k in range(c.K_max):
                    if not s1.requires_grad: s1 = s1.detach().requires_grad_(True)
                    Fc = self._F(s1, mu, o_t, pi0, pi_prior)
                    grad, = torch.autograd.grad(Fc.sum(), s1, create_graph=create_graph)
                    s1_new = s1 - step * grad
                    if c.adaptive_K:
                        Fn = self._F(s1_new, mu, o_t, pi0, pi_prior)
                        active = active & ((Fc - Fn) > c.tol * Fc.abs()).detach()
                        s1 = torch.where(active[:, None], s1_new, s1); K_used = K_used + active.float()
                        if not active.any(): break
                    else:
                        s1 = s1_new; K_used = K_used + 1
            if not create_graph: s1 = s1.detach()
        with torch.no_grad():
            nll = F.cross_entropy(logits.detach(), o_t, reduction="none")
            Sbar = 0.9 * st["Sbar"] + 0.1 * nll
            pi1 = self.prec.update(st["pi1"], (s1 - mu1).detach().view(B, self.G, -1))[0]                 # grouped Gamma posteriors
            pi3 = (self.prec.update(st["pi3"], (s1 - mu3).detach().view(B, self.G, -1))[0]
                   if (mu3 is not None and c.pi3_source == "gamma") else st["pi3"])           # the tracker is the control arm; dead otherwise
            Ff = self._F(s1.detach(), mu.detach(), o_t, pi0.detach() if torch.is_tensor(pi0) else pi0, pi_prior.detach()).mean()
        new = {"s1": s1, "pi1": pi1, "pi3": pi3, "Sbar": Sbar, "t": st["t"] + 1}
        if self.memory is not None:
            new["mem"] = self.memory.write(st["mem"], st["s1"], s1, gamma=(gamma if c.arousal in ("both", "beta", "learned") else None))
        if self.attractor is not None: new["s3"] = s3
        if self.training and st["t"] % c.regulate_every == 0:                  # lambda is ESTIMATED in every condition (reported per run);
            self.core.lyapunov_step(st["s1"], self._td_input.detach())          # the gain is only ACTED on when regulation is on
            if c.regulate_core: self.core.homeostatic_step()
        diag = {"F": Ff, "K": K_used, "gamma": gamma.detach(), "beta": (new["mem"]["beta"].mean() if self.memory is not None else torch.tensor(0.0)),
                "nov": (new["mem"]["nov"].mean() if self.memory is not None else torch.tensor(1.0)), "pi1_max": self.prec.ratio(pi1).max()}
        return logits, new, diag

    def forward(self, tokens: torch.Tensor, state=None):
        B, T = tokens.shape
        st = state if state is not None else self.init_state(B, self.emb.weight)
        out, Ks, Fs, gs, bs, ns, pm = [], [], [], [], [], [], []
        for t in range(T):
            logits, st, dg = self.step(st, tokens[:, t])
            out.append(logits); Ks.append(dg["K"]); Fs.append(dg["F"]); gs.append(dg["gamma"].mean()); bs.append(dg["beta"]); ns.append(dg["nov"]); pm.append(dg["pi1_max"])
        self.last_diag = {"K_mean": torch.stack(Ks).mean().item(), "F_mean": torch.stack(Fs).mean().item(), "gain": float(self.core.base_gain),
                          "lyapunov": float(self.core.lam), "gamma_mean": torch.stack(gs).mean().item(), "beta_mean": torch.stack(bs).mean().item(),
                          "novelty_mean": torch.stack(ns).mean().item(), "pi1_max": torch.stack(pm).max().item()}
        return torch.stack(out, 1), st


def lm_loss(model: FlyAI, tokens: torch.Tensor):
    logits, _ = model(tokens)
    return F.cross_entropy(logits[:, 1:].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
