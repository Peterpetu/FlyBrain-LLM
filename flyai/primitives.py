"""Building blocks shared by every FlyAI variant.

All of these are *dynamical* objects: they change per token and mostly carry no trainable
parameters. That is deliberate (see docs/aif/AXIOMS.md, AX3/AX6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DivisiveNorm(nn.Module):
    """Divisive normalisation  y = relu(z) / (sigma + mean_j relu(z_j)) * target.

    The antennal-lobe / APL motif: a single pooled inhibitory signal keeps population
    activity in a fixed working range regardless of input strength (Olsen & Wilson 2010;
    Lin 2014). `sigma` (semi-saturation) and `target` (working level) are learned scalars."""

    def __init__(self, init_sigma: float = 0.1, target: float = 1.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(init_sigma))))
        self.log_target = nn.Parameter(torch.tensor(float(np.log(target))))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = F.relu(z)
        return r / (torch.exp(self.log_sigma) + r.mean(-1, keepdim=True)) * torch.exp(self.log_target)


class GammaPrecision:
    """Online Bayesian estimate of the precision (inverse variance) of a prediction error.

    Conjugate Gamma posterior with exponential forgetting rho over a group of n error
    dimensions treated as n i.i.d. scalars sharing one precision:
        a_t = rho a_{t-1} + n/2,   b_t = rho b_{t-1} + ||e_t||^2 / 2,   pi_t = a_t / b_t
    (the same stationary precision as the per-dimension form that divides both increments
    by n; prior pseudo-count n/2). Precisions are BOUNDED (MODEL_SPEC section 6, v0.1
    revision): b is floored so a/b cannot overflow and the ratio is clamped to
    [PI_MIN, PI_MAX]. Stateless functional API; the state is an (a, b) tuple of tensors
    that every consumer reads through `ratio()`, so all of them see the same bounded value."""

    PI_MIN, PI_MAX, B_FLOOR = 1e-3, 1e3, 1e-8

    def __init__(self, rho: float = 0.9, a0: float = 1.0, b0: float = 1.0):
        self.rho, self.a0, self.b0 = rho, a0, b0

    def init(self, like: torch.Tensor):
        return (torch.full_like(like, self.a0), torch.full_like(like, self.b0))

    @classmethod
    def ratio(cls, state) -> torch.Tensor:
        a, b = state
        return (a / b.clamp_min(cls.B_FLOOR)).clamp(cls.PI_MIN, cls.PI_MAX)

    def update(self, state, err: torch.Tensor, detach: bool = True):
        a, b = state
        e2 = err.detach() if detach else err
        n = e2.shape[-1]
        a = self.rho * a + 0.5 * n
        b = (self.rho * b + 0.5 * e2.pow(2).sum(-1)).clamp_min(self.B_FLOOR)
        return (a, b), self.ratio((a, b))


class DaleCore(nn.Module):
    """Sparse, sign-constrained recurrent population with a homeostatic global gain
    regulated toward a target Lyapunov exponent.

    h_t = (1 - tau) h_{t-1} + tau * phi( W_in x_t + g * h_{t-1} @ W_eff ),      (W_in: affine input map with bias)
    W_eff = mask . exp(V) . sign[:, None],   phi = tanh (default) or divisive normalisation

    * `sign`: fixed per-unit sign, `inh_frac` of units inhibitory (MaleCNS: 0.308)
    * `mask`: fixed random sparse pattern with `fan_in` presynaptic units per unit
    * magnitudes are log-parametrised (exponentiated-gradient style; naive |W| column
      constraints train poorly, Cornford 2021) and initialised balanced (E and I input to
      every unit cancel in the mean, Rajan & Abbott 2006) and scaled so the spectral
      radius of W_eff is 1 at gain 1 -- so `g` is the classical chaos-control parameter
      (Sompolinsky 1988): lambda_max rises monotonically with g through 0 near g ~ 1.
    * `nonlin='dn'` (divisive normalisation) is available but makes the dynamics nearly
      gain-invariant (the normaliser cancels g), so criticality cannot be regulated
      through g in that mode; keep normalisation at the input / expansion stages instead.
    * `g` is a buffer regulated by `homeostatic_step`, a dithered extremum-seeking controller:
      the applied gain alternates base x (1 +/- dither) between regulation events, the slope
      d(lambda)/d(g) is estimated by demodulating the single-step Lyapunov increment with the
      dither sign, and the base gain moves against the measured slope (robust to either sign;
      settles at the peak of lambda(g) when the target is unreachable).

    `lyapunov_step` estimates the largest Lyapunov exponent of the map h -> step(h, x) with
    one Jacobian-vector product along a persistent unit perturbation (Benettin 1980)."""

    def __init__(self, n: int, d_in: int, inh_frac: float = 0.308, fan_in: int = 32, seed: int = 0,
                 init_gain: float = 1.0, tau_init: float = 0.5, nonlin: str = "tanh", target_lambda: float = -0.02):
        super().__init__()
        rng = np.random.default_rng(seed)                                       # numpy: works under torch's meta device too
        self.n, self.nonlin, self.target_lambda = n, nonlin, target_lambda
        sign_np = np.ones(n, dtype=np.float32); sign_np[rng.permutation(n)[: int(round(inh_frac * n))]] = -1.0
        exc_idx, inh_idx = np.flatnonzero(sign_np > 0), np.flatnonzero(sign_np < 0)
        # stratified fan-in: every unit receives round(inh_frac * fan_in) inhibitory and the rest
        # excitatory inputs, so E/I balance per unit is always achievable
        n_i = max(1, int(round(inh_frac * fan_in))); n_e = fan_in - n_i
        mask_np = np.zeros((n, n), dtype=np.float32)
        for j in range(n):
            e_src = rng.permutation(exc_idx)[: n_e + 1]; e_src = e_src[e_src != j][:n_e]
            i_src = rng.permutation(inh_idx)[: n_i + 1]; i_src = i_src[i_src != j][:n_i]
            mask_np[e_src, j] = 1.0; mask_np[i_src, j] = 1.0
        W_np = (rng.random((n, n), dtype=np.float32) + 0.2) * mask_np
        exc = (W_np * (sign_np[:, None] > 0)).sum(0, keepdims=True); inh = (W_np * (sign_np[:, None] < 0)).sum(0, keepdims=True)
        W_np = np.where(sign_np[:, None] < 0, W_np * (exc / np.maximum(inh, 1e-6)), W_np)
        rho_np = float(np.abs(np.linalg.eigvals(W_np * sign_np[:, None])).max()) or 1.0
        W_np = W_np / rho_np                                                    # spectral radius 1 at gain 1
        sign, mask, W = torch.tensor(sign_np), torch.tensor(mask_np), torch.tensor(W_np)
        self.register_buffer("sign", sign); self.register_buffer("mask", mask)
        self.V = nn.Parameter(torch.log(W.clamp_min(1e-6)) * mask + (1 - mask) * (-9.0))
        self.W_in = nn.Linear(d_in, n)
        self.dn = DivisiveNorm() if nonlin == "dn" else None
        self.logit_tau = nn.Parameter(torch.full((1,), float(np.log(tau_init / (1 - tau_init)))))
        self.register_buffer("gain", torch.tensor(float(init_gain)))          # gain applied in step(): base_gain x (1 +/- dither)
        self.register_buffer("base_gain", torch.tensor(float(init_gain)))     # controller set-point (what diagnostics report)
        self.register_buffer("lam", torch.tensor(0.0))                        # EMA (r = 0.05) largest-Lyapunov estimate
        self.register_buffer("lam_inst", torch.tensor(0.0))                   # last single-step log growth (for demodulation)
        self.register_buffer("lam_bar", torch.tensor(0.0))                    # EMA of lam_inst (demodulation baseline)
        self.register_buffer("delta", F.normalize(torch.tensor(rng.standard_normal(n).astype(np.float32)), dim=0))
        self.register_buffer("dither_sign", torch.tensor(1.0))
        self.register_buffer("slope", torch.tensor(1.0))                      # EMA estimate of d(lambda)/d(gain); prior +1 (Sompolinsky regime)

    @property
    def W(self):
        return torch.exp(self.V) * self.mask

    def W_eff(self) -> torch.Tensor:
        return self.mask * torch.exp(self.V) * self.sign[:, None]

    @torch.no_grad()
    def ei_column_imbalance(self) -> float:
        """Rajan-Abbott check: mean signed input per unit relative to total input magnitude."""
        We = self.W_eff(); return float(We.sum(0).abs().mean() / We.abs().sum(0).mean().clamp_min(1e-9))

    @torch.no_grad()
    def rebalance(self):
        """Rajan-Abbott projection: rescale inhibitory magnitudes per target unit so that E and I
        input sums cancel (keeps the outlier eigenvalue away as weights drift during training)."""
        We = torch.exp(self.V) * self.mask
        exc = (We * (self.sign[:, None] > 0)).sum(0, keepdim=True); inh = (We * (self.sign[:, None] < 0)).sum(0, keepdim=True)
        scale = torch.where(self.sign[:, None] < 0, (exc / inh.clamp_min(1e-9)).expand_as(We), torch.ones_like(We))
        self.V.add_(torch.log(scale.clamp_min(1e-9)) * self.mask)

    @torch.no_grad()
    def spectral_radius(self) -> float:
        return float(torch.linalg.eigvals(self.W_eff()).abs().max())

    def step(self, h: torch.Tensor, x: torch.Tensor, gain: Optional[torch.Tensor] = None) -> torch.Tensor:
        tau = torch.sigmoid(self.logit_tau)
        g = self.gain.detach().clone() if gain is None else gain
        z = self.W_in(x) + g * (h @ self.W_eff())
        r = self.dn(z) if self.dn is not None else torch.tanh(z)
        return (1 - tau) * h + tau * r

    @torch.no_grad()
    def lyapunov_step(self, h: torch.Tensor, x: torch.Tensor, r: float = 0.05) -> torch.Tensor:
        h0, x0 = h[:1].detach(), x[:1].detach()
        _, jv = torch.func.jvp(lambda hh: self.step(hh, x0), (h0,), (self.delta[None],))
        nrm = jv.norm().clamp_min(1e-12)
        self.delta.copy_(jv[0] / nrm)
        self.lam_inst.copy_(torch.log(nrm))
        self.lam.mul_(1 - r).add_(r * self.lam_inst)
        return self.lam

    @torch.no_grad()
    def set_gain(self, g: float):
        """Fix the gain (base and applied) at `g`; use this for fixed-gain arms."""
        self.base_gain.fill_(float(g)); self.gain.fill_(float(g)); return self.gain

    @torch.no_grad()
    def homeostatic_step(self, target: Optional[float] = None, eta: float = 0.02, lo: float = 0.05, hi: float = 5.0,
                         dither: float = 0.1, r_slope: float = 0.05, deadband: float = 0.01):
        """Dithered extremum-seeking control of the base gain toward lambda = target.

        Between regulation events the applied gain is base x (1 + dither * s), s = +/-1
        alternating. The slope d(lambda)/d(gain) is estimated by demodulating the single-step
        Lyapunov increment measured under that dither (EMA, rate r_slope; prior +1), and the
        base gain moves against the slope it actually measures:
            base <- base - eta * sign(slope) * (lambda - target)      (no move inside the dead-band)
        If lambda(g) peaks below the target the estimated slope changes sign at the peak and
        the gain settles there instead of running to a bound."""
        target = self.target_lambda if target is None else target
        s = self.dither_sign                                                   # sign of the dither that produced lam_inst
        self.slope.mul_(1 - r_slope).add_(r_slope * (self.lam_inst - self.lam_bar) * s / (dither * self.base_gain))
        self.lam_bar.mul_(0.9).add_(0.1 * self.lam_inst)
        err = self.lam - target
        if abs(float(err)) > deadband:
            direction = torch.sign(self.slope) if float(self.slope) != 0.0 else torch.tensor(1.0)
            self.base_gain.sub_(eta * direction * err).clamp_(lo, hi)
        self.dither_sign.neg_()
        self.gain.copy_(self.base_gain * (1.0 + dither * self.dither_sign))
        return self.gain

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None, regulate: bool = False):
        B, T, _ = x.shape
        h = h0 if h0 is not None else x.new_zeros(B, self.n)
        outs = []
        for t in range(T):
            if regulate:
                self.lyapunov_step(h, x[:, t]); self.homeostatic_step()
            h = self.step(h, x[:, t])
            outs.append(h)
        return torch.stack(outs, 1), h


class NoveltyMBON:
    """Fly Bloom-filter familiarity detector on a k-WTA code (Dasgupta et al. 2018; the alpha'3
    novelty compartment, Hattori 2017). One weight per KC, depressed by `delta` when the KC
    fires and recovering by `tick` otherwise. novelty = mean weight of the active KCs, read
    BEFORE the update, in [0, 1] (1 = never seen). Stateless functional API."""

    def __init__(self, delta: float = 0.5, tick: float = 0.01):
        self.delta, self.tick = delta, tick

    def init(self, B: int, n_kc: int, like: torch.Tensor):
        return like.new_ones(B, n_kc)

    def read(self, w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:          # idx (B, k) -> (B,)
        return w.gather(1, idx).mean(-1)

    def update(self, w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        hit = torch.zeros_like(w).scatter_(1, idx, 1.0)
        w = w + self.tick * (1.0 - w) * (1.0 - hit)          # recovery only where the KC did not fire
        return w * (1.0 - self.delta * hit)


def param_groups(model: nn.Module, weight_decay: float):
    """AdamW groups: no weight decay on log-/logit-parameterised quantities, gains, biases and
    norms (weight decay on log-magnitudes is a growth prior the Lyapunov regulator would fight)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        last = name.split(".")[-1]
        if p.ndim < 2 or last in ("V", "c_level") or last.startswith(("log", "logit", "ga", "gb", "ar_", "theta", "err_")) or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def after_optimizer_step(model: nn.Module):
    """Projections to run after every optimiser step: E/I column re-balance of every DaleCore."""
    for m in model.modules():
        if isinstance(m, DaleCore):
            m.rebalance()
