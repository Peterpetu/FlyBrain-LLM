"""Slow belief state as a bank of amplitude-free continuous attractors, v0.2.

J rings of n units (8 columns suffice; Noorman 2024): local cosine excitation, global
inhibition, two shifter populations driven by a learned scalar velocity per ring, and a
2-d sinusoidal (von Mises) drive u_j = a_j(x) cos(phi) + b_j(x) sin(phi) whose amplitude
|(a, b)| is the precision of the evidence. Units are threshold-linear with a soft ceiling
and a slow leak, so the bump amplitude grows with confirming evidence and decays without it:
amplitude = certainty (Kutschireiter 2023). `readout` returns rates plus (cos, sin, amp)
per ring; `precision` returns exp(c_j) * amp_j per ring (level-3 precision source)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttractorBank(nn.Module):
    def __init__(self, d: int, n_rings: int = 8, n_units: int = 8, shift: float = 2 * np.pi / 8, tau: float = 0.5):
        super().__init__()
        self.J, self.n = n_rings, n_units
        ang = torch.arange(n_units) * 2 * np.pi / n_units
        Wee = torch.cos(ang[:, None] - ang[None, :]).clamp_min(0)
        Wl = torch.cos(ang[:, None] - ang[None, :] + shift).clamp_min(0)
        Wr = torch.cos(ang[:, None] - ang[None, :] - shift).clamp_min(0)
        for W in (Wee, Wl, Wr):
            W /= W.sum(0, keepdim=True)
        self.register_buffer("Wee", Wee); self.register_buffer("Wl", Wl); self.register_buffer("Wr", Wr)
        self.register_buffer("cs", torch.cos(ang)); self.register_buffer("sn", torch.sin(ang))
        self.vel = nn.Linear(d, n_rings)
        self.drive = nn.Linear(d, 2 * n_rings)                        # (a_j, b_j): von Mises natural parameters
        self.log_gain = nn.Parameter(torch.zeros(3))                  # ee, shift, inhibition
        self.logit_tau = nn.Parameter(torch.full((1,), float(np.log(tau / (1 - tau)))))
        self.log_c = nn.Parameter(torch.zeros(n_rings))               # precision scale per ring
        self.leak = 0.02
        self.out_dim = n_rings * (n_units + 3)

    def init_state(self, B: int, like: torch.Tensor):
        return like.new_zeros(B, self.J, self.n) + 0.1 * self.cs.clamp_min(0)

    def step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g = torch.exp(self.log_gain); tau = torch.sigmoid(self.logit_tau)
        v = torch.tanh(self.vel(x))[..., None]
        ab = self.drive(x).view(x.shape[0], self.J, 2)
        u = ab[..., :1] * self.cs + ab[..., 1:] * self.sn                # (B, J, n)
        rec = g[0] * h @ self.Wee + g[1] * (F.relu(v) * (h @ self.Wl) + F.relu(-v) * (h @ self.Wr))
        inh = g[2] * h.mean(-1, keepdim=True)
        r = F.relu(rec - inh + u); r = r / (1.0 + 0.1 * r.mean(-1, keepdim=True))
        return (1 - tau) * h + tau * r - self.leak * h * h.mean(-1, keepdim=True)

    def polar(self, h: torch.Tensor):
        c = (h * self.cs).sum(-1); s_ = (h * self.sn).sum(-1)
        amp = torch.sqrt(c ** 2 + s_ ** 2 + 1e-8)
        return c / amp, s_ / amp, amp                                     # each (B, J)

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        c, s_, amp = self.polar(h)
        return torch.cat([h, c[..., None], s_[..., None], amp[..., None]], -1).reshape(h.shape[0], -1)

    def precision(self, h: torch.Tensor) -> torch.Tensor:              # (B, J)
        return torch.exp(self.log_c) * self.polar(h)[2]
