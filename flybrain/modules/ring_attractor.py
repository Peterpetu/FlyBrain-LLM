"""Central-complex heading circuit as a differentiable recurrent state.

Three parametrisations, same size and same velocity input, so that the *ring topology*
is the only difference:

* connectome : the extracted 148-neuron EPG / PEN_a / PEN_b / PEG / Delta7 matrix
               (contact counts, signed by neurotransmitter prediction), with one learned
               log-gain per (source population, target population) block, learned per-
               population bias and time constant. Angular velocity enters by multiplicative
               modulation of the left vs right PEN populations, as in the fly.
* canonical  : idealised ring of the same size (cosine local excitation, shifted PEN
               rings, global inhibition) -- positive control for the mechanism.
* unstructured : fully learned recurrent matrix of the same size with the same velocity
               channels -- the "plain RNN with the same velocity input" falsification control.

Velocity is a learned scalar function of the current token embedding. Readout exposes the
EPG population vector (cos, sin) and its amplitude (certainty; Kutschireiter 2023).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..connectome.circuits import load_circuit


class RingAttractor(nn.Module):
    def __init__(self, d_model: int, mode: str = "connectome", n_units: Optional[int] = None,
                 circuit_dir: Optional[Path] = None, seed: int = 0):
        super().__init__()
        self.mode = mode
        g = torch.Generator().manual_seed(seed)
        if mode == "connectome":
            z = load_circuit("central_complex", circuit_dir)
            W = torch.tensor(z["W"], dtype=torch.float32)                 # row = presynaptic
            sign = torch.tensor(z["sign"], dtype=torch.float32)
            pop = z["pop_of"].astype(str); angle = torch.tensor(z["angle"], dtype=torch.float32)
            # side of each neuron from glomerulus string (first glomerulus letter); Delta7 use output side
            glom = z["glomeruli"].astype(str)
            side = np.array([s[0] if s else "L" for s in glom])
            pops = ["EPG", "PEN_a", "PEN_b", "PEG", "Delta7"]
            pop_id = torch.tensor([pops.index(p) for p in pop])
            # normalise each block by its total so learned gains start comparable
            Wn = torch.zeros_like(W)
            for r in range(len(pops)):
                for s in range(len(pops)):
                    blk = W[pop_id == r][:, pop_id == s]
                    if blk.sum() > 0:
                        idx_r = (pop_id == r).nonzero().ravel(); idx_s = (pop_id == s).nonzero().ravel()
                        Wn[idx_r[:, None], idx_s[None, :]] = blk / blk.abs().max()
            W0 = Wn * sign[:, None]
            W0 = W0 / max(float(torch.linalg.eigvals(W0.abs()).abs().max()), 1e-6)   # unit spectral radius at init
            self.register_buffer("W0", W0)
            self.register_buffer("pop_id", pop_id)
            self.register_buffer("angle", angle)
            self.register_buffer("is_epg", pop_id == 0)
            pen = (pop_id == 1) | (pop_id == 2)
            self.register_buffer("vel_in", torch.where(pen & torch.tensor(side == "L"), 1.0, 0.0) - torch.where(pen & torch.tensor(side == "R"), 1.0, 0.0))
            self.n = W.shape[0]; self.n_pops = len(pops)
            self.log_gain = nn.Parameter(torch.zeros(self.n_pops, self.n_pops))
        else:
            n = n_units or 148
            self.n = n
            n_epg = n // 3; n_pen = n // 3; n_inh = n - n_epg - n_pen
            pop_id = torch.cat([torch.zeros(n_epg), torch.ones(n_pen), torch.full((n_inh,), 2)]).long()
            angle = torch.zeros(n)
            angle[:n_epg] = torch.arange(n_epg) * 2 * np.pi / n_epg
            angle[n_epg:n_epg + n_pen] = torch.arange(n_pen) * 2 * np.pi / n_pen
            self.register_buffer("pop_id", pop_id); self.register_buffer("angle", angle)
            self.register_buffer("is_epg", pop_id == 0)
            vel = torch.zeros(n); half = n_pen // 2
            vel[n_epg:n_epg + half] = 1.0; vel[n_epg + half:n_epg + n_pen] = -1.0
            self.register_buffer("vel_in", vel)
            self.n_pops = 3
            if mode == "canonical":
                W = torch.zeros(n, n)
                ae, ap = angle[:n_epg], angle[n_epg:n_epg + n_pen]
                d = ae[:, None] - ae[None, :]
                W[:n_epg, :n_epg] = torch.cos(d).clamp_min(0)                                 # local excitation
                shift = 2 * np.pi / 8
                dl = ap[:half, None] - ae[None, :] + shift; dr = ap[half:, None] - ae[None, :] - shift
                W[n_epg:n_epg + half, :n_epg] = torch.cos(dl).clamp_min(0)                   # PEN_L -> EPG shifted
                W[n_epg + half:n_epg + n_pen, :n_epg] = torch.cos(dr).clamp_min(0)           # PEN_R -> EPG shifted
                W[:n_epg, n_epg:n_epg + n_pen] = torch.cos(ae[:, None] - ap[None, :]).clamp_min(0)  # EPG -> PEN local
                W[:n_epg, n_epg + n_pen:] = 1.0 / n_inh                                        # EPG -> inhibitory
                W[n_epg + n_pen:, :n_epg + n_pen] = -1.0 / n_inh                               # global inhibition
                W = W / max(float(torch.linalg.eigvals(W.abs()).abs().max()), 1e-6)
                self.register_buffer("W0", W)
                self.log_gain = nn.Parameter(torch.zeros(self.n_pops, self.n_pops))
            elif mode == "unstructured":
                self.W_free = nn.Parameter(torch.randn(n, n, generator=g) / np.sqrt(n))
            else:
                raise ValueError(mode)
        self.bias = nn.Parameter(torch.zeros(self.n_pops))
        self.log_tau = nn.Parameter(torch.zeros(self.n_pops))
        self.vel = nn.Linear(d_model, 1)                     # learned angular velocity from the token
        self.vel_gain = nn.Parameter(torch.ones(1))
        self.drive = nn.Linear(d_model, self.n)              # learned direct input drive (all modes)
        self.out_dim = self.n + 3

    def W(self):
        if self.mode == "unstructured":
            return self.W_free
        G = torch.exp(self.log_gain)[self.pop_id][:, self.pop_id]
        return self.W0 * G

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        W = self.W(); tau = torch.sigmoid(self.log_tau)[self.pop_id]; b = self.bias[self.pop_id]
        v = torch.tanh(self.vel(x)).squeeze(-1) * self.vel_gain                  # (B,T)
        u = self.drive(x)                                                        # (B,T,n)
        h = state if state is not None else x.new_zeros(B, self.n)
        if state is None:  # initialise a bump at angle 0 on EPG units
            h = h + self.is_epg.float() * torch.cos(self.angle).clamp_min(0)
        outs = []
        cs, sn = torch.cos(self.angle), torch.sin(self.angle)
        for t in range(T):
            mod = 1.0 + v[:, t, None] * self.vel_in[None, :]                     # PEN_L up / PEN_R down (or vice versa)
            inp = (h * mod) @ W + u[:, t] + b
            h = (1 - tau) * h + tau * torch.tanh(F.relu(inp))          # bounded firing rate, same in every mode
            e = h * self.is_epg
            pv_c = (e * cs).sum(-1, keepdim=True); pv_s = (e * sn).sum(-1, keepdim=True)
            amp = torch.sqrt(pv_c ** 2 + pv_s ** 2 + 1e-8)
            outs.append(torch.cat([h, pv_c / amp, pv_s / amp, amp], -1))
        return torch.stack(outs, 1), h
