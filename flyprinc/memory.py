"""A fast-weight associative memory in which every fly-derived principle is a switch with a
matched control (docs/phase3/EXPERIMENT_PLAN.md).

    address    dense            keys are the (normalised) input vectors: d_key rows, all active
               sparse_random    fixed random expansion (fan_in inputs per cell) + k-WTA: n_cells rows, k active   [P1]
               sparse_learned   learned expansion + k-WTA                                                      (control)
               dense_expanded   learned expansion, no k-WTA: n_cells rows, all active                          (control)
    normalise  divisive normalisation of the input before the expansion                                        [P9]
    write      delta (error-corrected) | hebb                                                                  [P2]
    gate       const | precision (a = 1, b = 0 fixed) | precision_learned (a, b learned) | learned (MLP)         [P3]
    betas      per-compartment write rates (const gate); e.g. (0.1, 1.0) = slow integrating + fast labile trace
    route      all | familiarity (row-level trace; saturates under load, kept as a control) |
               confirmation: compartment c writes only when the observation AGREES with the fast compartment's
               own prediction by more than route_thresh[c] (cosine): surprise goes to the labile trace, a
               confirmed repeat reaches the slow compartments (the spacing rule)                              [P5]
    decays     per-compartment decay (fixed log-spaced or learned)                                             [P5]
    readout    sum | recency (newest compartment wins) | confirm: two compartments (slow, labile); the labile trace is
               trusted in proportion to the number of CONSECUTIVE observations that confirmed it, counted
               content-wise (a confirmation vector accumulated along the observed value, reset on contradiction) [P4]
    slow_rows  a slow dense store filled only during `sleep` by replay of the fast store                        [P6]

Compute is matched by the number of active rows per event (k for sparse addresses, d_key for
dense ones); state is NOT matched by construction and is always reported (state_numel).
Everything is batched over B independent streams; all state is functional (dicts of tensors)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ADDRESSES = ("dense", "sparse_random", "sparse_learned", "dense_expanded")
WRITES = ("delta", "hebb")
GATES = ("const", "precision", "precision_learned", "learned")
READOUTS = ("sum", "recency", "confirm", "learned")
ROUTES = ("all", "familiarity", "confirmation")
STATE_INPUTS = ("nconf", "a", "zc", "logage")      # per-address STATES a learned gate may be given (E0-admitted: nconf, a, zc)


@dataclass
class MemConfig:
    d_key: int = 64
    d_val: int = 16
    address: str = "dense"
    n_cells: int = 1024
    k_active: int = 64
    fan_in: int = 7
    normalise: bool = True
    write: str = "delta"
    gate: str = "const"
    beta: float = 1.0                             # const gate value (all compartments) unless `betas` is given
    betas: Optional[Sequence[float]] = None       # per-compartment const rates
    rho: float = 0.9                              # precision-tracker forgetting
    n_compartments: int = 1
    decays: Optional[Sequence[float]] = None      # per-compartment decay alpha; None = no decay
    learn_decay: bool = False
    route: str = "all"
    route_thresh: Optional[Sequence[float]] = None   # per-compartment threshold on familiarity (route = familiarity) or on agreement (route = confirmation)
    fam_decay: float = 0.995                      # per-write decay of the familiarity trace
    readout: str = "sum"
    confirm_tol: float = 1.5                      # confirm readout: an observation confirms the labile trace if its squared error is below
                                                  # confirm_tol x the tracked expected squared error (so the criterion adapts to the noise level)
    slow_rows: int = 0                            # 0 = no consolidation store
    slow_eta: float = 0.5
    sleep_passes: int = 4
    sleep_select: str = "readable"                # readable | random
    sleep_thresh: float = 0.3                     # readability: ||fast read|| above this (values are unit norm)
    gate_inputs: Sequence[str] = ()               # learned gate: extra per-address STATE inputs (subset of STATE_INPUTS); () = the free control
    n_state_slots: int = 4                        # fixed number of state input slots (unused slots read 0): arms with and without states have equal parameters
    familiar_norm: float = 0.25                   # ||fast read|| above which an address counts as already written (contradiction mass zc)
    seed: int = 0

    def __post_init__(self):
        for name, val, allowed in (("address", self.address, ADDRESSES), ("write", self.write, WRITES), ("gate", self.gate, GATES),
                                   ("readout", self.readout, READOUTS), ("route", self.route, ROUTES)):
            if val not in allowed: raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
        C = self.n_compartments
        for name, seq in (("decays", self.decays), ("betas", self.betas), ("route_thresh", self.route_thresh)):
            if seq is not None and len(seq) != C: raise ValueError(f"{name}: one entry per compartment")
        if self.route in ("familiarity", "confirmation") and self.route_thresh is None: raise ValueError(f"route={self.route} needs route_thresh")
        if self.readout in ("confirm", "learned") and C != 2: raise ValueError(f"readout={self.readout} needs exactly 2 compartments (slow, labile)")
        if self.sleep_select not in ("readable", "random"): raise ValueError("sleep_select")
        self.gate_inputs = tuple(self.gate_inputs)
        if any(g not in STATE_INPUTS for g in self.gate_inputs): raise ValueError(f"gate_inputs must be a subset of {STATE_INPUTS}")
        if len(set(self.gate_inputs)) != len(self.gate_inputs) or len(self.gate_inputs) > self.n_state_slots: raise ValueError("gate_inputs: unique, at most n_state_slots")
        if self.gate_inputs and self.gate != "learned" and self.readout != "learned": raise ValueError("gate_inputs only apply to gate=learned or readout=learned")

    @property
    def tracks_contradiction(self) -> bool:
        return "a" in self.gate_inputs or "zc" in self.gate_inputs

    @property
    def tracks_confirmation(self) -> bool:
        return self.readout == "confirm" or "nconf" in self.gate_inputs or "zc" in self.gate_inputs

    @property
    def rows(self) -> int:
        return self.d_key if self.address == "dense" else self.n_cells

    @property
    def active_rows(self) -> int:
        return self.d_key if self.address == "dense" else (self.n_cells if self.address == "dense_expanded" else self.k_active)

    @property
    def state_numel(self) -> int:
        return self.n_compartments * self.rows * self.d_val + self.slow_rows * self.d_val


class FastMemory(nn.Module):
    def __init__(self, c: MemConfig):
        super().__init__()
        self.c = c; g = torch.Generator().manual_seed(c.seed); C = c.n_compartments
        if c.address == "sparse_random":
            E = torch.zeros(c.n_cells, c.d_key)
            for i in range(c.n_cells): E[i, torch.randperm(c.d_key, generator=g)[:c.fan_in]] = 1.0 / c.fan_in
            self.register_buffer("E", E)
        elif c.address in ("sparse_learned", "dense_expanded"):
            self.E = nn.Parameter(torch.randn(c.n_cells, c.d_key, generator=g) / np.sqrt(c.d_key))
        if c.address in ("sparse_learned", "dense_expanded"): self.theta = nn.Parameter(torch.zeros(1))     # learned threshold with a learned expansion
        elif c.address == "sparse_random": self.register_buffer("theta", torch.zeros(1))                   # fixed module: no trainable parameter
        else: self.theta = None
        if c.gate in ("precision", "precision_learned"):
            a, b = torch.ones(C), torch.zeros(C)
            if c.gate == "precision_learned": self.ga, self.gb = nn.Parameter(a), nn.Parameter(b)
            else: self.register_buffer("ga", a); self.register_buffer("gb", b)
        elif c.gate == "learned":
            # inputs: key, value, error, log squared error, log read norm (the delta rule's own intermediates) + n_state_slots state slots
            self.gate_mlp = nn.Sequential(nn.Linear(c.d_key + 2 * c.d_val + 2 + c.n_state_slots, 32), nn.Tanh(), nn.Linear(32, C))
        if c.readout == "learned":                                                                # learned mixing of the (slow, labile) reads:
            self.readout_mlp = nn.Sequential(nn.Linear(3 + c.n_state_slots, 16), nn.Tanh(), nn.Linear(16, 1))   # inputs: cos(slow, labile), log norms, state slots
        self.register_buffer("betas", torch.tensor(list(c.betas) if c.betas is not None else [c.beta] * C, dtype=torch.float32))
        dec = torch.tensor(list(c.decays) if c.decays is not None else [1.0] * C, dtype=torch.float32).clamp(1e-4, 1 - 1e-6)
        if c.learn_decay: self.decay_logit = nn.Parameter(torch.log(dec / (1 - dec)))
        else: self.register_buffer("decay_fixed", dec)
        if c.route in ("familiarity", "confirmation"):
            self.register_buffer("route_thresh", torch.tensor(list(c.route_thresh), dtype=torch.float32))
        if c.slow_rows > 0:
            self.register_buffer("Ws", torch.randn(c.rows, c.slow_rows, generator=g) / np.sqrt(c.rows))

    # ---- helpers ---------------------------------------------------------------------
    def decays(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logit) if self.c.learn_decay else self.decay_fixed

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        if not self.c.normalise: return x
        return x / (0.1 + x.abs().mean(-1, keepdim=True))                    # divisive normalisation by pooled activity

    def code(self, x: torch.Tensor) -> torch.Tensor:                        # (B, d_key) -> (B, rows), unit norm
        c = self.c; x = self._normalise(x)
        if c.address == "dense":
            return F.normalize(x, dim=-1)
        z = x @ self.E.t() - self.theta
        if c.address == "dense_expanded":
            return F.normalize(z, dim=-1)
        top, idx = z.topk(c.k_active, dim=-1)
        out = torch.zeros_like(z).scatter(-1, idx, F.relu(top) + 1e-6)     # k-WTA (+1e-6 keeps exactly k rows active)
        return F.normalize(out, dim=-1)

    def init_state(self, B: int, like: torch.Tensor) -> dict:
        c = self.c
        st = {"M": like.new_zeros(B, c.n_compartments, c.rows, c.d_val), "a": like.new_ones(B, c.n_compartments),
              "b": like.new_full((B, c.n_compartments), 1.0 / c.d_val),           # prior expected squared error per dim of a unit-norm value: 1/d_val
              "t": like.new_zeros(B, c.n_compartments, c.rows), "fam": like.new_zeros(B, c.rows),
              "conf": like.new_zeros(B, c.rows, c.d_val), "clock": like.new_zeros(B)}                  # conf: confirmation vectors per row
        if c.slow_rows > 0: st["Ms"] = like.new_zeros(B, c.slow_rows, c.d_val)
        if c.tracks_contradiction:                                                       # contradiction mass (ATTEST evidence channel) and its floor mass;
            st["Z"] = like.new_zeros(B, c.rows); st["sig"] = like.new_zeros(B); st["Zc"] = like.new_zeros(B, c.rows)   # familiarity-gated variant
        return st

    def rho_floor(self) -> float:                                                        # expected crosstalk of a scalar row accumulator, per unit mass
        c = self.c
        return c.k_active / c.n_cells if c.address in ("sparse_random", "sparse_learned") else 1.0 / np.sqrt(c.rows)

    def state_inputs(self, st: dict, k: torch.Tensor, pred_l: torch.Tensor) -> torch.Tensor:
        """The per-address state slots shown to the learned gate, (B, n_state_slots); unused slots are 0. All detached."""
        c = self.c; B = k.shape[0]; slots = k.new_zeros(B, c.n_state_slots)
        for j, name in enumerate(c.gate_inputs):
            if name == "nconf":                                                          # confirmations counted along the labile trace's own value
                slots[:, j] = (torch.einsum("br,brv->bv", k, st["conf"]) * F.normalize(pred_l, dim=-1)).sum(-1)
            elif name == "a":                                                            # relu(Z . k - rho sigma), log scale
                slots[:, j] = torch.log1p(F.relu((k * st["Z"]).sum(-1) - self.rho_floor() * st["sig"]))
            elif name == "zc":
                slots[:, j] = torch.log1p(F.relu((k * st["Zc"]).sum(-1)))
            elif name == "logage":                                                       # median age of the address's active rows
                idx = k.topk(c.active_rows, dim=-1).indices
                slots[:, j] = torch.log1p((st["clock"][:, None] - st["t"][:, 0].gather(-1, idx)).median(-1).values)
        return slots.detach()

    def familiarity(self, st: dict, k: torch.Tensor) -> torch.Tensor:      # code-weighted familiarity of the address, (B,)
        return (k * st["fam"]).sum(-1) / k.sum(-1).clamp_min(1e-6)

    # ---- read ---------------------------------------------------------------------------
    def read_fast(self, st: dict, x: torch.Tensor, code: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.c; k = self.code(x) if code is None else code
        reads = torch.einsum("br,bcrv->bcv", k, st["M"])                              # (B, C, d_val)
        if c.readout == "recency" and c.n_compartments > 1:
            age = torch.einsum("br,bcr->bc", k, st["t"]); w = torch.softmax(age * 4.0, dim=1)
            return (w[..., None] * reads).sum(1)
        if c.readout == "confirm":
            conf_read = torch.einsum("br,brv->bv", k, st["conf"])                       # accumulated confirmations at this address (vector)
            n_conf = (conf_read * F.normalize(reads[:, 1], dim=-1)).sum(-1)              # ... counted along the labile trace's own value
            lam = torch.sigmoid(2.0 * (n_conf - 1.0))[:, None]                         # 0.12 unconfirmed, 0.5 once, 0.88 twice
            return reads[:, 0] * (1 - lam) + reads[:, 1] * lam
        if c.readout == "learned":                                                      # the same mixing, learned from free intermediates + state slots
            r0, r1 = reads[:, 0], reads[:, 1]
            feats = torch.cat([F.cosine_similarity(r0, r1, dim=-1)[:, None], torch.log(r0.norm(dim=-1) + 1e-6)[:, None], torch.log(r1.norm(dim=-1) + 1e-6)[:, None],
                               self.state_inputs(st, k, r1)], -1)
            lam = torch.sigmoid(self.readout_mlp(feats))
            return r0 * (1 - lam) + r1 * lam
        return reads.sum(1)

    def read(self, st: dict, x: torch.Tensor) -> torch.Tensor:
        k = self.code(x); r = self.read_fast(st, x, k)
        if self.c.slow_rows > 0:
            ks = F.normalize(k @ self.Ws, dim=-1); r = r + torch.einsum("bs,bsv->bv", ks, st["Ms"])
        return r

    # ---- write --------------------------------------------------------------------------
    def write(self, st: dict, x: torch.Tensor, v: torch.Tensor, gamma: Optional[torch.Tensor] = None) -> dict:
        c = self.c; k = self.code(x); B = x.shape[0]; C = c.n_compartments
        pred = torch.einsum("br,bcrv->bcv", k, st["M"])                                # (B, C, d_val)
        err = v[:, None, :] - pred if c.write == "delta" else v[:, None, :].expand(-1, C, -1)
        e2 = (v[:, None, :] - pred).pow(2).mean(-1)                                    # (B, C) true squared error
        pi = st["a"] / st["b"].clamp_min(1e-8)
        fam = self.familiarity(st, k)                                                  # (B,)
        e0 = (v - pred[:, 0]).detach(); rn = pred[:, 0].detach().norm(dim=-1)          # the fast compartment's own error and read norm
        if c.gate == "const":
            beta = self.betas[None, :].expand(B, -1)
        elif c.gate in ("precision", "precision_learned"):
            beta = torch.sigmoid(self.ga * torch.log(pi.detach() * e2.detach() + 1e-6) + self.gb)
        else:                                                                           # learned gate: free intermediates + the state slots
            feats = torch.cat([x, v, e0, torch.log(e2[:, 0].detach() + 1e-6)[:, None], torch.log(rn + 1e-6)[:, None],
                               self.state_inputs(st, k, pred[:, -1])], -1)
            beta = torch.sigmoid(self.gate_mlp(feats))
        if gamma is not None: beta = beta * gamma[:, None]
        agree_cos = F.cosine_similarity(pred[:, 0], v, dim=-1)                          # does the observation agree with the fast compartment's prediction?
        if c.route == "familiarity":                                                    # row-level familiarity (control; saturates under load)
            beta = beta * (fam[:, None] >= self.route_thresh[None, :]).float()
        elif c.route == "confirmation":                                                 # spacing rule: a confirmed repeat reaches the slow compartments
            beta = beta * (agree_cos[:, None] >= self.route_thresh[None, :]).float()
        a = c.rho * st["a"] + 0.5; b = c.rho * st["b"] + 0.5 * e2.detach()
        dec = self.decays()[None, :, None, None]
        M = st["M"] * dec + beta[..., None, None] * k[:, None, :, None] * err[:, :, None, :]
        clock = st["clock"] + 1.0; active = (k > 0)
        t = torch.where(active[:, None, :], clock[:, None, None].expand_as(st["t"]), st["t"])
        fam_new = st["fam"] * c.fam_decay + active.float() * (1 - st["fam"] * c.fam_decay)          # rises toward 1 where written
        conf = st["conf"]; new = {}
        if c.tracks_confirmation:                                                       # does this observation confirm the labile trace (last compartment)?
            expected = (st["b"][:, -1] / st["a"][:, -1].clamp_min(1e-8)).clamp_min(1e-4)     # tracked expected squared error of the labile trace
            agree = (e2[:, -1] < c.confirm_tol * expected).float()[:, None]
            conf_read = torch.einsum("br,brv->bv", k, conf)
            target = agree * (conf_read + v) + (1 - agree) * 0.0                        # +1 along v on agreement, reset to 0 on contradiction
            conf = conf + k[:, :, None] * (target - conf_read)[:, None, :]              # delta write of the confirmation vector (content-based count)
        if c.tracks_contradiction:                                                      # |written residual| mass at the address (E0: a, zc)
            ev = e0.abs().mean(-1); b0 = beta[:, 0].detach()
            new["Z"] = st["Z"] + b0[:, None] * k * ev[:, None]; new["sig"] = st["sig"] + b0 * ev
            contradict = (1 - agree[:, 0]) * (rn >= c.familiar_norm).float()            # a disagreement at an address that already holds something
            new["Zc"] = st["Zc"] + b0[:, None] * k * (ev * contradict)[:, None]
        return {**st, "M": M, "a": a, "b": b, "t": t, "fam": fam_new, "conf": conf, "clock": clock, **new}

    # ---- consolidation ------------------------------------------------------------------
    @torch.no_grad()
    def sleep(self, st: dict, replay_x: torch.Tensor) -> dict:
        """Replay the given keys (B, R, d_key) through the fast store and consolidate what is still
        readable into the slow store with `sleep_passes` delta-rule passes."""
        c = self.c
        if c.slow_rows == 0: return st
        Ms = st["Ms"]
        for _ in range(c.sleep_passes):
            for j in range(replay_x.shape[1]):
                x = replay_x[:, j]; k = self.code(x); r = self.read_fast(st, x, k)
                keep = (r.norm(dim=-1) > c.sleep_thresh).float() if c.sleep_select == "readable" else torch.ones(r.shape[0], device=r.device)
                ks = F.normalize(k @ self.Ws, dim=-1); pred = torch.einsum("bs,bsv->bv", ks, Ms)
                Ms = Ms + c.slow_eta * keep[:, None, None] * ks[:, :, None] * (r - pred)[:, None, :]
        return {**st, "Ms": Ms}
