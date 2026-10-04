"""Minimal pre-norm residual sequence model with a pluggable token mixer.

mixer in {"fastweight", "ring", "rnn", "attention"}; every layer = mixer + MLP with
residuals, RMSNorm, tied output head optional. Small by design (CPU-scale experiments).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.fast_weights import FastWeightMemory, FWConfig
from ..modules.ring_attractor import RingAttractor
from ..modules.masked_rnn import MaskedDaleRNN


@dataclass
class ModelConfig:
    vocab: int = 64
    d_model: int = 64
    n_layers: int = 2
    mixer: str = "fastweight"
    mixer_kwargs: dict = field(default_factory=dict)
    mlp_mult: int = 2
    n_out: Optional[int] = None      # classification head size (defaults to vocab)


class Attention(nn.Module):
    def __init__(self, d, n_heads=2):
        super().__init__(); self.h = n_heads; self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)

    def forward(self, x, state=None):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, d // self.h).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        pos = torch.arange(T, device=x.device); q, k = _rope(q, pos), _rope(k, pos)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, T, d)), None


def _rope(x, pos):
    d = x.shape[-1]; half = d // 2
    freqs = 1.0 / (10000 ** (torch.arange(0, half, device=x.device) / half))
    ang = pos[:, None] * freqs[None]; c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


class Mixer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model; kw = dict(cfg.mixer_kwargs)
        if cfg.mixer == "fastweight":
            self.core = FastWeightMemory(FWConfig(d_model=d, **kw)); self.proj = None
        elif cfg.mixer == "ring":
            self.core = RingAttractor(d, **kw); self.proj = nn.Linear(self.core.out_dim, d)
        elif cfg.mixer == "rnn":
            n = kw.pop("n_hidden"); self.core = MaskedDaleRNN(d, kw.pop("W_adj", None), kw.pop("sign", None), n, **kw)
            self.proj = nn.Linear(n, d)
        elif cfg.mixer == "attention":
            self.core = Attention(d, **kw); self.proj = None
        else:
            raise ValueError(cfg.mixer)

    def forward(self, x):
        y, _ = self.core(x)
        return self.proj(y) if self.proj is not None else y


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n1 = nn.RMSNorm(d); self.mix = Mixer(cfg)
        self.n2 = nn.RMSNorm(d); self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_mult * d), nn.GELU(), nn.Linear(cfg.mlp_mult * d, d))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.mlp(self.n2(x))


class SeqModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_out or cfg.vocab)

    def forward(self, tokens):
        x = self.emb(tokens)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
