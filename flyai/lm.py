"""HybridFlyAILM: a decoder with FlyAI filter blocks and softmax attention every 4th block
(the Qwen3-Next / Kimi-Linear 3:1 layout), plus a twin with Gated-DeltaNet mixers in the
same slots for matched comparisons. Standard causal next-token training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import FlyAIBlock, BlockConfig
from flybrain.models.backbone import Attention
from flybrain.modules.fast_weights import FastWeightMemory, FWConfig


@dataclass
class LMConfig:
    vocab: int = 50257
    d: int = 768
    n_layers: int = 12
    n_heads: int = 12
    mixer: str = "flyai"          # flyai | gdn | attention
    attn_every: int = 4
    mlp_mult: int = 4
    tie_weights: bool = True
    block: dict = field(default_factory=dict)      # BlockConfig overrides
    gdn_key: int = 64
    seed: int = 0                                  # seeds the fixed random structures (Dale mask/sign, KC expansion) of every block


class Layer(nn.Module):
    def __init__(self, cfg: LMConfig, i: int):
        super().__init__(); d = cfg.d
        use_attn = cfg.mixer == "attention" or (cfg.attn_every > 0 and (i + 1) % cfg.attn_every == 0)
        self.kind = "attention" if use_attn else cfg.mixer
        self.n1 = nn.RMSNorm(d); self.n2 = nn.RMSNorm(d)
        if self.kind == "attention":
            self.mix = Attention(d, n_heads=cfg.n_heads)
        elif self.kind == "flyai":
            self.mix = FlyAIBlock(BlockConfig(d=d, n_heads=cfg.n_heads, seed=cfg.seed * 1000 + i, **cfg.block))
        elif self.kind == "gdn":
            self.mix = FastWeightMemory(FWConfig(d_model=d, n_heads=cfg.n_heads, d_value=d // cfg.n_heads, key_mode="dense", d_key=cfg.gdn_key,
                                                 write="delta", gate="learned", decay="learned", conv_kernel=4))
        else:
            raise ValueError(cfg.mixer)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_mult * d), nn.GELU(), nn.Linear(cfg.mlp_mult * d, d))

    def forward(self, x, state=None):
        y, st = self.mix(self.n1(x), state) if self.kind != "attention" else (self.mix(self.n1(x))[0], None)
        x = x + y
        return x + self.mlp(self.n2(x)), st


class HybridFlyAILM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__(); self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d); nn.init.normal_(self.emb.weight, std=0.02)   # GPT-2 scale; tied head
        self.layers = nn.ModuleList([Layer(cfg, i) for i in range(cfg.n_layers)])
        self.norm = nn.RMSNorm(cfg.d); self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        if cfg.tie_weights: self.head.weight = self.emb.weight

    def forward(self, tokens, states=None):
        x = self.emb(tokens); new_states = []
        for i, layer in enumerate(self.layers):
            x, st = layer(x, None if states is None else states[i]); new_states.append(st)
        return self.head(self.norm(x)), new_states

    def loss(self, tokens):
        logits, _ = self(tokens[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), tokens[:, 1:].reshape(-1))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_numel(self) -> int:
        """Recurrent state carried per batch row across all recurrent layers (elements)."""
        return sum(l.mix.state_numel() for l in self.layers if l.kind == "flyai")

    def diagnostics(self):
        out = {}
        for i, l in enumerate(self.layers):
            if l.kind == "flyai":
                out[f"layer{i}_gain"] = float(l.mix.core.base_gain); out[f"layer{i}_lyapunov"] = float(l.mix.core.lam)
        return out


def make_lm(size: str = "125m", mixer: str = "flyai", vocab: int = 50257, seed: int = 0, **block) -> HybridFlyAILM:
    sizes = {"tiny": dict(d=64, n_layers=4, n_heads=4), "20m": dict(d=384, n_layers=8, n_heads=6), "50m": dict(d=512, n_layers=8, n_heads=8),
             "125m": dict(d=768, n_layers=12, n_heads=12), "350m": dict(d=1024, n_layers=24, n_heads=16)}
    s = sizes[size]
    blk = dict(n_kc=2048 if size != "tiny" else 128, n_pn=64 if size != "tiny" else 16, d_v=max(8, s["d"] // s["n_heads"] // 2),
               n_rings=8 if size != "tiny" else 2, n_ring_units=8, n_groups=s["n_heads"], fan_in=8 if size != "tiny" else 5)
    gdn_key = block.pop("gdn_key", None) or (s["d"] // s["n_heads"])      # matched twin: key dim = head dim
    blk.update(block)
    return HybridFlyAILM(LMConfig(vocab=vocab, mixer=mixer, block=blk, gdn_key=gdn_key, seed=seed, **s))
