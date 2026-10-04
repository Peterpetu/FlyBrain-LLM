"""Module-level probe: runs a memory over a stream, scores queries by nearest value embedding,
trains the memory's own parameters (if any) by backpropagation through the stream, and reports
accuracy stratified by tag."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import FastMemory, MemConfig
from .tasks import Vocab, WRITE, QUERY, SLEEP


class Probe(nn.Module):
    def __init__(self, cfg: MemConfig, voc: Vocab, replay_recent: int = 64):
        super().__init__()
        assert cfg.d_key == voc.d_key and cfg.d_val == voc.d_val
        self.mem = FastMemory(cfg); K, V = voc.tensors()
        self.register_buffer("K", K); self.register_buffer("V", V); self.replay_recent = replay_recent

    def run(self, stream: dict, sigma: float = 0.0, rng: torch.Generator | None = None):
        """Returns (loss, per-tag accuracy dict, n queries). Gradients flow through the whole stream."""
        kinds, keys, vals, tags = stream["kinds"], torch.as_tensor(stream["keys"]), torch.as_tensor(stream["vals"]), stream["tags"]
        B, T = keys.shape; st = self.mem.init_state(B, self.K); losses, hits, tag_of = [], [], []
        recent = []
        for t in range(T):
            kind = int(kinds[t]); x = self.K[keys[:, t]]
            if kind == WRITE:
                v = self.V[vals[:, t]]
                if sigma > 0: v = v + sigma / np.sqrt(v.shape[-1]) * torch.randn(v.shape, generator=rng, device=v.device, dtype=v.dtype)   # sigma = noise/signal norm ratio
                st = self.mem.write(st, x, v); recent.append(keys[:, t])
                if len(recent) > self.replay_recent: recent.pop(0)
            elif kind == QUERY:
                r = self.mem.read(st, x)                                             # (B, d_val)
                logits = F.normalize(r, dim=-1) @ self.V.t() * 10.0                  # cosine scores (temperature 10)
                losses.append(F.cross_entropy(logits, vals[:, t], reduction="none")); hits.append((logits.argmax(-1) == vals[:, t]).float()); tag_of.append(int(tags[t]))
            else:                                                                    # SLEEP: replay the most recent writes
                if recent:
                    rx = self.K[torch.stack(recent, 1)]                              # (B, R, d_key)
                    st = self.mem.sleep(st, rx)
        if not losses: raise ValueError("stream has no query events (too short for its probe schedule?)")
        L = torch.stack(losses, 1); H = torch.stack(hits, 1); tag_of = np.asarray(tag_of)
        acc = {int(g): float(H[:, tag_of == g].mean()) for g in np.unique(tag_of)}
        acc["all"] = float(H.mean())
        return L.mean(), acc, int(L.numel())


def evaluate(probe: Probe, streams: list[dict], sigma: float = 0.0, seed: int = 0) -> dict:
    probe.eval(); g = torch.Generator().manual_seed(seed); accs = {}
    with torch.no_grad():
        for s in streams:
            _, a, _ = probe.run(s, sigma, g)
            for k, v in a.items(): accs.setdefault(k, []).append(v)
    return {("acc" if k == "all" else f"acc_tag{k}"): float(np.mean(v)) for k, v in accs.items()}


def train_probe(probe: Probe, make_stream, steps: int, lr: float, sigma: float = 0.0, seed: int = 0, log=None) -> dict:
    """Train the memory's learnable parameters (gate MLP, learned expansion, learned decays, precision
    scalars) on freshly sampled streams; no-op if the memory has no parameters."""
    params = [p for p in probe.parameters() if p.requires_grad]
    if not params or steps == 0: return {"trained": False, "steps": 0}
    opt = torch.optim.Adam(params, lr=lr); g = torch.Generator().manual_seed(seed + 7); rng = np.random.default_rng(seed + 11)
    probe.train(); hist = []
    for s in range(steps):
        stream = make_stream(rng); loss, acc, _ = probe.run(stream, sigma, g)
        if not torch.isfinite(loss): return {"trained": True, "steps": s, "diverged": True}
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        hist.append(float(loss))
        if log is not None and (s + 1) % 10 == 0: log(s + 1, float(loss), acc["all"])
    return {"trained": True, "steps": steps, "diverged": False, "train_loss_last": float(np.mean(hist[-10:]))}
