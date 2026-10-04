"""One training loop for every experiment: AdamW, linear warm-up + cosine decay, gradient
clipping, periodic evaluation with fixed held-out batches, jsonl logging."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


def train(model, batch_fn: Callable[[np.random.Generator], tuple], steps: int, lr: float, evals: dict[str, Callable[[], float]],
          seed: int, run_dir: Path | None = None, eval_every: int = 50, warmup: int = 20, wd: float = 0.01, clip: float = 1.0,
          ignore_index: int = -100, min_lr_frac: float = 0.1) -> dict:
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    sched = lambda s: (s + 1) / warmup if s < warmup else min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
    log = open(run_dir / "log.jsonl", "w") if run_dir else None
    hist = {k: [] for k in evals}; losses = []; diverged = False; t0 = time.time()
    model.train()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * sched(s)
        x, y = batch_fn(rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=ignore_index)
        if not torch.isfinite(loss):
            diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip); opt.step()
        losses.append(float(loss))
        if (s + 1) % eval_every == 0 or s + 1 == steps:
            model.eval()
            with torch.no_grad():
                ev = {k: float(f()) for k, f in evals.items()}
            model.train()
            for k, v in ev.items():
                hist[k].append(v)
            if log:
                log.write(json.dumps({"step": s + 1, "loss": float(np.mean(losses[-eval_every:])), **ev, "t": time.time() - t0}) + "\n"); log.flush()
    if log:
        log.close()
    out = {"diverged": diverged, "train_loss_final": float(np.mean(losses[-50:])) if losses else float("nan"), "steps_done": len(losses)}
    for k, h in hist.items():
        out[k] = h[-1] if h else float("nan")
        out[k + "_last2"] = float(np.mean(h[-2:])) if h else float("nan")
        out[k + "_best"] = float(np.max(h)) if h else float("nan")
        out[k + "_min"] = float(np.min(h)) if h else float("nan")
        out[k + "_hist"] = h
    return out
