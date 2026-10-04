"""AIF-H1: K-step free-energy inference per token vs amortised-only, at matched compute.
Character-level LM on TinyShakespeare. See PREREG.md (frozen at first run)."""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyai.model import FlyAI, FlyAIConfig, lm_loss
from flyai.primitives import param_groups, after_optimizer_step
from flybrain.tasks.text import CharCorpus
from flybrain.harness import run_grid, analyse, markdown_table, write_summary
from flybrain.harness.train import train as _train  # noqa: F401  (loop below mirrors it but uses lm_loss)

EXP = Path(__file__).resolve().parent
T, B = 96, 32


def build(cfg, vocab):
    kw = {k: v for k, v in cfg.items() if k in FlyAIConfig.__dataclass_fields__}
    return FlyAI(FlyAIConfig(vocab=vocab, **kw))


def eval_bpc(model, corp, max_chars=60_000):
    model.eval(); data = corp.val[:max_chars]; n = (len(data) - 1) // T; tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, n, 64):
            idx = torch.arange(s, min(s + 64, n)) * T
            x = torch.stack([data[i:i + T + 1] for i in idx])
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(logits[:, 1:].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="sum")
            tot += float(loss); cnt += x[:, 1:].numel()
    model.train(); return tot / cnt / np.log(2)


def train_fn(cfg, seed, run_dir):
    corp = CharCorpus("tinyshakespeare"); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = build(dict(cfg, seed=seed), corp.vocab)
    opt = torch.optim.AdamW(param_groups(model, 0.01), lr=cfg["lr"], betas=(0.9, 0.95))
    steps, warm = cfg["steps"], 20; hist = []; t0 = time.time(); log = open(run_dir / "log.jsonl", "w"); diverged = False
    for s in range(steps):
        lr = cfg["lr"] * ((s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
        for g in opt.param_groups: g["lr"] = lr
        x, _ = corp.batch("train", B, T, rng)
        loss = lm_loss(model, x)
        if not torch.isfinite(loss): diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); after_optimizer_step(model)
        if (s + 1) % 100 == 0 or s + 1 == steps:
            bpc = eval_bpc(model, corp); hist.append(bpc)
            log.write(__import__("json").dumps({"step": s + 1, "loss": float(loss), "val_bpc": bpc, **model.last_diag, "t": time.time() - t0}) + "\n"); log.flush()
    log.close()
    return {"diverged": diverged, "val_bpc": hist[-1] if hist else float("nan"), "val_bpc_last2": float(np.mean(hist[-2:])) if hist else float("nan"),
            "val_bpc_hist": hist, "K_mean": model.last_diag["K_mean"], "lyapunov": model.last_diag["lyapunov"], "gain": model.last_diag["gain"],
            "params": sum(p.numel() for p in model.parameters())}


CONDS = {
    "flyai_K3": dict(K_max=3, adaptive_K=True, n=128),
    "flyai_K0": dict(K_max=0, n=128),
    "flyai_K0_wide": dict(K_max=0, n=192),          # ~2.25x core FLOPs: compute-matched amortised control
    "flyai_K3_fixed": dict(K_max=3, adaptive_K=False, n=128),
    "flyai_K3_noattr": dict(K_max=3, adaptive_K=True, n=128, use_attractor=False),   # descriptive: does the attractor bank help LM loss?
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--threads", type=int, default=4); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {k: dict(v, steps=args.steps, n_kc=512, d=64) for k, v in CONDS.items() if not args.only or k in args.only}
    lrs = [1e-3, 3e-3, 1e-2]
    res = run_grid(EXP, C, lrs, args.seeds, train_fn)
    an = analyse(res, "val_bpc_last2", "flyai_K3", ["flyai_K0", "flyai_K0_wide", "flyai_K3_fixed", "flyai_K3_noattr"], higher_is_better=False, extra_metrics=["K_mean", "lyapunov", "params"])
    md = "## AIF-H1: per-token free-energy inference (K steps) vs amortised-only, TinyShakespeare val bits/char\n\n" + markdown_table(an, "val_bpc", ["K_mean", "params"])
    write_summary(EXP, an, md); print(md)
