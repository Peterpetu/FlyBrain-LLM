"""AIF-H3: arousal (precision hyper-prior from running surprise) on topic-switch streams."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyai.model import FlyAI, FlyAIConfig
from flyai.primitives import param_groups, after_optimizer_step
from flyai.tasks import TopicSwitchStream
from flybrain.harness import run_grid, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
T, B, POST = 96, 32, 16


def evaluate(model, batches):
    model.eval(); tot, cnt, ptot, pcnt = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for x, y, sw in batches:
            logits, _ = model(torch.cat([x[:, :1], y], 1))          # predict y[:, t] from x[:, :t+1]
            nll = F.cross_entropy(logits[:, 1:].reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(y.shape)
            tot += float(nll.sum()); cnt += nll.numel()
            post = torch.zeros_like(sw)
            for k in range(POST):
                post[:, k:] |= sw[:, :sw.shape[1] - k]
            ptot += float(nll[post].sum()); pcnt += int(post.sum())
    model.train(); return tot / cnt / np.log(2), ptot / max(pcnt, 1) / np.log(2)


def train_fn(cfg, seed, run_dir):
    stream = TopicSwitchStream(); torch.manual_seed(seed); rng = np.random.default_rng(seed); ev = np.random.default_rng(1234)
    fixed = [stream.batch("val", 64, T, ev) for _ in range(4)]
    kw = {k: v for k, v in cfg.items() if k in FlyAIConfig.__dataclass_fields__ and k != 'seed'}
    model = FlyAI(FlyAIConfig(vocab=stream.vocab, **kw, seed=seed))
    opt = torch.optim.AdamW(param_groups(model, 0.01), lr=cfg["lr"], betas=(0.9, 0.95))
    steps, warm = cfg["steps"], 20; hist = []; t0 = time.time(); log = open(run_dir / "log.jsonl", "w"); diverged = False
    for s in range(steps):
        lr = cfg["lr"] * ((s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
        for g in opt.param_groups: g["lr"] = lr
        x, y, _ = stream.batch("train", B, T, rng)
        logits, _ = model(torch.cat([x[:, :1], y], 1))
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, logits.shape[-1]), y.reshape(-1))
        if not torch.isfinite(loss): diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); after_optimizer_step(model)
        if (s + 1) % 100 == 0 or s + 1 == steps:
            bpc, post = evaluate(model, fixed); hist.append((bpc, post))
            log.write(json.dumps({"step": s + 1, "loss": float(loss), "val_bpc": bpc, "post_switch_bpc": post, **model.last_diag, "t": time.time() - t0}) + "\n"); log.flush()
    log.close(); h = np.array(hist) if hist else np.full((1, 2), np.nan)
    if diverged: h = np.full_like(h, np.nan)
    return {"diverged": diverged, "val_bpc_last2": float(h[-2:, 0].mean()), "post_switch_bpc_last2": float(h[-2:, 1].mean()), "params": sum(p.numel() for p in model.parameters())}


CONDS = {"arousal": dict(arousal="both"), "arousal_pi0": dict(arousal="pi0"), "arousal_beta": dict(arousal="beta"), "arousal_learned": dict(arousal="learned"), "no_arousal": dict(arousal="none")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--variant", default=None, help="e.g. replication: separate runs_<variant>/ and SUMMARY_<variant>.md"); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {k: dict(v, steps=args.steps, K_max=1, adaptive_K=False, n=128, n_kc=512, d=64) for k, v in CONDS.items() if not args.only or k in args.only}   # PREREG: K = 1 exactly
    res = run_grid(EXP, C, [1e-3, 3e-3, 1e-2], args.seeds, train_fn, variant=args.variant)
    an = analyse(res, "val_bpc_last2", "arousal", ["no_arousal", "arousal_learned", "arousal_pi0", "arousal_beta"], higher_is_better=False, extra_metrics=["post_switch_bpc_last2", "params"])
    md = "## AIF-H3: topic-switch stream, val bits/char (lower better), best LR per condition\n\n" + markdown_table(an, "val_bpc", ["post_switch_bpc_last2"])
    write_summary(EXP, an, md, variant=args.variant); print(md)
