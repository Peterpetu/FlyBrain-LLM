"""AIF-H2: inferred-precision gate vs learned / constant gate on non-stationary MQAR."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyai.model import FlyAI, FlyAIConfig
from flyai.primitives import param_groups, after_optimizer_step
from flyai.tasks import nonstationary_mqar, same_index_logits
from flybrain.harness import run_grid, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
VOCAB, T, NPAIRS, B, REBIND = 48, 96, 8, 32, 4


def build(cfg):
    kw = {k: v for k, v in cfg.items() if k in FlyAIConfig.__dataclass_fields__}
    return FlyAI(FlyAIConfig(vocab=VOCAB, **kw))


def accs(model, batches):
    model.eval(); pre, post = [], []
    with torch.no_grad():
        for x, y in batches:
            logits, _ = same_index_logits(model, x, VOCAB - 1); nk = VOCAB // 3
            pred = logits[..., nk:2 * nk].argmax(-1) + nk; half = x.shape[1] // 2          # answers restricted to the value vocabulary (chance = 1/16 exactly)
            m_pre = (y != -100) & (torch.arange(x.shape[1])[None] < half); m_post = (y != -100) & (torch.arange(x.shape[1])[None] >= half)
            pre.append((pred[m_pre] == y[m_pre]).float().mean().item()); post.append((pred[m_post] == y[m_post]).float().mean().item())
    model.train(); return float(np.mean(pre)), float(np.mean(post))


def train_fn(cfg, seed, run_dir):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); ev = np.random.default_rng(1234)
    model = build(dict(cfg, seed=seed))
    fixed = [nonstationary_mqar(B, T, NPAIRS, VOCAB, ev, REBIND) for _ in range(4)]
    fixed_long = [nonstationary_mqar(B, 2 * T, NPAIRS, VOCAB, ev, REBIND) for _ in range(2)]
    opt = torch.optim.AdamW(param_groups(model, 0.01), lr=cfg["lr"], betas=(0.9, 0.95))
    steps, warm = cfg["steps"], 20; hist = []; t0 = time.time(); log = open(run_dir / "log.jsonl", "w"); diverged = False
    for s in range(steps):
        lr = cfg["lr"] * ((s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
        for g in opt.param_groups: g["lr"] = lr
        x, y = nonstationary_mqar(B, T, NPAIRS, VOCAB, rng, REBIND)
        logits, _ = same_index_logits(model, x, VOCAB - 1)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss): diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); after_optimizer_step(model)
        if (s + 1) % 100 == 0 or s + 1 == steps:
            pre, post = accs(model, fixed); pre_l, post_l = accs(model, fixed_long); hist.append((pre, post, pre_l, post_l))
            log.write(json.dumps({"step": s + 1, "loss": float(loss), "acc_pre": pre, "acc_post": post, "acc_pre_T192": pre_l, "acc_post_T192": post_l, **model.last_diag, "t": time.time() - t0}) + "\n"); log.flush()
    log.close()
    h = np.array(hist) if hist else np.full((1, 4), np.nan)
    if diverged: h = np.full_like(h, np.nan)                                     # a diverged run never enters a table with a finite metric
    return {"diverged": diverged, "acc_post_last2": float(h[-2:, 1].mean()), "acc_pre_last2": float(h[-2:, 0].mean()),
            "acc_post_T192_last2": float(h[-2:, 3].mean()), "acc_pre_T192_last2": float(h[-2:, 2].mean()),
            "params": sum(p.numel() for p in model.parameters())}


CONDS = {"precision": dict(memory_gate="precision"), "surprise": dict(memory_gate="surprise"), "learned": dict(memory_gate="learned"),
         "const": dict(memory_gate="const"), "no_memory": dict(use_memory=False)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--variant", default=None, help="e.g. replication: separate runs_<variant>/ and SUMMARY_<variant>.md"); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {k: dict(v, steps=args.steps, K_max=1, adaptive_K=False, n=128, n_kc=512, d=64) for k, v in CONDS.items() if not args.only or k in args.only}   # PREREG: K = 1 exactly
    res = run_grid(EXP, C, [1e-3, 3e-3, 1e-2], args.seeds, train_fn, variant=args.variant)
    an = analyse(res, "acc_post_last2", "precision", ["surprise", "learned", "const", "no_memory"], extra_metrics=["acc_pre_last2", "acc_post_T192_last2", "acc_pre_T192_last2", "params"])
    an["pre_rebind"] = analyse(res, "acc_pre_last2", "precision", ["surprise", "learned", "const", "no_memory"])["comparisons"]   # PREREG guard: not worse pre-rebind beyond seed spread
    md = "## AIF-H2: post-rebind recall accuracy (chance 0.0625), best LR per condition\n\n" + markdown_table(an, "acc_post", ["acc_pre_last2", "acc_post_T192_last2", "acc_pre_T192_last2"])
    write_summary(EXP, an, md, variant=args.variant); print(md)
