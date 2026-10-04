"""AIF-H4: Lyapunov-regulated core gain vs fixed gains on long-range retention."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyai.model import FlyAI, FlyAIConfig
from flyai.primitives import param_groups, after_optimizer_step
from flyai.tasks import retention_batch, same_index_logits
from flybrain.harness import run_grid, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
VOCAB, T, B, NI = 48, 64, 32, 4


def acc(model, batches):
    model.eval(); out = []
    with torch.no_grad():
        for x, y in batches:
            logits, _ = same_index_logits(model, x, VOCAB - 1); m = y != -100; nk = VOCAB // 3
            pred = logits[..., nk:2 * nk].argmax(-1) + nk                           # answers restricted to the value vocabulary (chance = 1/16 exactly)
            out.append((pred[m] == y[m]).float().mean().item())
    model.train(); return float(np.mean(out))


def train_fn(cfg, seed, run_dir):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); ev = np.random.default_rng(1234)
    kw = {k: v for k, v in cfg.items() if k in FlyAIConfig.__dataclass_fields__ and k != 'seed'}
    model = FlyAI(FlyAIConfig(vocab=VOCAB, **kw, seed=seed))
    if cfg.get("fixed_gain") is not None:
        model.core.set_gain(cfg["fixed_gain"])
    fixed = {L: [retention_batch(B, L, VOCAB, ev, NI) for _ in range(4 if L == T else 2)] for L in (T, 2 * T, 4 * T)}
    opt = torch.optim.AdamW(param_groups(model, 0.01), lr=cfg["lr"], betas=(0.9, 0.95))
    steps, warm = cfg["steps"], 20; hist = []; t0 = time.time(); log = open(run_dir / "log.jsonl", "w"); diverged = False
    for s in range(steps):
        lr = cfg["lr"] * ((s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
        for g in opt.param_groups: g["lr"] = lr
        x, y = retention_batch(B, T, VOCAB, rng, NI)
        logits, _ = same_index_logits(model, x, VOCAB - 1)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss): diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); after_optimizer_step(model)
        if (s + 1) % 100 == 0 or s + 1 == steps:
            a = {L: acc(model, fixed[L]) for L in fixed}; hist.append([a[T], a[2 * T], a[4 * T]])
            log.write(json.dumps({"step": s + 1, "loss": float(loss), "acc_T64": a[T], "acc_T128": a[2 * T], "acc_T256": a[4 * T], **model.last_diag, "t": time.time() - t0}) + "\n"); log.flush()
    log.close(); h = np.array(hist) if hist else np.full((1, 3), np.nan)
    if diverged: h = np.full_like(h, np.nan)
    return {"diverged": diverged, "acc_T64_last2": float(h[-2:, 0].mean()), "acc_T128_last2": float(h[-2:, 1].mean()), "acc_T256_last2": float(h[-2:, 2].mean()),
            "lyapunov": model.last_diag["lyapunov"], "gain": model.last_diag["gain"], "params": sum(p.numel() for p in model.parameters())}


CONDS = {"regulated": dict(regulate_core=True), "fixed_0.5": dict(regulate_core=False, fixed_gain=0.5),
         "fixed_1.0": dict(regulate_core=False, fixed_gain=1.0), "fixed_2.0": dict(regulate_core=False, fixed_gain=2.0)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--variant", default=None, help="e.g. replication: separate runs_<variant>/ and SUMMARY_<variant>.md"); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {k: dict(v, steps=args.steps, K_max=1, adaptive_K=False, n=128, use_memory=False, d=64) for k, v in CONDS.items() if not args.only or k in args.only}   # PREREG: K = 1 exactly
    LRS = [1e-3, 3e-3, 1e-2]
    res = run_grid(EXP, C, LRS, args.seeds, train_fn, variant=args.variant)
    controls = ["fixed_0.5", "fixed_1.0", "fixed_2.0"]
    if "regulated" in C and not args.only:
        # PREREG post-hoc arm: the gain fixed at the value the regulated core converged to (mean final base gain at its best LR)
        an0 = analyse(res, "acc_T64_last2", "regulated", controls)
        lr_reg = an0["table"]["regulated"]["lr"]
        g_conv = round(float(np.nanmean([r["gain"] for r in res if r["condition"] == "regulated" and r["lr"] == lr_reg])), 2)
        C2 = {"fixed_conv": dict(C["regulated"], regulate_core=False, fixed_gain=g_conv)}
        res += run_grid(EXP, C2, LRS, args.seeds, train_fn, variant=args.variant); controls.append("fixed_conv")
    an = analyse(res, "acc_T64_last2", "regulated", controls, extra_metrics=["acc_T128_last2", "acc_T256_last2", "lyapunov", "gain"])
    md = "## AIF-H4: long-range retention accuracy at train length 64 (chance 0.0625), best LR per condition\n\n" + markdown_table(an, "acc_T64", ["acc_T128_last2", "acc_T256_last2", "lyapunov", "gain"])
    write_summary(EXP, an, md, variant=args.variant); print(md)
