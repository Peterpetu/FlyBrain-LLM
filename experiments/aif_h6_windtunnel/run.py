"""AIF-H6: Bayesian wind tunnel — KL between the model's predictive and the exact HMM posterior predictive."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyai.model import FlyAI, FlyAIConfig
from flyai.primitives import param_groups, after_optimizer_step
from flyai.tasks import HMMWindTunnel, predictive_kl
from flybrain.models import SeqModel, ModelConfig
from flybrain.harness import run_grid, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
V, T, B, SKIP = 16, 64, 32, 8


class Wrapped(torch.nn.Module):
    """SeqModel baselines predict token t+1 at position t; shift so logits[:, t] predicts x[:, t]."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        lg = self.m(x[:, :-1]); pad = torch.full_like(lg[:, :1], 0.0)
        return torch.cat([pad, lg], 1), None


def build(cfg, seed):
    if cfg["mixer"] == "flyai":
        return FlyAI(FlyAIConfig(vocab=V, d=64, n=128, n_kc=512, K_max=cfg["K_max"], adaptive_K=False, use_attractor=cfg.get("use_attractor", True), seed=seed))   # PREREG: exactly K steps
    if cfg["mixer"] == "gdn":
        return Wrapped(SeqModel(ModelConfig(vocab=V, d_model=64, n_layers=1, mixer="fastweight", mixer_kwargs=dict(n_heads=2, d_value=16, key_mode="dense", d_key=32, write="delta", gate="learned", decay="learned", conv_kernel=4))))
    return Wrapped(SeqModel(ModelConfig(vocab=V, d_model=64, n_layers=2, mixer="attention", mixer_kwargs=dict(n_heads=2))))


def kl_eval(model, batches):
    model.eval(); out = []
    with torch.no_grad():
        for x, post in batches:
            logits, _ = model(x); out.append(predictive_kl(logits[:, SKIP:], post[:, SKIP:]).mean().item())
    model.train(); return float(np.mean(out))


def precision_tracking(model, batches):
    """Spearman correlation between the model's inferred level-1 precision (per token) and the true
    posterior predictive's concentration (negative entropy). Only defined for FlyAI models."""
    if not isinstance(model, FlyAI): return float("nan")
    from scipy.stats import spearmanr
    model.eval(); pis, conc = [], []
    with torch.no_grad():
        for x, post in batches[:2]:
            st = model.init_state(x.shape[0], model.emb.weight)
            for t in range(x.shape[1]):
                if t >= SKIP:                                                   # belief precision ENTERING step t (before o_t is observed),
                    pis.append(model.prec.ratio(st["pi1"]).mean(-1))            # aligned with the predictive for o_t
                    conc.append((post[:, t] * torch.log(post[:, t] + 1e-9)).sum(-1))
                _, st, _ = model.step(st, x[:, t])
    model.train(); r = spearmanr(torch.cat(pis).numpy(), torch.cat(conc).numpy()).correlation
    return float(r)


def train_fn(cfg, seed, run_dir):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); ev = np.random.default_rng(1234); tun = HMMWindTunnel(6, V)
    fixed = [tun.sample(B, T, ev) for _ in range(8)]
    model = build(cfg, seed)
    opt = torch.optim.AdamW(param_groups(model, 0.01), lr=cfg["lr"], betas=(0.9, 0.95))
    steps, warm = cfg["steps"], 20; hist = []; t0 = time.time(); log = open(run_dir / "log.jsonl", "w"); diverged = False
    for s in range(steps):
        lr = cfg["lr"] * ((s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
        for g in opt.param_groups: g["lr"] = lr
        x, _ = tun.sample(B, T, rng); logits, _ = model(x)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, V), x[:, 1:].reshape(-1))
        if not torch.isfinite(loss): diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); after_optimizer_step(model)
        if (s + 1) % 100 == 0 or s + 1 == steps:
            kl = kl_eval(model, fixed); hist.append(kl)
            log.write(json.dumps({"step": s + 1, "loss": float(loss), "kl_bits": kl, "t": time.time() - t0}) + "\n"); log.flush()
    log.close()
    return {"diverged": diverged, "kl_bits_last2": float(np.mean(hist[-2:])) if (hist and not diverged) else float("nan"), "kl_bits_hist": hist,
            "precision_tracking_spearman": precision_tracking(model, fixed) if not diverged else float("nan"),
            "K_mean": model.last_diag["K_mean"] if isinstance(model, FlyAI) else float("nan"), "params": sum(p.numel() for p in model.parameters())}


CONDS = {"flyai_K2": dict(mixer="flyai", K_max=2), "flyai_K0": dict(mixer="flyai", K_max=0), "flyai_K2_noattr": dict(mixer="flyai", K_max=2, use_attractor=False),
         "gdn": dict(mixer="gdn"), "attention": dict(mixer="attention")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--variant", default=None, help="e.g. replication: separate runs_<variant>/ and SUMMARY_<variant>.md"); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {k: dict(v, steps=args.steps) for k, v in CONDS.items() if not args.only or k in args.only}
    res = run_grid(EXP, C, [1e-3, 3e-3, 1e-2], args.seeds, train_fn, variant=args.variant)
    an = analyse(res, "kl_bits_last2", "flyai_K2", ["flyai_K0", "flyai_K2_noattr", "gdn", "attention"], higher_is_better=False, extra_metrics=["precision_tracking_spearman", "K_mean", "params"])
    md = "## AIF-H6: KL(true HMM predictive || model) in bits, positions 8-64 (lower better), best LR per condition\n\n" + markdown_table(an, "kl_bits", ["precision_tracking_spearman", "K_mean"])
    write_summary(EXP, an, md, variant=args.variant); print(md)
