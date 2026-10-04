"""Reliability pilot for the redesigned rung 1: memory-only models (2 layers, d = 128) on the dense rung-1 task
(T = 256, 24 keys, re-bindings 0.2, outliers 0.15, queries near their key), 6,000 steps, three seeds per arm.
Does every seed converge (so the arms can be compared at convergence), and how do the arms compare on the
decisive strata (re-bound keys, after-outlier keys) and overall? Exploratory: not a pre-registered test."""
import sys, time, math, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig
from flyprinc.lmtasks import LMTaskConfig, make_batch, STRATA

dev = "cuda"; task = LMTaskConfig(T=256, keys_per_seq=24, window=256, p_rebind=0.2, p_outlier=0.15, far_fraction=0.0)
ARMS = {"two stores + states": dict(n_traces=2, states=("nconf", "zc")), "two stores, free": dict(n_traces=2, states=())}


def evaluate(model, seed=777, batches=8, B=64):
    rng = np.random.default_rng(seed); hits = {}; cnt = {}; model.eval()
    with torch.no_grad():
        for _ in range(batches):
            toks, tgt, strat, _ = make_batch(task, B, rng); x = torch.as_tensor(toks, device=dev)
            ok = (model(x).argmax(-1).cpu().numpy() == tgt); m = tgt >= 0
            for i, s in enumerate(STRATA):
                sel = m & (strat == i); hits[s] = hits.get(s, 0) + int(ok[sel].sum()); cnt[s] = cnt.get(s, 0) + int(sel.sum())
            hits["all"] = hits.get("all", 0) + int(ok[m].sum()); cnt["all"] = cnt.get("all", 0) + int(m.sum())
    model.train(); acc = {k: hits[k] / cnt[k] for k in hits}; acc["decisive"] = (acc["rebound"] + acc["after_outlier"]) / 2; return acc


def run(arm, seed, steps=6000, B=64, lr=1e-3):
    torch.manual_seed(seed)
    lc = LayerConfig(d_model=128, n_heads=4, d_k=32, d_v=32, d_c=8, chunk=32, sense_block=32, compile_chunk=True, **ARMS[arm])
    model = HybridLM(HybridConfig(vocab=task.vocab, d_model=128, n_layers=2, n_heads=4, window=256, layer=lc, attn_every=99, pos="rope", tied=False, emb_std=1.0)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = np.random.default_rng(seed + 1); accs = []; marks = []; t0 = time.time()
    for s in range(steps):
        toks, tgt, _, _ = make_batch(task, B, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
        logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        accs.append((logits[m].argmax(-1) == y[m]).float().mean().item())
        if (s + 1) % 1000 == 0: marks.append(f"{s + 1}:{np.mean(accs[-100:]):.2f}")
    acc = evaluate(model)
    print(f"{arm:22s} seed {seed} | train {' '.join(marks)} | held-out: all {acc['all']:.3f}, decisive {acc['decisive']:.3f} (rebound {acc['rebound']:.3f}, after_outlier {acc['after_outlier']:.3f}), single {acc['single']:.3f}, confirmed {acc['confirmed']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return acc


res = {}
for seed in (0, 1, 2):
    for arm in ARMS:
        res[f"{arm}|{seed}"] = run(arm, seed)
print(json.dumps(res))
