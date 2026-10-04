"""Rung-1b pilot (exploratory): memory-only models (2 layers, d = 128, one store) on the dense rung-1 generator in its
contradiction-free capacity form (P distinct keys, 2-5 consistent presentations each, 1-2 queries near the key),
dense 32-dimensional keys vs the Kenyon-cell address (fixed random expansion to 512 cells, 7 key dims per cell,
k-WTA 32 on: the E1 ratios), P in {32, 64} at T = 8 P. Does every arm converge, and how do they compare?"""
import sys, time, math, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig
from flyprinc.lmtasks import LMTaskConfig, make_batch

dev = "cuda"
ARMS = {"dense": dict(), "sparse": dict(address="sparse", n_cells=512, k_active=32, fan_in=7)}


def evaluate(model, task, seed=777, batches=8, B=64):
    rng = np.random.default_rng(seed); hits = n = 0; model.eval()
    with torch.no_grad():
        for _ in range(batches):
            toks, tgt, _, _ = make_batch(task, B, rng); pred = model(torch.as_tensor(toks, device=dev)).argmax(-1).cpu().numpy(); m = tgt >= 0
            hits += int((pred[m] == tgt[m]).sum()); n += int(m.sum())
    model.train(); return hits / n


def run(arm, P, seed=0, steps=6000, B=64, lr=1e-3, block=16):
    torch.manual_seed(seed); T = 8 * P
    task = LMTaskConfig(T=T, keys_per_seq=P, window=T, p_rebind=0.0, p_outlier=0.0, far_fraction=0.0)
    lc = LayerConfig(d_model=128, n_heads=4, d_k=32, d_v=32, d_c=8, chunk=block, sense_block=block, compile_chunk=True, states=(), **ARMS[arm])
    model = HybridLM(HybridConfig(vocab=task.vocab, d_model=128, n_layers=2, n_heads=4, window=T, layer=lc, attn_every=99, pos="rope", tied=False, emb_std=1.0)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = np.random.default_rng(seed + 1); accs = []; marks = []; t0 = time.time()
    for s in range(steps):
        toks, tgt, _, _ = make_batch(task, B, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
        logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        accs.append((logits[m].argmax(-1) == y[m]).float().mean().item())
        if (s + 1) % 1000 == 0: marks.append(f"{s + 1}:{np.mean(accs[-100:]):.2f}")
    acc = evaluate(model, task)
    print(f"{arm:7s} P={P:3d} T={T:4d} seed {seed} | train {' '.join(marks)} | held-out {acc:.3f} | peak {torch.cuda.max_memory_reserved() / 2**30:.1f} GB ({time.time() - t0:.0f}s)", flush=True)
    return acc


res = {}
for P in (32, 64):
    for arm in ARMS:
        torch.cuda.reset_peak_memory_stats(); res[f"{arm}|P{P}"] = run(arm, P)
print(json.dumps(res))
