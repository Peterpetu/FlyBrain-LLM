"""Controlled learnability sweep: which model types learn which dense recall task within 3,000 steps?
Fixed: d = 128, rotary positions (attention layers only), untied head, embedding std 1, lr 1e-3, batch 64, 3,000 steps, seed 0.
Tasks: textbook MQAR (T = 128, 32 pairs) and a dense rung-1 task (T = 256, 24 keys, repeated presentations, re-bindings 0.2,
outliers 0.15, queries near their key; the generator then leaves little filler). The hybrid's attention window is 32 tokens,
so most answers must come from the memory layers. Accuracy on query answers, mean of the last 100 steps."""
import sys, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig
from flyprinc.lmtasks import LMTaskConfig, make_batch

dev = "cuda"; VOCAB = 1282


def mqar_batch(B, n_pairs, rng):
    T = 4 * n_pairs; toks = np.zeros((B, T), np.int64); tgt = np.full((B, T), -1, np.int64)
    for b in range(B):
        keys = rng.choice(512, n_pairs, replace=False); vals = 512 + rng.integers(0, 512, n_pairs)
        toks[b, 0:2 * n_pairs:2] = keys; toks[b, 1:2 * n_pairs:2] = vals
        for j, i in enumerate(rng.permutation(n_pairs)):
            p = 2 * n_pairs + 2 * j; toks[b, p] = keys[i]; toks[b, p + 1] = vals[i]; tgt[b, p] = vals[i]
    return toks, tgt


dense_task = LMTaskConfig(T=256, keys_per_seq=24, window=256, p_rebind=0.2, p_outlier=0.15, far_fraction=0.0)
TASKS = {"mqar128": (128, lambda B, r: mqar_batch(B, 32, r)),
         "dense_rung1": (256, lambda B, r: make_batch(dense_task, B, r)[:2])}
MODELS = {
    "attention only (2 layers)": dict(attn_every=1, n_layers=2, lk=dict(states=())),
    "memory only, one store (2 layers)": dict(attn_every=99, n_layers=2, lk=dict(states=())),
    "memory only, two stores + states (2 layers)": dict(attn_every=99, n_layers=2, lk=dict(states=("nconf", "zc"), n_traces=2)),
    "hybrid 3 mem + 1 attn(w32), two stores free": dict(attn_every=4, n_layers=4, window=32, lk=dict(states=(), n_traces=2)),
    "hybrid 3 mem + 1 attn(w32), two stores + states": dict(attn_every=4, n_layers=4, window=32, lk=dict(states=("nconf", "zc"), n_traces=2)),
}


def run(task_name, model_name, steps=3000, B=64, lr=1e-3):
    T, batch_fn = TASKS[task_name]; spec = MODELS[model_name]; torch.manual_seed(0)
    lc = LayerConfig(d_model=128, n_heads=4, d_k=32, d_v=32, d_c=8, chunk=32, sense_block=32, **spec["lk"])
    model = HybridLM(HybridConfig(vocab=VOCAB, d_model=128, n_layers=spec["n_layers"], n_heads=4, window=spec.get("window", T), layer=lc,
                                  attn_every=spec["attn_every"], pos="rope", tied=False, emb_std=1.0)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = np.random.default_rng(1); accs = []; marks = []; t0 = time.time()
    for s in range(steps):
        toks, tgt = batch_fn(B, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
        logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        accs.append((logits[m].argmax(-1) == y[m]).float().mean().item())
        if (s + 1) % 1000 == 0: marks.append(f"{s + 1}:{np.mean(accs[-100:]):.2f}")
    print(f"{task_name:12s} | {model_name:48s} -> {' '.join(marks)}  ({time.time() - t0:.0f}s)", flush=True)


for task_name in TASKS:
    for model_name in MODELS:
        run(task_name, model_name)
