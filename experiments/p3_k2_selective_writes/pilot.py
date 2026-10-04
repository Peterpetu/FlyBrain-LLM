"""EXPLORATORY pilot (one seed, no verdict): does the sparse address's capacity gain depend on selective writing?

First-principles reading of rung 1b against the language prototypes: the Kenyon-cell address raised capacity on a task whose
memory layer learns to write a few key-value pairs strongly among filler (and the fly writes only on reinforcement), while a
language model writes every token (block A's gate statistics). If the gain depends on selective writing, forcing every token
to be written at full strength should shrink it on the same synthetic task. Four arms on rung 1b's capacity task at P = 64,
key width 16 (the regime where the dense store is the bottleneck): dense and sparse keys, each with the learned input-only
write gate (Gated DeltaNet) and with a constant gate that writes every token at strength 1 (erase 1, forget drive 0.05).
Usage: python experiments/p3_k2_selective_writes/pilot.py [--steps 6000]"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from flyprinc.hybrid import HybridConfig, HybridLM, LayerConfig  # noqa: E402
from flyprinc.lmtasks import LMTaskConfig, BatchStream, make_batch  # noqa: E402

P, DK, CELLS, K, FAN_IN = 64, 16, 256, 16, 4
ARMS = {"dense_learned_gate": dict(), "sparse_learned_gate": dict(address="sparse", n_cells=CELLS, k_active=K, fan_in=FAN_IN),
        "dense_write_all": dict(const_gate=(1.0, 1.0, 0.05)), "sparse_write_all": dict(address="sparse", n_cells=CELLS, k_active=K, fan_in=FAN_IN, const_gate=(1.0, 1.0, 0.05))}


def run(arm, steps, seed, B, device, workers):
    torch.manual_seed(seed); T = 8 * P; task = LMTaskConfig(T=T, keys_per_seq=P, window=T, p_rebind=0.0, p_outlier=0.0, far_fraction=0.0)
    lc = LayerConfig(d_model=128, n_heads=4, d_k=DK, d_v=32, d_c=8, states=(), x_only_gate=True, tie_gamma=True, chunk=64, compile_chunk=True, **ARMS[arm])
    model = HybridLM(HybridConfig(vocab=task.vocab, d_model=128, n_layers=2, n_heads=4, window=T, layer=lc, attn_every=99, pos="rope", tied=False, emb_std=1.0)).to(device)
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    stream = iter(BatchStream(task, B, seed + 1, steps, workers=workers)); accs = []; t0 = time.time(); marks = []
    for s in range(steps):
        toks, tgt, _, _ = next(stream); x, y = torch.as_tensor(toks, device=device), torch.as_tensor(tgt, device=device)
        logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        accs.append((logits[m].argmax(-1) == y[m]).float().mean().item())
        if (s + 1) % 1000 == 0: marks.append(f"{s + 1}:{np.mean(accs[-100:]):.2f}")
    stream.close(); model.eval(); rng = np.random.default_rng(777); hits = n = 0
    with torch.no_grad():
        for _ in range(8):
            toks, tgt, _, _ = make_batch(task, B, rng); pred = model(torch.as_tensor(toks, device=device)).argmax(-1).cpu().numpy(); msk = tgt >= 0
            hits += int((pred[msk] == tgt[msk]).sum()); n += int(msk.sum())
    return {"arm": arm, "seed": seed, "held_out_acc": hits / n, "train_curve": marks, "train_acc_last": float(np.mean(accs[-100:])), "wall_min": (time.time() - t0) / 60}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--arms", nargs="*", default=list(ARMS)); a = ap.parse_args()
    out = Path(__file__).resolve().parent / "pilot_results.jsonl"
    for arm in a.arms:
        if (Path(__file__).resolve().parent / "STOP").exists(): print("STOP file found"); break
        r = run(arm, a.steps, a.seed, a.B, "cuda", a.workers); print(json.dumps(r), flush=True)
        with open(out, "a") as f: f.write(json.dumps(r) + "\n")
