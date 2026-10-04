"""Step time of the rung-1 v3 model (2 memory layers, d = 128, T = 256, batch 64) for sensing blocks of 8, 16 and 32 tokens,
state and free arms, compiled per-chunk step; median of 20 steps after 5 warm-up steps (compilation excluded)."""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig
from flyprinc.lmtasks import LMTaskConfig, make_batch

dev = "cuda"; task = LMTaskConfig(T=256, keys_per_seq=24, window=256, p_rebind=0.2, p_outlier=0.15, far_fraction=0.0)
for block in (8, 16, 32):
    for arm, lk in (("state", dict(n_traces=2, states=("nconf", "zc"))), ("free", dict(n_traces=2, states=())), ("single_wide", dict(states=(), d_k=73))):
        torch.manual_seed(0)
        lc = LayerConfig(d_model=128, n_heads=4, d_k=lk.pop("d_k", 32), d_v=32, d_c=8, chunk=block, sense_block=block, compile_chunk=True, **lk)
        model = HybridLM(HybridConfig(vocab=task.vocab, d_model=128, n_layers=2, n_heads=4, window=256, layer=lc, attn_every=99, pos="rope", tied=False, emb_std=1.0)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3); rng = np.random.default_rng(0); times = []
        for s in range(25):
            toks, tgt, _, _ = make_batch(task, 64, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
            torch.cuda.synchronize(); t = time.time(); logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
            opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
            if s >= 5: times.append(time.time() - t)
        print(f"block {block:2d} {arm:12s}: {np.median(times):.3f} s per step -> 6,000 steps = {np.median(times) * 6000 / 60:.0f} min", flush=True)
