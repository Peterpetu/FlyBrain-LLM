"""One process trains many layer configurations in turn, as a grid does: every configuration must stay compiled.
Before the fix torch.compile kept 8 graphs of the chunk function and then ran it uncompiled (about 5 times slower) with only
a warning. Prints each configuration's median step time after warm-up; the warning, if it appears, goes to stderr."""
import sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig
from flyprinc.lmtasks import LMTaskConfig, make_batch

if __name__ == "__main__":
    dev = "cuda"; task = LMTaskConfig(T=256, keys_per_seq=24, window=256, p_rebind=0.2, p_outlier=0.15, far_fraction=0.0)
    configs = [dict(n_traces=2, states=("nconf", "zc")), dict(n_traces=2, states=()), dict(states=(), d_k=73), dict(states=()),
               dict(n_traces=2, states=("a",), a_floor=False), dict(n_traces=2, states=(), const_gate=((0.2, 0.2, 0.05), (0.5, 0.5, 0.05))),
               dict(states=("nconf", "zc")), dict(n_traces=2, states=("nconf", "zc"), feature_path=False), dict(n_traces=2, states=("nconf", "zc"), slot_path=False),
               dict(n_traces=2, states=(), x_only_gate=True), dict(states=(), address="sparse", n_cells=256, k_active=16, fan_in=4, d_k=16),
               dict(states=(), address="product", n_cells=256, k_active=16, pk_n1=16, d_k=16)]
    rng = np.random.default_rng(0)
    for i, extra in enumerate(configs):
        torch.manual_seed(0); d_k = extra.pop("d_k", 32)
        lc = LayerConfig(d_model=128, n_heads=4, d_k=d_k, d_v=32, d_c=8, chunk=8, sense_block=8, compile_chunk=True, **extra)
        model = HybridLM(HybridConfig(vocab=task.vocab, d_model=128, n_layers=2, n_heads=4, window=256, layer=lc, attn_every=99, pos="rope", tied=False, emb_std=1.0)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3); times = []
        for s in range(8):
            toks, tgt, _, _ = make_batch(task, 64, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
            torch.cuda.synchronize(); t = time.time(); logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
            opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize(); times.append(time.time() - t)
        with torch.no_grad(): model(x)                                        # evaluation without gradients: another graph of the same function
        print(f"config {i:2d}: median step {np.median(times[3:]):.3f} s", flush=True)
