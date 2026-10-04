"""Does a length curriculum make the rung-1 task learnable within budget? The rung-1 generator (with re-bindings and outliers),
rotary positions, untied head, embedding std 1, lr 1e-3; stages of growing length and key count, 1,000 steps each.
Reports query accuracy at the end of each stage (last 100 training steps) and, at the final stage, on far queries only."""
import sys, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig
from flyprinc.lmtasks import LMTaskConfig, make_batch

dev = "cuda"
STAGES = [(128, 4), (256, 8), (512, 16), (1024, 32), (2048, 48)]


def run(name, arch, steps_per_stage=1000, B=16, lr=1e-3, compile_chunk=True):
    torch.manual_seed(0)
    lc = LayerConfig(d_model=256, n_heads=4, d_k=64, d_v=64, d_c=16, chunk=64, sense_block=64, n_traces=2, states=("nconf", "zc"), compile_chunk=compile_chunk)
    cfg = HybridConfig(vocab=1282, d_model=256, n_layers=4, n_heads=4, window=512, layer=lc, attn_every=1 if arch == "attn_full" else 4,
                       pos="rope", tied=False, emb_std=1.0)
    model = HybridLM(cfg).to(dev)
    total = steps_per_stage * len(STAGES)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    rng = np.random.default_rng(1); out = []; t0 = time.time()
    for T, keys in STAGES:
        if arch == "attn_full":
            for m in model.mix: m.w = T                                               # full context at every stage
        task = LMTaskConfig(T=T, keys_per_seq=keys, window=min(512, T // 2), p_rebind=0.2, p_outlier=0.15, far_fraction=0.5)
        accs = []; far_hits = far_n = 0
        for s in range(steps_per_stage):
            toks, tgt, _, far = make_batch(task, B, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
            logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            pred = logits.argmax(-1); accs.append((pred[m] == y[m]).float().mean().item())
            if s >= steps_per_stage - 100:
                fm = m & torch.as_tensor(far, device=dev); far_hits += int((pred[fm] == y[fm]).sum()); far_n += int(fm.sum())
        out.append(f"T{T}/{keys}k: {np.mean(accs[-100:]):.2f} (far {far_hits / max(far_n, 1):.2f})")
        print(f"  {name} stage T={T} keys={keys}: query acc {np.mean(accs[-100:]):.3f}, far {far_hits / max(far_n, 1):.3f}  [{time.time() - t0:.0f}s]", flush=True)
    print(f"{name} -> " + " | ".join(out), flush=True)


which = sys.argv[1] if len(sys.argv) > 1 else "both"
if which in ("both", "attn"): run("attention only (full context), rope", "attn_full", compile_chunk=False)
if which in ("both", "hybrid"): run("hybrid 3 mem + 1 SWA, two stores + states, rope", "hybrid")
