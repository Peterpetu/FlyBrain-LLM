"""Why does the corrected configuration learn textbook MQAR (T = 64) but not the rung-1 generator (T >= 256)?
Separate length from format, and test the position scheme. Attention-only, 2 layers, d = 128, lr 1e-3, 3,000 steps.
Accuracy on query answers (chance 1/512; the in-context guessing level is 1/keys)."""
import sys, time, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import flyprinc.hybrid as hy
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


def run(name, batch_fn, T, pos, tied, emb_std, steps=3000, B=32, lr=1e-3):
    torch.manual_seed(0)
    lc = LayerConfig(d_model=128, n_heads=4, d_k=32, d_v=32, d_c=8)
    cfg = HybridConfig(vocab=VOCAB, d_model=128, n_layers=2, n_heads=4, window=T, layer=lc, attn_every=1, pos="learned" if pos == "learned" else "rope", max_len=T, tied=tied, emb_std=emb_std)
    model = HybridLM(cfg).to(dev)
    if pos == "none":
        for m in model.mix: m.rope = False                                              # no positional information at all (NoPE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = np.random.default_rng(1); accs = []; out = []; t0 = time.time()
    for s in range(steps):
        toks, tgt = batch_fn(B, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
        logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        accs.append((logits[m].argmax(-1) == y[m]).float().mean().item())
        if (s + 1) % 500 == 0: out.append(f"{s + 1}:{np.mean(accs[-100:]):.3f}")
    print(f"{name:62s} -> " + " ".join(out) + f"  ({time.time() - t0:.0f}s)", flush=True)


clean = LMTaskConfig(T=256, keys_per_seq=8, window=256, p_rebind=0.0, p_outlier=0.0, far_fraction=0.0)
def lm256(B, r):
    t, g, _, _ = make_batch(clean, B, r); return t, g

for pos, tied, std, label in (("rope", True, 0.02, "rope, tied, std 0.02 (original)"), ("learned", False, 1.0, "learned pos, untied, std 1 (corrected)"),
                              ("rope", False, 1.0, "rope, untied, std 1"), ("none", False, 1.0, "no positions, untied, std 1")):
    run(f"lmtasks T=256 8 keys clean | {label}", lm256, 256, pos, tied, std)
run("MQAR textbook T=256 64 pairs | learned pos, untied, std 1", lambda B, r: mqar_batch(B, 64, r), 256, "learned", False, 1.0)
run("MQAR textbook T=256 64 pairs | rope, untied, std 1", lambda B, r: mqar_batch(B, 64, r), 256, "rope", False, 1.0)
