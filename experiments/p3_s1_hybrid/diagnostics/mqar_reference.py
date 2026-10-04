"""Bug hunt: a standard off-the-shelf transformer (PyTorch TransformerEncoderLayer, learned absolute positions,
causal mask) vs HybridLM's attention-only model on the SAME textbook MQAR batches and training loop.
If the reference solves MQAR and HybridLM does not, the bug is in HybridLM's attention path."""
import sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig

dev = "cuda"; VOCAB = 1282; NK = 512; NV = 512


def mqar_batch(B, n_pairs, rng):
    T = 4 * n_pairs; toks = np.zeros((B, T), np.int64); tgt = np.full((B, T), -1, np.int64)
    for b in range(B):
        keys = rng.choice(NK, n_pairs, replace=False); vals = NK + rng.integers(0, NV, n_pairs)
        toks[b, 0:2 * n_pairs:2] = keys; toks[b, 1:2 * n_pairs:2] = vals
        for j, i in enumerate(rng.permutation(n_pairs)):
            p = 2 * n_pairs + 2 * j; toks[b, p] = keys[i]; toks[b, p + 1] = vals[i]; tgt[b, p] = vals[i]
    return toks, tgt


class RefGPT(nn.Module):
    def __init__(self, d, layers, heads, T):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d); self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True) for _ in range(layers)])
        self.nf = nn.LayerNorm(d); self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, x):
        T = x.shape[1]; h = self.tok(x) + self.pos(torch.arange(T, device=x.device))[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        for blk in self.blocks: h = blk(h, src_mask=mask, is_causal=True)
        return self.head(self.nf(h))


def train(name, model, lr, steps, n_pairs=16, B=64, log_every=500):
    torch.manual_seed(0); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = np.random.default_rng(1); accs = []; out = []; t0 = time.time()
    for s in range(steps):
        toks, tgt = mqar_batch(B, n_pairs, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
        logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        accs.append((logits[m].argmax(-1) == y[m]).float().mean().item())
        if (s + 1) % log_every == 0: out.append(f"{s + 1}:{np.mean(accs[-100:]):.2f}")
    print(f"{name} -> " + " ".join(out) + f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":                                                      # mqar_ablate.py imports this module's batch generator
    torch.manual_seed(0); train("reference GPT (learned pos, d=128, L=2)", RefGPT(128, 2, 4, 64).to(dev), 1e-3, 3000)
    torch.manual_seed(0); lc = LayerConfig(d_model=128, n_heads=4, d_k=32, d_v=32, d_c=8)
    train("HybridLM attention-only (RoPE, d=128, L=2)", HybridLM(HybridConfig(vocab=VOCAB, d_model=128, n_layers=2, n_heads=4, window=64, layer=lc, attn_every=1)).to(dev), 1e-3, 3000)
