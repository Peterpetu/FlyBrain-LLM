"""Which difference between HybridLM and the reference GPT blocks MQAR learning? Toggle one at a time:
rotary positions vs learned absolute positions, tied vs untied output head, embedding init std 0.02 vs 1."""
import sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
import flyprinc.hybrid as hy
from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from mqar_reference import mqar_batch, VOCAB, dev

_rope_orig = hy._rope


class Variant(HybridLM):
    def __init__(self, c, abs_pos=False, untied=False, emb_std=0.02, T=64):
        super().__init__(c)
        nn.init.normal_(self.emb.weight, std=emb_std)
        self.abs_pos = nn.Embedding(T, c.d_model) if abs_pos else None
        self.head = nn.Linear(c.d_model, c.vocab, bias=False) if untied else None

    def forward(self, tokens):
        x = self.emb(tokens)
        if self.abs_pos is not None: x = x + self.abs_pos(torch.arange(tokens.shape[1], device=tokens.device))[None]
        for i, m in enumerate(self.mix):
            x = x + m(self.n1[i](x)); x = x + self.mlp[i](self.n2[i](x))
        h = self.nf(x)
        return self.head(h) if self.head is not None else h @ self.emb.weight.t()


def train(name, model, lr=1e-3, steps=3000, B=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = np.random.default_rng(1); accs = []; out = []; t0 = time.time()
    for s in range(steps):
        toks, tgt = mqar_batch(B, 16, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
        logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        accs.append((logits[m].argmax(-1) == y[m]).float().mean().item())
        if (s + 1) % 500 == 0: out.append(f"{s + 1}:{np.mean(accs[-100:]):.2f}")
    print(f"{name:55s} -> " + " ".join(out) + f"  ({time.time() - t0:.0f}s)", flush=True)


def make(rope=True, **kw):
    hy._rope = _rope_orig if rope else (lambda x, pos: x)
    torch.manual_seed(0); lc = LayerConfig(d_model=128, n_heads=4, d_k=32, d_v=32, d_c=8)
    return Variant(HybridConfig(vocab=VOCAB, d_model=128, n_layers=2, n_heads=4, window=64, layer=lc, attn_every=1), **kw).to(dev)


train("baseline (RoPE, tied, emb std 0.02)", make())
train("untied head", make(untied=True))
train("emb std 1.0", make(emb_std=1.0))
train("untied head + emb std 1.0", make(untied=True, emb_std=1.0))
train("learned abs pos, no RoPE", make(rope=False, abs_pos=True))
train("learned abs pos, no RoPE, untied, emb std 1.0", make(rope=False, abs_pos=True, untied=True, emb_std=1.0))
hy._rope = _rope_orig
