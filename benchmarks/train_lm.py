"""Tier-2 token-level LM training (GPU). Not run in this repository.

Hybrid decoder: every 4th block uses softmax attention (RoPE), the others use the chosen
fast-weight mixer. Uses the same FastWeightMemory implementation as the CPU experiments,
so the MB variant is exactly the tested module. Training data: FineWeb-Edu sample via
`datasets` (streaming), GPT-2 tokenizer.

Example:
  python benchmarks/train_lm.py --mixer mb  --params 125m --tokens 2.5e9 --lr 3e-3 --seed 0
  python benchmarks/train_lm.py --mixer gdn --params 125m --tokens 2.5e9 --lr 1e-3 --seed 0
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.modules.fast_weights import FastWeightMemory, FWConfig, matched_dense_dkey
from flybrain.models.backbone import Attention

SIZES = {"50m": dict(d=512, L=8, H=8), "125m": dict(d=768, L=12, H=12), "350m": dict(d=1024, L=24, H=16)}


def mixer(kind, d, H, seed):
    if kind == "attention":
        return Attention(d, n_heads=H), None
    dv = d // H
    mb = FWConfig(d_model=d, n_heads=H, d_value=dv, key_mode="sparse_random", n_pn=64, n_kc=2048, fan_in=5, kwta_frac=0.05,
                  write="delta", gate="error", decay="learned", seed=seed)
    if kind == "mb":
        return FastWeightMemory(mb), None
    if kind == "mb_dense_expanded":
        return FastWeightMemory(FWConfig(**{**mb.__dict__, "key_mode": "dense_expanded", "kwta": False})), None
    if kind == "gdn":
        dk = matched_dense_dkey(mb, "effective")
        return FastWeightMemory(FWConfig(d_model=d, n_heads=H, d_value=dv, key_mode="dense", d_key=dk, write="delta", gate="learned", decay="learned")), nn.Identity()
    raise ValueError(kind)


class Block(nn.Module):
    def __init__(self, d, H, kind, seed):
        super().__init__(); self.n1 = nn.RMSNorm(d); self.mix, _ = mixer(kind, d, H, seed)
        self.n2 = nn.RMSNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        y, _ = self.mix(self.n1(x)); x = x + y
        return x + self.mlp(self.n2(x))


class HybridLM(nn.Module):
    def __init__(self, vocab, d, L, H, kind, seed):
        super().__init__(); self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, H, "attention" if (i + 1) % 4 == 0 else kind, seed + i) for i in range(L)])
        self.norm = nn.RMSNorm(d); self.head = nn.Linear(d, vocab, bias=False); self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixer", default="mb"); ap.add_argument("--params", default="125m"); ap.add_argument("--tokens", type=float, default=2.5e9)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--out", default="benchmarks/runs"); args = ap.parse_args()
    from datasets import load_dataset; from transformers import AutoTokenizer
    torch.manual_seed(args.seed); dev = "cuda"
    tok = AutoTokenizer.from_pretrained("gpt2"); s = SIZES[args.params]
    model = HybridLM(len(tok), s["d"], s["L"], s["H"], args.mixer, args.seed).to(dev)
    n_params = sum(p.numel() for p in model.parameters()); print("params", n_params)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True).shuffle(seed=args.seed, buffer_size=10_000)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps = int(args.tokens // (args.batch * args.seq)); warm = 1000
    out = Path(args.out) / f"{args.mixer}_{args.params}_seed{args.seed}"; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"params": n_params}))
    buf, it, t0 = [], iter(ds), time.time()
    for step in range(steps):
        while len(buf) < args.batch * (args.seq + 1):
            buf.extend(tok(next(it)["text"])["input_ids"] + [tok.eos_token_id])
        x = torch.tensor(buf[: args.batch * (args.seq + 1)], device=dev).view(args.batch, args.seq + 1); buf = buf[args.batch * (args.seq + 1):]
        lr = args.lr * min(1, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups: g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1).float(), x[:, 1:].flatten())
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 100 == 0:
            print(json.dumps({"step": step, "loss": loss.item(), "lr": lr, "t": time.time() - t0}), flush=True)
        if step % 5000 == 0 or step == steps - 1:
            torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
