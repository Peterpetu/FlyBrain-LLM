"""Token-level training of FlyAI / Gated-DeltaNet / attention hybrid LMs.

Two data modes:
  --data local:<path.txt>        byte-level tokens from a local text file (CPU smoke tests, any machine)
  --data fineweb-edu             HuggingFaceFW/fineweb-edu sample-10BT, streaming, GPT-2 tokenizer (GPU)
Sizes: tiny (tests) | 50m | 125m | 350m  (see benchmarks/configs/*.json for the pre-registered settings).

Examples
  # CPU smoke (also what the repository's own test does):
  python benchmarks/train_flyai_lm.py --size tiny --mixer flyai --data local:data/text/tinyshakespeare.txt --steps 20 --seq 64 --batch 8
  # GPU, pre-registered 125M run and its matched control:
  python benchmarks/train_flyai_lm.py --size 125m --mixer flyai --data fineweb-edu --tokens 2.5e9 --lr 3e-3 --seed 0 --bf16
  python benchmarks/train_flyai_lm.py --size 125m --mixer gdn   --data fineweb-edu --tokens 2.5e9 --lr 1e-3 --seed 0 --bf16
Every run writes config.json, log.jsonl, model.pt (and an HF export via benchmarks/hf_flyai.py when --export is set),
and logs measured FLOPs/token (torch.utils.flop_counter, one forward+backward at start) next to tokens/s.
Multi-GPU: `torchrun --nproc_per_node=N benchmarks/train_flyai_lm.py ...` (DDP; each rank streams its own shard
by seed; --batch is per rank; rank 0 logs and saves).
"""
from __future__ import annotations

import argparse, contextlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyai.lm import make_lm


class LocalBytes:
    def __init__(self, path, seq, batch, seed):
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8).astype(np.int64)
        n = int(len(data) * 0.95); self.train, self.val = torch.from_numpy(data[:n]), torch.from_numpy(data[n:])
        self.seq, self.batch, self.rng, self.vocab = seq, batch, np.random.default_rng(seed), 256

    def next(self):
        ix = self.rng.integers(0, len(self.train) - self.seq - 1, size=self.batch)
        return torch.stack([self.train[i:i + self.seq + 1] for i in ix])

    def val_batches(self, n=8):
        k_max = n if n is not None else len(self.val) // (self.batch * self.seq) + 1
        for k in range(k_max):
            ix = np.arange(k * self.batch, (k + 1) * self.batch) * self.seq
            ix = ix[ix + self.seq + 1 < len(self.val)]
            if len(ix) == 0: break
            yield torch.stack([self.val[i:i + self.seq + 1] for i in ix])


class FineWebEdu:
    """Streaming FineWeb-Edu. The 64 validation documents come from a FIXED-seed stream (identical
    on every rank and every training seed) and are excluded from every rank's training stream by
    content hash, so multi-GPU shards cannot train on another rank's validation text."""
    VAL_SEED, N_VAL = 20260915, 64

    def __init__(self, seq, batch, seed):
        import hashlib
        from datasets import load_dataset; from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("gpt2"); self.vocab = len(self.tok)
        base = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        self.val_docs = [d["text"] for _, d in zip(range(self.N_VAL), base.shuffle(seed=self.VAL_SEED, buffer_size=10_000))]
        self._h = lambda t: hashlib.sha1(t.encode("utf-8", "ignore")).digest()
        self._val_hashes = {self._h(t) for t in self.val_docs}
        self.it, self.buf, self.seq, self.batch = iter(base.shuffle(seed=seed, buffer_size=10_000)), [], seq, batch

    def _fill(self, need):
        while len(self.buf) < need:
            text = next(self.it)["text"]
            if self._h(text) in self._val_hashes: continue                      # held-out document: never trained on
            self.buf.extend(self.tok(text)["input_ids"] + [self.tok.eos_token_id])

    def next(self):
        need = self.batch * (self.seq + 1); self._fill(need)
        x = torch.tensor(self.buf[:need]).view(self.batch, self.seq + 1); self.buf = self.buf[need:]; return x

    def val_batches(self, n=8):
        ids = []
        for d in self.val_docs: ids += self.tok(d)["input_ids"] + [self.tok.eos_token_id]
        ids = torch.tensor(ids); per = self.batch * (self.seq + 1)
        for k in range(len(ids) // per if n is None else min(n, len(ids) // per)):
            yield ids[k * per:(k + 1) * per].view(self.batch, self.seq + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="125m"); ap.add_argument("--mixer", default="flyai", choices=["flyai", "gdn", "attention"])
    ap.add_argument("--data", default="fineweb-edu"); ap.add_argument("--tokens", type=float, default=2.5e9); ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seq", type=int, default=2048); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--wd", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true"); ap.add_argument("--K", type=int, default=None); ap.add_argument("--block", default="{}", help="JSON of BlockConfig overrides")
    ap.add_argument("--gdn_key", type=int, default=None, help="GDN twin key dim; default = head dim d / n_heads (PREREG_tier2)"); ap.add_argument("--out", default="benchmarks/runs"); ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--export", action="store_true", help="also save a HuggingFace export for lm-eval"); ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--compile", action="store_true", help="torch.compile the FlyAI per-token step (any device)")
    ap.add_argument("--compile_mode", default="default", help="torch.compile mode; 'reduce-overhead' (CUDA graphs) is opt-in and unverified for this per-token loop")
    ap.add_argument("--config", default=None, help="benchmarks/configs/*.json: takes size, mixer, block and train defaults from it")
    ap.add_argument("--arm", default=None, help="name of a `controls` entry of --config whose overrides (mixer / block) are merged in; run dir is named after it")
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps (effective batch = batch x grad_accum x world)")
    args = ap.parse_args()
    block = json.loads(args.block); arm_name = None
    if args.config:
        cfgj = json.load(open(args.config)); given = {a[2:].split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
        args.size = cfgj.get("size", args.size); args.mixer = cfgj.get("mixer", args.mixer)
        block = {**cfgj.get("block", {}), **block}
        for k, v in cfgj.get("train", {}).items():                       # config train defaults unless given on the CLI
            if k in ("tokens", "seq", "batch", "lr", "warmup", "wd", "grad_accum") and k not in given: setattr(args, k, v)
            if k in ("bf16", "compile") and v and k not in given: setattr(args, k, True)
        if args.arm:
            arm = cfgj["controls"][args.arm]; arm_name = args.arm
            args.mixer = arm.get("mixer", args.mixer); block = {**block, **arm.get("block", {})}
    if args.threads: torch.set_num_threads(args.threads)
    # ---- distributed (torchrun) ----
    ddp = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    if ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        if torch.cuda.is_available(): torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.manual_seed(args.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = LocalBytes(args.data[6:], args.seq, args.batch, args.seed * 1000 + rank) if args.data.startswith("local:") else FineWebEdu(args.seq, args.batch, args.seed * 1000 + rank)
    if args.K is not None: block["K_max"] = args.K
    model = make_lm(args.size, args.mixer, vocab=data.vocab, gdn_key=args.gdn_key, seed=args.seed, **block).to(dev)
    raw = model
    # measured FLOPs per token: one forward+backward on a real batch, before compile / DDP wrapping and WITHOUT the
    # activation-checkpoint recompute (which is not compute the model needs, and which the gdn/attention twins do not have)
    xf = data.next().to(dev); fly_blocks = [l.mix for l in raw.layers if l.kind == "flyai"]; saved = [b.c.chunk for b in fly_blocks]
    for b in fly_blocks: b.c.chunk = 1 << 30
    with FlopCounterMode(display=False) as fc:
        raw.loss(xf).backward()
    for b, ch in zip(fly_blocks, saved): b.c.chunk = ch
    raw.zero_grad(set_to_none=True)
    flops_per_token = fc.get_total_flops() / xf[:, :-1].numel()
    state_numel = raw.state_numel()                                      # recurrent state per batch row (PREREG_tier2 reporting)
    if args.compile:
        for l in raw.layers:
            if l.kind == "flyai": l.mix.step = torch.compile(l.mix.step, dynamic=False, mode=args.compile_mode)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[int(os.environ.get("LOCAL_RANK", 0))] if dev == "cuda" else None, find_unused_parameters=True)
    n_params = raw.n_params()
    steps = args.steps or int(args.tokens // (args.batch * args.seq * world * args.grad_accum))
    out = Path(args.out) / f"{arm_name or args.mixer}_{args.size}_seed{args.seed}_lr{args.lr:g}"; out.mkdir(parents=True, exist_ok=True)   # LR in the name: sweep runs never overwrite
    if rank == 0:
        (out / "config.json").write_text(json.dumps({**vars(args), "block": block, "params": n_params, "steps": steps, "device": dev, "world_size": world,
                                                     "flops_per_token": flops_per_token, "state_numel_per_row": state_numel, "gdn_key_used": raw.cfg.gdn_key}, indent=1))
    from flyai.primitives import param_groups, after_optimizer_step
    opt = torch.optim.AdamW(param_groups(raw, args.wd), lr=args.lr, betas=(0.9, 0.95))
    log = open(out / "log.jsonl", "a") if rank == 0 else None; t0 = time.time(); tokens_seen = 0
    if rank == 0: print(f"{args.mixer} {args.size}: {n_params/1e6:.1f}M params, {flops_per_token/1e6:.2f} MFLOPs/token (fwd+bwd), {steps} steps on {dev} x {world}", flush=True)
    for step in range(steps):
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / max(1, steps))))
        for g in opt.param_groups: g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x = data.next().to(dev)
            sync = contextlib.nullcontext() if (not ddp or micro == args.grad_accum - 1) else model.no_sync()   # one all-reduce per optimiser step
            with sync, torch.autocast(dev, dtype=torch.bfloat16, enabled=args.bf16 and dev == "cuda"):
                loss = (model.module if ddp else model).loss(x) if not ddp else _ddp_loss(model, x)
            (loss / args.grad_accum).backward()
            tokens_seen += x.numel() * world
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0); opt.step(); after_optimizer_step(raw)
        if step % 50 == 0 or step == steps - 1:
            lt = loss.detach().clone()
            if ddp: dist.all_reduce(lt); lt /= world
            if rank == 0:
                rec = {"step": step, "loss": float(lt), "lr": lr, "tokens": tokens_seen, "tok_per_s": tokens_seen / (time.time() - t0), "flops_per_s": flops_per_token * tokens_seen / (time.time() - t0), **raw.diagnostics()}
                if dev == "cuda": rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % args.eval_every == 0 or step == steps - 1:
            raw.eval(); tot, cnt = 0.0, 0
            with torch.no_grad():
                for xb in data.val_batches(n=None):                       # the whole held-out set, not the first 8 batches
                    xb = xb.to(dev)
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=args.bf16 and dev == "cuda"):
                        l = raw.loss(xb)
                    tot += float(l) * (xb.shape[1] - 1) * xb.shape[0]; cnt += (xb.shape[1] - 1) * xb.shape[0]
            raw.train(); vl = torch.tensor([tot, cnt], device=dev)
            if ddp: dist.all_reduce(vl)
            if rank == 0:
                rec = {"step": step, "val_loss": float(vl[0] / max(vl[1], 1)), "val_ppl": math.exp(float(vl[0] / max(vl[1], 1)))}
                print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                torch.save(raw.state_dict(), out / "model.pt")
    if args.export and rank == 0:
        from benchmarks.hf_flyai import export_checkpoint
        export_checkpoint(raw, raw.cfg, "gpt2" if not args.data.startswith("local:") else None, out / "hf")
    if rank == 0: (out / "done").write_text(json.dumps({"steps": steps, "tokens": tokens_seen}))   # completion marker (model.pt is written at every eval)
    if ddp: dist.barrier(); dist.destroy_process_group()
    if rank == 0: print("done", out)


def _ddp_loss(ddp_model, x):
    """Run the DDP-wrapped forward (so gradients are synchronised) and compute the causal LM loss."""
    logits, _ = ddp_model(x[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x[:, 1:].reshape(-1))


if __name__ == "__main__":
    main()
