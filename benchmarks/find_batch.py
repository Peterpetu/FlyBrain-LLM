#!/usr/bin/env python3
"""Find the largest batch size that fits on the available GPU for a config, and report
tokens/s and peak memory for each mixer. Run this FIRST on a new machine.

  python benchmarks/find_batch.py --size 50m --seq 512 --mixers flyai gdn --compile
On CPU it only checks that the configuration runs (memory is reported as RSS)."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flyai.lm import make_lm


def try_batch(size, mixer, block, seq, batch, steps, dev, compile_):
    torch.manual_seed(0)
    if dev == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = make_lm(size, mixer, vocab=50257, **block).to(dev)
    if compile_ and dev == "cuda":
        for l in m.layers:
            if l.kind == "flyai": l.mix.step = torch.compile(l.mix.step, dynamic=False, mode="reduce-overhead")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    x = torch.randint(0, 50257, (batch, seq + 1), device=dev)
    try:
        for i in range(steps + 1):
            if i == 1:
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = m.loss(x)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if dev == "cuda": torch.cuda.synchronize()
        dt = (time.time() - t0) / steps
        mem = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
        return {"ok": True, "tok_per_s": batch * seq / dt, "step_s": dt, "peak_mem_gb": mem, "params": m.n_params()}
    except torch.cuda.OutOfMemoryError:
        return {"ok": False}
    finally:
        del m, opt
        if dev == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="50m"); ap.add_argument("--seq", type=int, default=512); ap.add_argument("--mixers", nargs="+", default=["flyai", "gdn"])
    ap.add_argument("--config", default=None, help="benchmarks/configs/*.json to take the block settings from"); ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--max_batch", type=int, default=64); args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    block = json.load(open(args.config))["block"] if args.config else {}
    print(f"device {dev}; size {args.size}; seq {args.seq}; block {block}")
    for mixer in args.mixers:
        b = args.max_batch; res = None
        while b >= 1:
            r = try_batch(args.size, mixer, block, args.seq, b, args.steps, dev, args.compile)
            if r["ok"]: res = (b, r); break
            b //= 2
        print(json.dumps({"mixer": mixer, "batch": res[0] if res else 0, **(res[1] if res else {})}), flush=True)
