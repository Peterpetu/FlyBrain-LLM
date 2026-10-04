"""P3-E1: sparse random addressing vs dense keys at equal active rows (module level). See PREREG.md."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyprinc import MemConfig, Probe, Vocab, evaluate, train_probe, capacity_stream
from flybrain.harness import run_grid, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
PS = (32, 64, 128, 256, 512); B, N_STREAMS, P_TRAIN = 8, 6, 128
VOC = Vocab(n_keys=2048, n_vals=64, d_key=64, d_val=16, seed=1234)
BASE = dict(d_key=64, d_val=16, n_cells=1024, k_active=64, fan_in=7, write="delta", gate="const", beta=1.0)
CONDS = {
    "sparse_random": dict(address="sparse_random"),
    "dense": dict(address="dense"),
    "dense_expanded": dict(address="dense_expanded"),
    "sparse_learned": dict(address="sparse_learned"),
    "sparse_nonorm": dict(address="sparse_random", normalise=False),
    "sparse_hebb": dict(address="sparse_random", write="hebb"),
}
TRAINED = {"dense_expanded", "sparse_learned"}


def held_out():
    rng = np.random.default_rng(777)
    return {P: [capacity_stream(B, P, VOC, rng) for _ in range(N_STREAMS)] for P in PS}


def train_fn(cfg, seed, run_dir):
    torch.manual_seed(seed)
    kw = {k: v for k, v in cfg.items() if k in MemConfig.__dataclass_fields__ and k != "seed"}   # the harness adds seed/lr/condition to every run config
    probe = Probe(MemConfig(**kw, seed=seed), VOC); t0 = time.time()
    log = open(run_dir / "log.jsonl", "w")
    tr = train_probe(probe, lambda r: capacity_stream(B, P_TRAIN, VOC, r), steps=cfg["steps"] if cfg["condition"] in TRAINED else 0, lr=cfg["lr"], seed=seed,
                     log=lambda s, l, a: (log.write(json.dumps({"step": s, "loss": l, "acc": a}) + "\n"), log.flush()))
    out = {"diverged": bool(tr.get("diverged", False)), **{f"train_{k}": v for k, v in tr.items()}}
    ho = held_out()
    for P in PS:
        m = evaluate(probe, ho[P]); out[f"acc_P{P}"] = m["acc"]
    if out["diverged"]:
        for P in PS: out[f"acc_P{P}"] = float("nan")
    c = probe.mem.c
    out.update({"state_numel": c.state_numel, "active_rows": c.active_rows, "params": sum(p.numel() for p in probe.parameters() if p.requires_grad), "wall_s": time.time() - t0})
    log.close(); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--variant", default=None); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {k: dict(BASE, **v, steps=args.steps) for k, v in CONDS.items() if not args.only or k in args.only}
    lrs = {k: ([1e-3, 3e-3, 1e-2] if k in TRAINED else [0.0]) for k in C}
    res = run_grid(EXP, C, [0.0], args.seeds, train_fn, per_condition_lrs=lrs, variant=args.variant)
    an = analyse(res, "acc_P128", "sparse_random", ["dense", "dense_expanded", "sparse_learned", "sparse_nonorm", "sparse_hebb"],
                 extra_metrics=[f"acc_P{P}" for P in PS if P != 128] + ["state_numel", "active_rows"])
    md = "## P3-E1: one-shot recall accuracy at P = 128 stored pairs (chance 1/64), module level\n\n" + markdown_table(an, "acc_P128", ["acc_P32", "acc_P64", "acc_P256", "acc_P512", "state_numel", "active_rows"])
    write_summary(EXP, an, md, variant=args.variant); print(md)
