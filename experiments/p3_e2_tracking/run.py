"""P3-E2 (v2): does a per-address state improve a learned write gate? Module level. See PREREG.md."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyprinc import MemConfig, Probe, Vocab, evaluate, train_probe, rebind_stream
from flybrain.harness import run_grid, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
REGIMES = [(0.3, 0.0), (1.0, 0.0), (0.3, 0.1), (1.0, 0.1)]          # (sigma, p_outlier)
B, N_STREAMS, P, T_EVAL, T_TRAIN = 8, 4, 48, 1500, 600
VOC = Vocab(n_keys=2048, n_vals=64, d_key=64, d_val=16, seed=1234)
BASE = dict(d_key=64, d_val=16, address="sparse_random", n_cells=1024, k_active=64, fan_in=7, write="delta")
CONDS = {
    "const_0.3": dict(gate="const", beta=0.3), "const_0.6": dict(gate="const", beta=0.6), "const_1.0": dict(gate="const", beta=1.0),
    "precision": dict(gate="precision"), "two_trace": dict(gate="const", n_compartments=2, betas=(0.2, 1.0), readout="confirm"),
    "learned_free": dict(gate="learned"), "learned_state": dict(gate="learned", gate_inputs=("nconf", "a", "zc")),
}
TRAINED = {"learned_free", "learned_state"}


def held_out():
    rng = np.random.default_rng(778)
    return {r: [rebind_stream(B, P, T_EVAL, VOC, rng, p_change=0.001, p_outlier=r[1]) for _ in range(N_STREAMS)] for r in REGIMES}


def train_fn(cfg, seed, run_dir):
    torch.manual_seed(seed)
    kw = {k: v for k, v in cfg.items() if k in MemConfig.__dataclass_fields__ and k != "seed"}   # the harness adds seed/lr/condition to every run config
    probe = Probe(MemConfig(**kw, seed=seed), VOC); t0 = time.time(); log = open(run_dir / "log.jsonl", "w")
    tr = {"trained": False, "steps": 0}                              # generalist training: the four regimes in round robin
    if cfg["condition"] in TRAINED and cfg["steps"] > 0:
        params = [p for p in probe.parameters() if p.requires_grad]; opt = torch.optim.Adam(params, lr=cfg["lr"]); g = torch.Generator().manual_seed(seed + 7)
        rng = np.random.default_rng(seed + 11); probe.train(); hist = []; diverged = False
        for s in range(cfg["steps"]):
            sig, po = REGIMES[s % len(REGIMES)]
            stream = rebind_stream(B, P, T_TRAIN, VOC, rng, p_change=0.001, p_outlier=po)
            loss, acc, _ = probe.run(stream, sig, g)
            if not torch.isfinite(loss): diverged = True; break
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); hist.append(float(loss))
            if (s + 1) % 10 == 0: log.write(json.dumps({"step": s + 1, "loss": float(loss), "acc": acc["all"]}) + "\n"); log.flush()
        tr = {"trained": True, "steps": len(hist), "diverged": diverged, "train_loss_last": float(np.mean(hist[-10:])) if hist else float("nan")}
    out = {"diverged": bool(tr.get("diverged", False)), **{f"train_{k}": v for k, v in tr.items()}}
    ho = held_out(); prim = []
    for (sig, po), streams in ho.items():
        m = evaluate(probe, streams, sigma=sig); tag = f"s{sig:g}_o{po:g}"
        out[f"acc_{tag}"] = m["acc"]; prim.append(m["acc"])
        for k, v in m.items():
            if k.startswith("acc_tag"): out[f"{k}_{tag}"] = v
    out["acc_mean"] = float(np.mean(prim))
    if out["diverged"]:
        for k in list(out):
            if k.startswith("acc"): out[k] = float("nan")
    out.update({"state_numel": probe.mem.c.state_numel, "params": sum(p.numel() for p in probe.parameters() if p.requires_grad), "wall_s": time.time() - t0})
    log.close(); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=80); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--variant", default=None); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {k: dict(BASE, **v, steps=args.steps) for k, v in CONDS.items() if not args.only or k in args.only}
    lrs = {k: ([1e-3, 3e-3, 1e-2] if k in TRAINED else [0.0]) for k in C}
    res = run_grid(EXP, C, [0.0], args.seeds, train_fn, per_condition_lrs=lrs, variant=args.variant)
    consts = [k for k in C if k.startswith("const_")]
    extra = [f"acc_s{s:g}_o{o:g}" for s, o in REGIMES] + ["acc_tag2_s0.3_o0.1", "acc_tag2_s1_o0.1", "acc_tag1_s0.3_o0.1", "acc_tag1_s1_o0"]
    an = analyse(res, "acc_mean", "learned_state", ["learned_free", "precision", "two_trace"] + consts, extra_metrics=extra)     # K0: nested control first
    an["precision_vs_fixed"] = analyse(res, "acc_mean", "precision", consts + ["learned_free"])["comparisons"]                 # P3
    an["two_trace_vs_precision"] = analyse(res, "acc_mean", "two_trace", ["precision"] + consts)["comparisons"]                # P4
    if consts:                                                      # the pre-registered fixed-rate control = the best const by seed-mean primary
        best_const = max(consts, key=lambda k: an["table"][k]["acc_mean"]["mean"] if k in an["table"] else -1); an["best_const"] = best_const
    md = "## P3-E2 (v2): learned gate with vs without per-address states; fixed rules for reference (overall accuracy, mean over 4 regimes), module level\n\n" + markdown_table(an, "acc_mean", [f"acc_s{s:g}_o{o:g}" for s, o in REGIMES] + ["acc_tag2_s0.3_o0.1", "acc_tag1_s0.3_o0.1"])
    write_summary(EXP, an, md, variant=args.variant); print(md)
