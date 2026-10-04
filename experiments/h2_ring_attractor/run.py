"""H2: ring attractor state tracking with length extrapolation. See PREREG.md."""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flybrain.models import SeqModel, ModelConfig, count_params
from flybrain.tasks.compass import compass_batch, compass_vocab
from flybrain.harness import run_grid, train, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
VOCAB, D, T, B, NSEC = compass_vocab(4), 32, 32, 32, 8
CONDS = {
    "ring_connectome": dict(mixer="ring", mode="connectome"),
    "ring_canonical": dict(mixer="ring", mode="canonical", n_units=148),
    "rnn_unstructured": dict(mixer="ring", mode="unstructured", n_units=148),
    "rnn_plain": dict(mixer="rnn", n_hidden=148, dale=False),
    "gdn": dict(mixer="fastweight", n_heads=2, d_value=16, key_mode="dense", d_key=32, write="delta", gate="learned", decay="learned"),
    "attention": dict(mixer="attention", n_heads=2),
}


def build(cfg):
    kw = {k: v for k, v in cfg.items() if k not in ("lr", "seed", "condition", "mixer", "steps")}
    if cfg["mixer"] in ("ring",):
        kw["seed"] = cfg["seed"]
    return SeqModel(ModelConfig(vocab=VOCAB, d_model=D, n_layers=1, mixer=cfg["mixer"], mixer_kwargs=kw, n_out=NSEC))


def acc(model, batches):
    return float(np.mean([(model(x).argmax(-1) == y).float().mean().item() for x, y in batches]))


def train_fn(cfg, seed, run_dir):
    torch.manual_seed(seed); model = build(cfg)
    ev_rng = np.random.default_rng(1234)
    fixed = {f"acc_T{L}": [compass_batch(B, L, ev_rng, NSEC) for _ in range(4)] for L in (32, 128, 256, 512)}
    evals = {k: (lambda v=v: acc(model, v)) for k, v in fixed.items()}
    m = train(model, lambda rng: compass_batch(B, T, rng, NSEC), steps=cfg["steps"], lr=cfg["lr"], evals=evals, seed=seed, run_dir=run_dir, ignore_index=-100)
    m["params"] = count_params(model)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=600); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--threads", type=int, default=4); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {k: dict(v, steps=args.steps) for k, v in CONDS.items() if not args.only or k in args.only}
    lrs_h = [1e-3, 3e-3, 1e-2]; lrs_b = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    per = {k: (lrs_h if k in ("ring_connectome", "ring_canonical") else lrs_b) for k in C}
    res = run_grid(EXP, C, lrs_h, args.seeds, train_fn, per_condition_lrs=per)
    an = analyse(res, "acc_T128_last2", "ring_connectome", ["rnn_unstructured", "rnn_plain", "gdn", "attention", "ring_canonical"],
                 extra_metrics=["acc_T32_last2", "acc_T256_last2", "acc_T512_last2", "params"])
    md = "## H2: compass state tracking, accuracy at 4x training length (T=128), best LR per condition\n\n" + \
         markdown_table(an, "acc_T128", ["acc_T32_last2", "acc_T256_last2", "acc_T512_last2"])
    write_summary(EXP, an, md); print(md)
