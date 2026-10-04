"""H1b: mushroom-body memory in the MQAR capacity regime. See PREREG.md (frozen at first run)."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flybrain.models import SeqModel, ModelConfig, count_params
from flybrain.modules.fast_weights import FWConfig, fast_weight_flops, matched_dense_dkey
from flybrain.tasks.mqar import mqar_batch, mqar_accuracy
from flybrain.harness import run_grid, train, analyse, markdown_table, write_summary
from flybrain.connectome.circuits import load_circuit

EXP = Path(__file__).resolve().parent
VOCAB, D, T, NPAIRS, B = 192, 64, 160, 40, 32


def pn_type_matrix(side="R") -> np.ndarray:
    z = load_circuit(f"mushroom_body_{side}")
    W = z["W_pn_kc"].astype(np.float32)                       # (PN neurons, KC)
    types = z["label_pn_type"]; u, inv = np.unique(types, return_inverse=True)
    E = np.zeros((W.shape[1], len(u)), dtype=np.float32)      # (KC, PN types)
    for j in range(len(u)):
        E[:, j] = W[inv == j].sum(0)
    return E


def conditions(args):
    E = pn_type_matrix("R"); n_pn = E.shape[1]
    base = dict(n_heads=2, d_value=16, n_pn=n_pn, fan_in=5, kwta_frac=0.05, write="delta", gate="error", decay="learned", conv_kernel=4)
    mb = dict(base, key_mode="sparse_random", n_kc=1024)
    mbc = FWConfig(d_model=D, **mb)
    dk = matched_dense_dkey(mbc, "effective")
    C = {
        "mb": mb,
        "abl_dense_expanded": dict(base, key_mode="dense_expanded", n_kc=1024, kwta=False),
        "gdn_matched": dict(n_heads=2, d_value=16, key_mode="dense", d_key=dk, write="delta", gate="learned", decay="learned", conv_kernel=4),
    }
    if args.only:
        C = {k: v for k, v in C.items() if k in args.only}
    return C, dk


def build(cfg):
    kw = {k: v for k, v in cfg.items() if k not in ("lr", "seed", "condition", "mixer", "steps")}
    if cfg.get("mixer") == "attention":
        return SeqModel(ModelConfig(vocab=VOCAB, d_model=D, n_layers=2, mixer="attention", mixer_kwargs=kw))
    return SeqModel(ModelConfig(vocab=VOCAB, d_model=D, n_layers=1, mixer="fastweight", mixer_kwargs=dict(kw, seed=cfg["seed"])))


def train_fn(cfg, seed, run_dir):
    torch.manual_seed(seed)
    model = build(cfg)
    ev_rng = np.random.default_rng(1234)
    fixed = {"acc_in": [mqar_batch(B, T, NPAIRS, VOCAB, ev_rng) for _ in range(4)],
             "acc_pairs60_T256": [mqar_batch(B, 256, 60, VOCAB, ev_rng) for _ in range(2)],
             "acc_pairs8_T160": [mqar_batch(B, 160, 8, VOCAB, ev_rng) for _ in range(2)]}
    def ev(batches):
        return lambda: float(np.mean([mqar_accuracy(model(x), y) for x, y in batches]))
    evals = {k: ev(v) for k, v in fixed.items()}
    m = train(model, lambda rng: mqar_batch(B, T, NPAIRS, VOCAB, rng), steps=cfg["steps"], lr=cfg["lr"], evals=evals, seed=seed, run_dir=run_dir, eval_every=100)
    m["params"] = count_params(model)
    if cfg.get("mixer") != "attention":
        fw = FWConfig(d_model=D, **{k: v for k, v in cfg.items() if k in FWConfig.__dataclass_fields__})
        m["flops"] = fast_weight_flops(fw)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=800); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--threads", type=int, default=4); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C, dk = conditions(args)
    print(f"matched dense key dim = {dk}")
    lrs_h = [1e-3, 3e-3, 1e-2]; lrs_b = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    per = {k: (lrs_b if k == "gdn_matched" else lrs_h) for k in C}
    conds = {k: dict(v, steps=args.steps) for k, v in C.items()}
    res = run_grid(EXP, conds, lrs_h, args.seeds, train_fn, per_condition_lrs=per)
    an = analyse(res, "acc_in_last2", "mb", ["gdn_matched", "abl_dense_expanded"], extra_metrics=["acc_pairs60_T256", "acc_pairs8_T160", "params"])
    md = "## H1b capacity regime: MQAR accuracy (T=160, 40 pairs; chance 0.016), best LR per condition\n\n" + markdown_table(an, "acc_in", ["acc_pairs60_T256", "acc_pairs8_T160"])
    an["matched_dense_dkey"] = dk
    write_summary(EXP, an, md); print(md)
