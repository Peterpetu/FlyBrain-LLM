"""H3: dcSBM-sampled masks vs Erdos-Renyi as char-LM recurrent masks."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flybrain.connectome import load_connectome
from flybrain.connectome.controls import fit_dcsbm, sample_dcsbm, erdos_renyi, graph_stats
from flybrain.modules.masked_rnn import select_core
from flybrain.models import SeqModel, ModelConfig, count_params
from flybrain.tasks.text import CharCorpus
from flybrain.harness import run_grid, train, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
D, T, B = 64, 128, 32
_C, _FIT = None, None


def fit():
    global _C, _FIT
    if _FIT is None:
        _C = load_connectome(); idx = select_core(_C, 4096); W = _C.subgraph(idx)
        _FIT = fit_dcsbm(W, _C.cell_type[idx], _C.sign[idx])
        (EXP / "fit_stats.json").write_text(json.dumps({"fit_on": graph_stats(W), "n_blocks": int(len(_FIT["sizes"]))}, indent=1))
    return _FIT


def train_fn(cfg, seed, run_dir):
    corp = CharCorpus("tinyshakespeare"); torch.manual_seed(seed); rng = np.random.default_rng(1000 + seed)
    N = cfg["N"]
    Ws, blocks, sign = sample_dcsbm(fit(), N, rng)
    n = Ws.shape[0]
    if cfg["kind"] == "er":
        Ws = erdos_renyi(n, Ws.nnz, rng, weights=Ws.data.astype(np.float32))
    model = SeqModel(ModelConfig(vocab=corp.vocab, d_model=D, n_layers=1, mixer="rnn", mixer_kwargs=dict(n_hidden=n, W_adj=Ws, sign=sign, dale=True, init_from_counts=True)))
    evals = {"val_bpc": lambda: corp.eval_bpc(model, T, batch=64, max_chars=100_000)}
    m = train(model, lambda rng_: corp.batch("train", B, T, rng_), steps=cfg["steps"], lr=cfg["lr"], evals=evals, seed=seed, run_dir=run_dir, eval_every=100)
    m["params"] = count_params(model); m["graph"] = graph_stats(Ws); m["inh_frac"] = float((sign < 0).mean())
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024]); ap.add_argument("--threads", type=int, default=4); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {f"{k}_N{N}": dict(kind=k, N=N, steps=args.steps) for N in args.sizes for k in ("sbm", "er")}
    res = run_grid(EXP, C, [1e-3, 3e-3, 1e-2], args.seeds, train_fn)
    md = "## H3: dcSBM mask vs Erdos-Renyi, validation bits/char (lower is better)\n"; full = {}
    for N in args.sizes:
        an = analyse(res, "val_bpc_last2", f"sbm_N{N}", [f"er_N{N}"], higher_is_better=False, extra_metrics=["params"]); full[f"N{N}"] = an
        md += f"\n### N = {N}\n\n" + markdown_table(an, "val_bpc") + "\n"
    write_summary(EXP, full, md); print(md)
