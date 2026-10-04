"""H5: connectome graph as recurrent mask, with the full control set. See PREREG.md."""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flybrain.connectome import load_connectome
from flybrain.connectome.controls import degree_preserving_shuffle, erdos_renyi, graph_stats
from flybrain.modules.masked_rnn import select_core
from flybrain.models import SeqModel, ModelConfig, count_params
from flybrain.tasks.text import CharCorpus
from flybrain.harness import run_grid, train, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
D, T, B = 64, 128, 32
_C = None


def connectome():
    global _C
    if _C is None:
        _C = load_connectome()
    return _C


def make_graph(kind, N, seed):
    c = connectome(); idx = select_core(c, N); W = c.subgraph(idx); sign = c.sign[idx]
    rng = np.random.default_rng(1000 + seed)
    if kind == "real":
        return W, sign
    if kind == "shuffle":
        return degree_preserving_shuffle(W, rng, 10), sign
    if kind == "er":
        return erdos_renyi(N, W.nnz, rng, weights=W.data.astype(np.float32)), sign
    raise ValueError(kind)


def train_fn(cfg, seed, run_dir):
    corp = CharCorpus("tinyshakespeare"); torch.manual_seed(seed)
    N = cfg["N"]; kind = cfg["kind"]
    if kind in ("real", "shuffle", "er"):
        W, sign = make_graph(kind, N, seed)
        gs = graph_stats(W)
        kw = dict(n_hidden=N, W_adj=W, sign=sign, dale=True, init_from_counts=True)
    else:
        idx = select_core(connectome(), N); E = int(connectome().subgraph(idx).nnz)
        n_hidden = int(round(np.sqrt(E))) if kind == "dense_param" else N
        kw = dict(n_hidden=n_hidden, dale=False); gs = {"n": n_hidden, "E": n_hidden * n_hidden}
    model = SeqModel(ModelConfig(vocab=corp.vocab, d_model=D, n_layers=1, mixer="rnn", mixer_kwargs=kw))
    evals = {"val_bpc": lambda: corp.eval_bpc(model, T, batch=64, max_chars=100_000)}
    m = train(model, lambda rng: corp.batch("train", B, T, rng), steps=cfg["steps"], lr=cfg["lr"], evals=evals, seed=seed, run_dir=run_dir, eval_every=100)
    m["params"] = count_params(model); m["graph"] = gs; m["n_recurrent_params"] = model.blocks[0].mix.core.n_recurrent_params()
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024]); ap.add_argument("--threads", type=int, default=4); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    lrs_g = [1e-3, 3e-3, 1e-2]; lrs_d = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    C, per = {}, {}
    for N in args.sizes:
        for kind in ("real", "shuffle", "er", "dense_param", "dense_N"):
            name = f"{kind}_N{N}"; C[name] = dict(kind=kind, N=N, steps=args.steps); per[name] = lrs_g if kind in ("real", "shuffle", "er") else lrs_d
    res = run_grid(EXP, C, lrs_g, args.seeds, train_fn, per_condition_lrs=per)
    md = "## H5: validation bits/char on TinyShakespeare (lower is better), best LR per condition\n"
    full = {}
    for N in args.sizes:
        an = analyse([r for r in res if r["condition"].endswith(f"_N{N}")], "val_bpc_last2", f"real_N{N}", [f"shuffle_N{N}", f"er_N{N}", f"dense_param_N{N}", f"dense_N_N{N}"], higher_is_better=False, extra_metrics=["params"])
        full[f"N{N}"] = an
        md += f"\n### N = {N}\n\n" + markdown_table(an, "val_bpc") + "\n"
    write_summary(EXP, full, md); print(md)
