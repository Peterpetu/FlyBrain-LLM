"""H4: sign-constrained (Dale) recurrent layer vs unconstrained, long-context stability."""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flybrain.models import SeqModel, ModelConfig, count_params
from flybrain.tasks.text import CharCorpus
from flybrain.harness import run_grid, train, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
D, T, B, N = 64, 128, 32, 256


def long_context_probe(model, corp, L):
    model.eval()
    data = corp.val[: L + 1]; x = data[:L][None]; y = data[1:L + 1][None]
    emb = model.emb(x).detach().requires_grad_(True)
    h = emb
    for blk in model.blocks:
        h = blk(h)
    logits = model.head(model.norm(h))
    loss_all = torch.nn.functional.cross_entropy(logits[0], y[0], reduction="none")
    bpc = float(loss_all.mean() / np.log(2))
    hid, _ = model.blocks[0].mix.core(model.blocks[0].n1(emb))
    hn = hid.norm(dim=-1)[0]; ratio_h = float(hn[-128:].mean() / hn[:128].mean())
    g, = torch.autograd.grad(loss_all[-128:].sum(), emb)
    gr = float(g[0, :128].norm() / (g[0, -128:].norm() + 1e-9))
    model.train()
    return bpc, ratio_h, gr


def train_fn(cfg, seed, run_dir):
    corp = CharCorpus("tinyshakespeare"); torch.manual_seed(seed); rng = np.random.default_rng(1000 + seed)
    if cfg["kind"] == "dale":
        f = cfg["inh_frac"]; sign = np.where(rng.random(N) < f, -1, 1)
        A = sp.csr_matrix(np.ones((N, N), dtype=np.float32))
        kw = dict(n_hidden=N, W_adj=A, sign=sign, dale=True, init_from_counts=False)
    else:
        kw = dict(n_hidden=N, dale=False)
    model = SeqModel(ModelConfig(vocab=corp.vocab, d_model=D, n_layers=1, mixer="rnn", mixer_kwargs=kw))
    evals = {"val_bpc": lambda: corp.eval_bpc(model, T, batch=64, max_chars=100_000)}
    m = train(model, lambda r: corp.batch("train", B, T, r), steps=cfg["steps"], lr=cfg["lr"], evals=evals, seed=seed, run_dir=run_dir, eval_every=100)
    for L in (512, 1024, 2048):
        bpc, rh, gr = long_context_probe(model, corp, L)
        m[f"bpc_L{L}"] = bpc; m[f"hnorm_ratio_L{L}"] = rh; m[f"grad_ratio_L{L}"] = gr
    m["params"] = count_params(model)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=400); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=4); args = ap.parse_args()
    torch.set_num_threads(args.threads)
    C = {"unconstrained": dict(kind="dense", steps=args.steps)}
    for f in (0.2, 0.3, 0.5):
        C[f"dale_f{f}"] = dict(kind="dale", inh_frac=f, steps=args.steps)
    per = {k: ([3e-4, 1e-3, 3e-3, 1e-2, 3e-2] if k == "unconstrained" else [1e-3, 3e-3, 1e-2]) for k in C}
    res = run_grid(EXP, C, [1e-3, 3e-3, 1e-2], args.seeds, train_fn, per_condition_lrs=per)
    an = analyse(res, "val_bpc_last2", "dale_f0.3", ["unconstrained", "dale_f0.2", "dale_f0.5"], higher_is_better=False,
                 extra_metrics=["bpc_L512", "bpc_L2048", "hnorm_ratio_L2048", "grad_ratio_L2048"])
    md = "## H4: Dale-constrained vs unconstrained dense recurrent layer (256 units)\n\n" + markdown_table(an, "val_bpc@128", ["bpc_L512", "bpc_L2048", "hnorm_ratio_L2048", "grad_ratio_L2048"])
    write_summary(EXP, an, md); print(md)
