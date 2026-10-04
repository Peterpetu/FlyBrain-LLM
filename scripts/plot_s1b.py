"""Rung-1b figures from the finished runs (reads experiments/p3_s1b_sparse/runs/, writes experiments/p3_s1b_sparse/figures/):
(1) held-out recall accuracy by memory load (keys already stored when a query comes), one panel per P, one line per arm, mean
over seeds with every seed shown faintly; (2) learning curves (100-step training accuracy against step), mean over seeds.
Every legend entry states how many seeds it holds, so a figure drawn from a partial grid says so.
Usage: python scripts/plot_s1b.py [--variant replication]"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1] / "experiments" / "p3_s1b_sparse"
LOADS = (0, 16, 32, 64, 128)                                                  # run.py's LOAD_BINS lower edges
LABEL = {"dense": "Gated DeltaNet (dense keys)", "sparse": "Kenyon-cell address", "dense_c": "dense, equal compute", "dense_wide": "dense, equal memory",
         "product": "product keys", "sparse_learned": "learned expansion", "expand_dense": "expansion without sparsity",
         "full_dense": "full layer, dense", "full_sparse": "full layer, Kenyon-cell", "attn_only": "attention only"}
ORDER = list(LABEL)


def load(variant):
    root = EXP / ("runs" if variant is None else f"runs_{variant}"); runs = defaultdict(list)
    for mf in sorted(root.glob("*/lr*/seed*/metrics.json")):
        m = json.loads(mf.read_text()); arm, P = m["condition"].split("|P")
        curve = [json.loads(l) for l in open(mf.parent / "log.jsonl")] if (mf.parent / "log.jsonl").exists() else []
        runs[(arm, int(P))].append({"m": m, "curve": curve})
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default=None); a = ap.parse_args()
    runs = load(a.variant); Ps = sorted({P for _, P in runs}); out = EXP / "figures"; out.mkdir(exist_ok=True)
    if not runs: raise SystemExit("no finished runs")
    suf = "" if a.variant is None else f"_{a.variant}"; colors = dict(zip(ORDER, plt.cm.tab10.colors))

    fig, axes = plt.subplots(1, len(Ps), figsize=(max(8, 5 * len(Ps)), 4), squeeze=False)
    for ax, P in zip(axes[0], Ps):
        for arm in [x for x in ORDER if (x, P) in runs]:
            A = np.array([[r["m"].get(f"acc_load_{b}", np.nan) for b in LOADS] for r in runs[(arm, P)]], dtype=float)
            keep = ~np.all(np.isnan(A), axis=0); xs = np.arange(len(LOADS))[keep]
            for row in A: ax.plot(xs, row[keep], color=colors[arm], alpha=0.15, lw=0.8)
            ax.plot(xs, np.nanmean(A[:, keep], axis=0), color=colors[arm], lw=2, marker="o", ms=3, label=f"{LABEL[arm]} (n={len(A)})")
        ax.set_xticks(range(len(LOADS))); ax.set_xticklabels(["<16", "16-31", "32-63", "64-127", "128+"])
        ax.set_title(f"P = {P} keys"); ax.set_xlabel("keys already stored at the query"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="lower left")
    axes[0][0].set_ylabel("held-out recall accuracy")
    fig.suptitle("Rung 1b: recall by memory load (key width 16; mean over seeds, each seed faint)"); fig.tight_layout()
    fig.savefig(out / f"accuracy_by_load{suf}.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, len(Ps), figsize=(max(8, 5 * len(Ps)), 4), squeeze=False)
    for ax, P in zip(axes[0], Ps):
        for arm in [x for x in ORDER if (x, P) in runs]:
            curves =[r["curve"] for r in runs[(arm, P)] if r["curve"]]
            if not curves: continue
            steps = [c["step"] for c in curves[0]]; M = np.array([[c["train_acc"] for c in cur[:len(steps)]] for cur in curves if len(cur) >= len(steps)])
            for row in M: ax.plot(steps, row, color=colors[arm], alpha=0.15, lw=0.8)
            ax.plot(steps, M.mean(0), color=colors[arm], lw=2, label=f"{LABEL[arm]} (n={len(M)})")
        ax.set_title(f"P = {P} keys"); ax.set_xlabel("training step"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="center right")
    axes[0][0].set_ylabel("training accuracy (100-step mean)")
    fig.suptitle("Rung 1b: learning curves (mean over seeds, each seed faint)"); fig.tight_layout()
    fig.savefig(out / f"learning_curves{suf}.png", dpi=130); plt.close(fig)
    print(f"figures -> {out} ({sum(len(v) for v in runs.values())} runs, P = {Ps})")
