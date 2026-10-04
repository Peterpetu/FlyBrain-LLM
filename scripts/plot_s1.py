"""Rung-1 figures from the finished runs (reads experiments/p3_s1_hybrid/runs/, writes experiments/p3_s1_hybrid/figures/):
(1) the primary metric (held-out accuracy on the decisive strata) for every arm, one panel per outlier rate, every seed as a
point and the seed mean as a bar, the three verdict arms first; (2) accuracy per stratum (re-bound, after an outlier, single,
confirmed) for the three verdict arms; (3) learning curves (100-step training accuracy against step), mean over seeds with
every seed faint. Every label states how many seeds it holds, so a figure drawn from a partial grid says so.
Usage: python scripts/plot_s1.py [--variant replication]"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1] / "experiments" / "p3_s1_hybrid"
STRATA = (("acc_rebound", "re-bound"), ("acc_after_outlier", "after an outlier"), ("acc_single", "single"), ("acc_confirmed", "confirmed"))
LABEL = {"state": "two stores + states (hypothesis)", "free": "two stores, states zeroed (control)", "single_wide": "one store, equal memory (control)",
         "single": "one store", "hebb_column": "two stores + Hebbian column", "single_state": "one store + states",
         "state_slots_only": "states: slots path only", "state_feats_only": "states: features path only",
         "gdn_x_gate": "two stores, input-only gate", "best_fixed_0.5": "two stores, fixed gates 0.5", "best_fixed_1.0": "two stores, fixed gates 1.0",
         "attn_only": "attention only (reference)"}
ORDER = list(LABEL); VERDICT = ("state", "free", "single_wide")


def load(variant):
    root = EXP / ("runs" if variant is None else f"runs_{variant}"); runs = defaultdict(list)
    for mf in sorted(root.glob("*/lr*/seed*/metrics.json")):
        m = json.loads(mf.read_text()); arm, rate = m["condition"].split("|r")
        curve = [json.loads(l) for l in open(mf.parent / "log.jsonl")] if (mf.parent / "log.jsonl").exists() else []
        runs[(arm, float(rate))].append({"m": m, "curve": curve})
    return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default=None); a = ap.parse_args()
    runs = load(a.variant); rates = sorted({r for _, r in runs}); out = EXP / "figures"; out.mkdir(exist_ok=True)
    if not runs: raise SystemExit("no finished runs")
    suf = "" if a.variant is None else f"_{a.variant}"; colors = dict(zip(ORDER, list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors)))

    # (1) the primary metric, every arm
    fig, axes = plt.subplots(1, len(rates), figsize=(max(8, 6 * len(rates)), 4.8), squeeze=False)
    for ax, rate in zip(axes[0], rates):
        arms = [x for x in ORDER if (x, rate) in runs]
        for i, arm in enumerate(arms):
            v = np.array([r["m"]["acc_decisive"] for r in runs[(arm, rate)]], dtype=float); y = len(arms) - 1 - i
            ax.barh(y, np.nanmean(v), color=colors[arm], alpha=0.35 if arm in VERDICT else 0.18, height=0.7)
            ax.scatter(v, np.full(len(v), y), color=colors[arm], s=18, zorder=3, edgecolor="k", linewidth=0.3)
            ax.text(1.01, y, f"{np.nanmean(v):.3f} (n={len(v)})", va="center", fontsize=7)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([LABEL[x] for x in reversed(arms)], fontsize=7)
        ax.set_xlim(0, 1.15); ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_xlabel("held-out accuracy, decisive strata")
        ax.set_title(f"outlier rate {rate}"); ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Rung 1: the primary metric by arm (bar = seed mean, points = seeds; the verdict compares the first three arms)"); fig.tight_layout()
    fig.savefig(out / f"decisive_accuracy{suf}.png", dpi=130); plt.close(fig)

    # (2) per stratum, the verdict arms
    fig, axes = plt.subplots(1, len(rates), figsize=(max(8, 5.5 * len(rates)), 4.2), squeeze=False)
    for ax, rate in zip(axes[0], rates):
        arms = [x for x in VERDICT if (x, rate) in runs]
        for j, arm in enumerate(arms):
            A = np.array([[r["m"].get(k, np.nan) for k, _ in STRATA] for r in runs[(arm, rate)]], dtype=float)
            xs = np.arange(len(STRATA)) + (j - (len(arms) - 1) / 2) * 0.12
            for row in A: ax.scatter(xs, row, color=colors[arm], alpha=0.4, s=12)
            ax.plot(xs, np.nanmean(A, axis=0), color=colors[arm], lw=1.8, marker="o", ms=5, label=f"{LABEL[arm]} (n={len(A)})")
        ax.set_xticks(range(len(STRATA))); ax.set_xticklabels([n for _, n in STRATA])
        ax.set_title(f"outlier rate {rate}"); ax.set_xlabel("stratum of the query"); ax.set_ylim(0.6, 1.01); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    axes[0][0].set_ylabel("held-out accuracy")
    fig.suptitle("Rung 1: accuracy per stratum for the verdict arms (seed mean joined, every seed as a point)"); fig.tight_layout()
    fig.savefig(out / f"accuracy_by_stratum{suf}.png", dpi=130); plt.close(fig)

    # (3) learning curves, every arm
    fig, axes = plt.subplots(1, len(rates), figsize=(max(8, 6 * len(rates)), 4.5), squeeze=False)
    for ax, rate in zip(axes[0], rates):
        for arm in [x for x in ORDER if (x, rate) in runs]:
            curves = [r["curve"] for r in runs[(arm, rate)] if r["curve"]]
            if not curves: continue
            steps = [c["step"] for c in curves[0]]; M = np.array([[c["train_acc"] for c in cur[:len(steps)]] for cur in curves if len(cur) >= len(steps)])
            for row in M: ax.plot(steps, row, color=colors[arm], alpha=0.12, lw=0.7)
            ax.plot(steps, M.mean(0), color=colors[arm], lw=1.6 if arm in VERDICT else 1.0, label=f"{LABEL[arm]} (n={len(M)})")
        ax.set_title(f"outlier rate {rate}"); ax.set_xlabel("training step"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="lower right")
    axes[0][0].set_ylabel("training accuracy (100-step mean)")
    fig.suptitle("Rung 1: learning curves (mean over seeds, each seed faint)"); fig.tight_layout()
    fig.savefig(out / f"learning_curves{suf}.png", dpi=130); plt.close(fig)
    print(f"figures -> {out} ({sum(len(v) for v in runs.values())} runs, rates = {rates})")
