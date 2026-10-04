#!/usr/bin/env python3
"""Figures from experiment summaries: H5 scaling curve, H1 MQAR bars, H2 extrapolation."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]; OUT = R / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)


def h5():
    p = R / "experiments/h5_graph_mask/summary.json"
    if not p.exists(): return
    S = json.loads(p.read_text()); sizes = sorted(int(k[1:]) for k in S)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for kind, lab in [("real", "real connectome core"), ("shuffle", "degree-preserving shuffle"), ("er", "Erdos-Renyi"), ("dense_param", "dense, param-matched"), ("dense_N", "dense, same N")]:
        ys = [S[f"N{n}"]["table"][f"{kind}_N{n}"]["val_bpc_last2"] for n in sizes]
        ax.errorbar(sizes, [y["mean"] for y in ys], yerr=[y["sd"] for y in ys], marker="o", capsize=3, label=lab)
    ax.set_xscale("log", base=2); ax.set_xlabel("recurrent units N"); ax.set_ylabel("val bits/char (lower better)"); ax.set_title("H5: graph as recurrent mask, TinyShakespeare")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "h5_scaling.png", dpi=160)


def h1():
    p = R / "experiments/h1_mushroom_body/summary.json"
    if not p.exists(): return
    S = json.loads(p.read_text()); rows = {**S["table"], **S["h1a"]["table"]}
    names = list(rows); fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, m in enumerate(["acc_in_last2", "acc_pairs16_T128"]):
        ax.bar([j + i * 0.4 for j in range(len(names))], [rows[n][m]["mean"] for n in names], 0.4, yerr=[rows[n][m]["sd"] for n in names], capsize=2, label={"acc_in_last2": "T=64, 8 pairs", "acc_pairs16_T128": "T=128, 16 pairs"}[m])
    ax.set_xticks([j + 0.2 for j in range(len(names))]); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7); ax.set_ylabel("MQAR accuracy"); ax.axhline(1 / 21, ls="--", c="gray", lw=0.8)
    ax.set_title("H1: mushroom-body memory vs fast-weight baselines"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "h1_mqar.png", dpi=160)


def h2():
    p = R / "experiments/h2_ring_attractor/summary.json"
    if not p.exists(): return
    S = json.loads(p.read_text()); fig, ax = plt.subplots(figsize=(5.5, 3.6)); Ls = [32, 128, 256, 512]
    for cnd, row in S["table"].items():
        ys = [row[f"acc_T{L}_last2"] if L != 128 else row["acc_T128_last2"] for L in Ls]
        ax.errorbar(Ls, [y["mean"] for y in ys], yerr=[y["sd"] for y in ys], marker="o", capsize=3, label=cnd)
    ax.axvline(32, ls=":", c="gray"); ax.axhline(0.125, ls="--", c="gray", lw=0.8); ax.set_xscale("log", base=2); ax.set_xlabel("sequence length (train = 32)"); ax.set_ylabel("heading accuracy")
    ax.set_title("H2: compass state tracking, length extrapolation"); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "h2_extrapolation.png", dpi=160)


def aif():
    """One bar chart per AIF experiment: primary metric by condition, mean +/- SD over seeds."""
    specs = [("aif_h1_inference", "val_bpc_last2", "val bits/char (lower better)"), ("aif_h2_precision", "acc_post_last2", "post-rebind recall accuracy"),
             ("aif_h3_arousal", "val_bpc_last2", "val bits/char (lower better)"), ("aif_h4_criticality", "acc_T64_last2", "retention accuracy @64"),
             ("aif_h6_windtunnel", "kl_bits_last2", "KL to exact posterior (bits, lower better)")]
    for name, metric, label in specs:
        p = R / "experiments" / name / "summary.json"
        if not p.exists(): continue
        S = json.loads(p.read_text()); rows = S["table"]; names = list(rows)
        fig, ax = plt.subplots(figsize=(max(4, 1.1 * len(names)), 3.4))
        ax.bar(range(len(names)), [rows[n][metric]["mean"] for n in names], yerr=[rows[n][metric]["sd"] for n in names], capsize=3, color=["#3B6EA5" if n == S["hypothesis"] else "#9AA5B1" for n in names])
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8); ax.set_ylabel(label, fontsize=8); ax.set_title(name, fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    h5(); h1(); h2(); aif(); print("figures in", OUT)
