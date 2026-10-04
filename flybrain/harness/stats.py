"""Seed statistics: mean +/- SD, paired bootstrap CIs, and the 'smaller than seed spread' flag."""
from __future__ import annotations

import numpy as np


def summarize(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"), "n": int(len(v)),
            "min": float(v.min()), "max": float(v.max())}


def paired_bootstrap(a, b, n_boot: int = 20000, seed: int = 0) -> dict:
    """CI on mean(a - b) for seed-paired samples (same seeds in the same order)."""
    a = np.asarray(a, float); b = np.asarray(b, float); assert len(a) == len(b)
    d = a - b; rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    return {"diff_mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_diff_gt_0": float((boots > 0).mean())}


def compare(hyp, base, higher_is_better: bool = True, n_boot: int = 20000) -> dict:
    """Compare two seed-vectors: effect in SD units of the pooled seed spread, bootstrap CI,
    and whether the effect exceeds the seed spread at all."""
    h = summarize(hyp); b = summarize(base); pb = paired_bootstrap(hyp, base, n_boot); n = min(h["n"], b["n"])
    spread = float(np.nanmax([h["sd"], b["sd"]])) if n > 1 else float("nan")
    eff = (h["mean"] - b["mean"]) * (1 if higher_is_better else -1)
    finite = np.isfinite(spread) and spread > 0
    return {"hyp": h, "base": b, "n": n, "bootstrap": pb, "effect": float(eff),
            "effect_in_sd": float(eff / spread) if finite else float("nan"),            # undefined (never 'inf') below 2 seeds or at zero spread
            "exceeds_seed_spread": bool(finite and abs(eff) > spread),
            # with fewer than 3 seeds the percentile bootstrap CI is degenerate (it is the sample range), so it cannot count as evidence
            "ci_excludes_0": bool(n >= 3 and (pb["ci95"][0] > 0 or pb["ci95"][1] < 0))}


def best_lr(results: list[dict], condition: str, metric: str, higher_is_better: bool) -> tuple[float, list[dict]]:
    """Pick the LR whose seed-mean is best for a condition (baselines tuned the same way)."""
    by = {}
    for r in results:
        if r["condition"] == condition:
            by.setdefault(r["lr"], []).append(r)
    scored = {lr: np.mean([r[metric] for r in rs]) for lr, rs in by.items()}
    worst = -np.inf if higher_is_better else np.inf                       # a diverged/NaN seed makes that LR the worst, never the winner
    key = lambda lr: worst if np.isnan(scored[lr]) else scored[lr]
    lr = max(scored, key=key) if higher_is_better else min(scored, key=key)
    return lr, sorted(by[lr], key=lambda r: r["seed"])
