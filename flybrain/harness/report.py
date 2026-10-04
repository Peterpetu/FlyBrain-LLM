"""Turn a grid of runs into the pre-registered comparison table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .stats import best_lr, compare, summarize


def analyse(results: list[dict], metric: str, hypothesis: str, controls: list[str], higher_is_better: bool = True,
            extra_metrics: list[str] | None = None) -> dict:
    conds = sorted({r["condition"] for r in results})
    table = {}
    for cnd in conds:
        lr, runs = best_lr(results, cnd, metric, higher_is_better)
        row = {"lr": lr, "n_seeds": len(runs), "seeds": [r["seed"] for r in runs], metric: summarize([r[metric] for r in runs])}
        for m in (extra_metrics or []):
            if all(m in r for r in runs):
                row[m] = summarize([r[m] for r in runs])
        row["lrs_swept"] = sorted({r["lr"] for r in results if r["condition"] == cnd})
        row["diverged"] = int(sum(r.get("diverged", False) for r in results if r["condition"] == cnd))
        table[cnd] = row
    comps = {}
    if hypothesis in table:
        _, hyp_runs = best_lr(results, hypothesis, metric, higher_is_better)
        for cnd in controls:
            if cnd not in table:
                continue
            _, ctl_runs = best_lr(results, cnd, metric, higher_is_better)
            hb = {r["seed"]: r for r in hyp_runs}; cb = {r["seed"]: r for r in ctl_runs}
            common = sorted(set(hb) & set(cb))                       # paired BY SEED ID, never by list position
            comps[cnd] = compare([hb[sd][metric] for sd in common], [cb[sd][metric] for sd in common], higher_is_better)
            comps[cnd]["seeds"] = common
    return {"metric": metric, "hypothesis": hypothesis, "table": table, "comparisons": comps,
            "missing_controls": [c for c in controls if c not in table]}


def markdown_table(an: dict, metric_label: str | None = None, extra: list[str] | None = None) -> str:
    m = an["metric"]; lab = metric_label or m
    hdr = ["condition", "best LR", "seeds", f"{lab} mean +/- SD"] + [e for e in (extra or [])] + ["vs hypothesis: diff [95% CI]", "effect / seed-SD"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for cnd, row in an["table"].items():
        s = row[m]
        cells = [cnd.replace("|", "\\|"),                                  # grid keys such as "arm|r0.15" must not split the cell
                 f"{row['lr']:g}", str(row["n_seeds"]), f"{s['mean']:.4f} +/- {s['sd']:.4f}"]
        for e in (extra or []):
            cells.append(f"{row[e]['mean']:.4f} +/- {row[e]['sd']:.4f}" if e in row else "")
        c = an["comparisons"].get(cnd)
        if c:
            lo, hi = c["bootstrap"]["ci95"]
            cells += [f"{c['bootstrap']['diff_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]", f"{c['effect_in_sd']:+.1f}"]
        else:
            cells += ["(hypothesis)" if cnd == an["hypothesis"] else "", ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary(exp_dir: Path, an: dict, md: str, variant: str | None = None):
    exp_dir = Path(exp_dir); suf = "" if variant is None else f"_{variant}"
    (exp_dir / f"summary{suf}.json").write_text(json.dumps(an, indent=1, default=float))
    (exp_dir / f"SUMMARY{suf}.md").write_text(md)
