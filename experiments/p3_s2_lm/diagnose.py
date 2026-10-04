"""Exploratory diagnostic of a finished prototype run (experiments/p3_s2_lm/prototype.py): what does the far recall depend on?
Evaluates the saved model (a) as trained, (b) with every memory head's decay fixed to 0.9 per token at evaluation (a half-life of
about 7 tokens: the layers keep their local computation and lose their long-range memory), (c) with the attention window shrunk
to a given size, (d) with both; and reports recall by distance to the previous occurrence. Writes diagnostics.json next to the
run's metrics. If shortening the memory leaves far recall within 0.05 of its trained value, the memory layers are not what
recalls far facts in this model (block A's criterion).
Usage: python experiments/p3_s2_lm/diagnose.py --run experiments/p3_s2_lm/prototype_runs/dense_s1 [--window_small 64]"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm  # noqa: E402
from flyprinc.hybrid import StateMemoryLayer, SlidingWindowAttention  # noqa: E402


def shorten_memory(model, alpha0=0.9):
    """Every memory head forgets with a fixed per-token decay alpha0 (0.9: a half-life of about 7 tokens) instead of its learned
    decay, so the layers keep all their local computation and lose only what they carry over long distances. (Zeroing the
    layers outright removes six of the eight layers' computation and says nothing about memory: perplexity explodes.)
    Returns a function that restores the learned decays."""
    import math
    saved = []
    for m in model.mix:
        if isinstance(m, StateMemoryLayer):
            orig = m._gates
            def gated(*args, _orig=orig, _m=m, **kw):
                beta, gamma, lam = _orig(*args, **kw)
                fixed = (-math.log(alpha0) * _m.tau).view(1, -1, *([1] * (lam.ndim - 2))).expand_as(lam)   # alpha = exp(-lam / tau) = alpha0
                return beta, gamma, fixed
            saved.append((m, orig)); m._gates = gated
    return lambda: [setattr(m, "_gates", f) for m, f in saved]


@torch.no_grad()
def gate_statistics(model, T, B, device, max_windows=20):
    """What the trained gates do on real text: per memory layer and head, the median write strength beta, the fraction of
    tokens written strongly (beta > 0.5), and the memory half-life implied by the median decay alpha = exp(-lam / tau)
    (half-life = ln 2 / -ln alpha, in tokens). Collected over `max_windows` evaluation windows."""
    import math
    stats = []; hooks = []
    for li, m in enumerate(model.mix):
        if not isinstance(m, StateMemoryLayer): continue
        rec = {"layer": li, "beta": [], "alpha": []}; stats.append(rec); orig = m._gates
        def gated(*args, _orig=orig, _m=m, _rec=rec, **kw):
            beta, gamma, lam = _orig(*args, **kw)
            tau = _m.tau.view(1, -1, *([1] * (lam.ndim - 2)))
            _rec["beta"].append(beta.detach().float().flatten(0, 1).cpu() if beta.ndim == 3 else beta.detach().float().movedim(1, 0).flatten(1).cpu())
            _rec["alpha"].append(torch.exp(-lam / tau).detach().float().movedim(1, 0).flatten(1).cpu())
            return beta, gamma, lam
        m._gates = gated; hooks.append((m, orig))
    lm.evaluate(model, "validation", T, B, device, probes=None, max_windows=max_windows); model.eval()
    for m, orig in hooks: m._gates = orig
    out = {}
    for rec in stats:
        beta = torch.cat(rec["beta"], -1); alpha = torch.cat(rec["alpha"], -1)                   # (H, samples)
        hl = lambda a: float("inf") if a >= 1 else math.log(2) / -math.log(max(a, 1e-9))
        out[f"layer{rec['layer']}"] = {"beta_median_per_head": [round(float(x), 3) for x in beta.median(-1).values],
                                       "beta_gt_0.5_fraction_per_head": [round(float(x), 3) for x in (beta > 0.5).float().mean(-1)],
                                       "alpha_median_per_head": [round(float(x), 4) for x in alpha.median(-1).values],
                                       "half_life_tokens_per_head": [round(hl(float(x)), 1) for x in alpha.median(-1).values],
                                       "half_life_tokens_p90_per_head": [round(hl(float(x)), 1) for x in alpha.quantile(0.9, dim=-1)]}
    return out


def set_window(model, w):
    saved = [(m, m.w) for m in model.mix if isinstance(m, SlidingWindowAttention)]
    for m, _ in saved: m.w = w
    return lambda: [setattr(m, "w", w0) for m, w0 in saved]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--window_small", type=int, default=64)
    ap.add_argument("--eval_windows", type=int, default=None); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    run = Path(a.run); c = json.loads((run / "config.json").read_text()); lm.set_corpus(c["corpus"])
    model = lm.build_model(c["d_model"], c["n_layers"], c["n_heads"], c["window"], c["attn_every"], c["layer"], tied=True, emb_std=0.02).to(a.device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=a.device)); model.eval()
    probes = lm.load_probes("validation") if (lm.DATA / "probes_validation.npz").exists() else None
    T = c["T"]; B = max(1, 4096 // T); out = {"run": str(run), "corpus": c["corpus"], "arm": c["arm"], "window_trained": c["window"], "window_small": a.window_small}
    conditions = {"as_trained": [], "memory_short": [shorten_memory], f"window_{a.window_small}": [lambda m: set_window(m, a.window_small)],
                  f"memory_short_window_{a.window_small}": [shorten_memory, lambda m: set_window(m, a.window_small)]}
    for name, mods in conditions.items():
        restores = [f(model) for f in mods]
        r = lm.evaluate(model, "validation", T, B, a.device, probes=probes, max_windows=a.eval_windows, by_distance=True); model.eval()
        for undo in reversed(restores): undo()
        out[name] = r
        keys = ["val_ppl"] + [k for k in r if k.endswith("_acc")]
        print(f"{name:28s} " + "  ".join(f"{k.replace('probe_', '').replace('_acc', '')}={r[k]:.3f}" for k in keys if r[k] == r[k]), flush=True)
    out["gates"] = gate_statistics(model, T, B, a.device)
    for layer, g in out["gates"].items():
        print(f"{layer}: median write strength per head {g['beta_median_per_head']}; half-life (tokens) per head, median {g['half_life_tokens_per_head']}, p90 {g['half_life_tokens_p90_per_head']}", flush=True)
    (run / "diagnostics.json").write_text(json.dumps(out, indent=1))
