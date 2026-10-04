"""AIF-H5: EFE decoding on the ask-to-learn environment."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyai.model import FlyAI, FlyAIConfig
from flyai.primitives import param_groups, after_optimizer_step
from flyai.env_ask import transcript_batch, simulate, VOCAB, ASK, EQ, NK
from flyai.decoding import info_gain
from flybrain.harness.prereg import Prereg

from flybrain.harness.stats import compare, summarize
from flybrain.harness.runner import git_commit

EXP = Path(__file__).resolve().parent
B, T = 32, 24


def train_model(seed, steps, lr, run_dir):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = FlyAI(FlyAIConfig(vocab=VOCAB, d=64, n=128, n_kc=512, K_max=1, adaptive_K=False, seed=seed))      # PREREG: K = 1 exactly
    opt = torch.optim.AdamW(param_groups(model, 0.01), lr=lr, betas=(0.9, 0.95)); warm = 20; diverged = False
    for s in range(steps):
        l = lr * ((s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, steps - warm))))
        for g in opt.param_groups: g["lr"] = l
        x, _, _ = transcript_batch(B, rng, T=T)
        logits, _ = model(x); loss = F.cross_entropy(logits[:, 1:].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        if not torch.isfinite(loss): diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); after_optimizer_step(model)
    torch.save(model.state_dict(), run_dir / "model.pt")
    return model, float(loss), diverged


def make_entropy_policy(threshold):
    """Cheap competitor: ask iff the entropy of the model's own ANSWER distribution exceeds `threshold`.
    The answer distribution is the model's prediction at the answer position, i.e. after the '='
    marker: the state is advanced through a hypothetical '=' (functional step, nothing mutated)
    and the value-vocabulary slice of the next-token prediction is read there."""
    def decide(model, st, allowed):
        Bn = st["s1"].shape[0]
        _, st_eq, _ = model.step(st, torch.full((Bn,), EQ, dtype=torch.long))
        mu, _, _, _, _ = model._priors(st_eq)
        logits = model.out(model.out_norm(mu))[:, NK:2 * NK]              # value vocabulary at the answer position
        p = torch.softmax(logits, -1); H = -(p * torch.log(p + 1e-9)).sum(-1)
        return torch.where(H > threshold, ASK, EQ)
    return decide


def make_policy(beta_e):
    def decide(model, st, allowed):
        mu, _, _, _, pi_prior = model._priors(st)
        logits = model.out(model.out_norm(mu)); logp = F.log_softmax(logits, -1)
        G = torch.stack([-logp[:, ASK], -logp[:, EQ]], 1)
        if beta_e:
            ig = torch.stack([info_gain(model, st, mu, pi_prior, torch.full((st["s1"].shape[0],), t, dtype=torch.long)) for t in (ASK, EQ)], 1)
            G = G - beta_e * ig
        return torch.where(G.argmin(-1) == 0, ASK, EQ)
    return decide


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=300); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--episodes", type=int, default=512); args = ap.parse_args()
    torch.set_num_threads(args.threads); Prereg(EXP).freeze_or_check()
    betas = [0.0, 1.0, 3.0, 10.0]; thresholds = [0.5, 1.0, 1.5, 2.0]; results = {}
    for seed in args.seeds:
        rd = EXP / "runs" / f"seed{seed}"; rd.mkdir(parents=True, exist_ok=True)
        mfile = rd / "metrics.json"
        if mfile.exists():
            results[seed] = json.loads(mfile.read_text()); continue
        (rd / "config.json").write_text(json.dumps({"seed": seed, "steps": args.steps, "lr": 3e-3, "episodes": args.episodes, "betas": betas, "thresholds": thresholds}, indent=1))
        (rd / "env.json").write_text(json.dumps({"commit": git_commit(), "prereg_sha256": Prereg(EXP).digest(), "torch": torch.__version__, "threads": torch.get_num_threads()}, indent=1))
        model, loss, diverged = train_model(seed, args.steps, 3e-3, rd); out = {"train_loss": loss, "diverged": diverged}
        if diverged:
            print(f"[seed {seed}] training diverged; no decoding", flush=True); mfile.write_text(json.dumps(out, indent=1)); results[seed] = out; continue
        for be in betas:
            r = simulate(model, make_policy(be), args.episodes, np.random.default_rng(999 + seed)); out[f"beta{be:g}"] = r
            print(f"[seed {seed} beta_e={be:g}] " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
        for th in thresholds:
            r = simulate(model, make_entropy_policy(th), args.episodes, np.random.default_rng(999 + seed)); out[f"entropy{th:g}"] = r
            print(f"[seed {seed} entropy>{th:g}] " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
        mfile.write_text(json.dumps(out, indent=1)); results[seed] = out
    # choose beta on seed 0 by reward, evaluate on the others
    s0 = results[args.seeds[0]]
    if s0.get("diverged"): raise SystemExit("selection seed diverged: nothing to choose beta_e on")
    best = max(betas[1:], key=lambda b: s0[f"beta{b:g}"]["reward"]); best_th = max(thresholds, key=lambda t: s0[f"entropy{t:g}"]["reward"])
    rows = ["| seed | decoder | acc | ask known | ask unknown | reward |", "|---|---|---|---|---|---|"]
    for seed in args.seeds:
        if results[seed].get("diverged"):
            rows.append(f"| {seed} | (training diverged) | | | | |"); continue
        for name, key in (("greedy", "beta0"), (f"efe beta={best:g}", f"beta{best:g}"), (f"entropy>{best_th:g}", f"entropy{best_th:g}")):
            r = results[seed][key]; rows.append(f"| {seed} | {name} | {r['acc']:.3f} | {r['ask_rate_known']:.3f} | {r['ask_rate_unknown']:.3f} | {r['reward']:.3f} |")
    held = [s for s in args.seeds[1:] if not results[s].get("diverged")]
    V = {}
    if held:
        dg = [results[s]["beta0"]["reward"] for s in held]; de = [results[s][f"beta{best:g}"]["reward"] for s in held]
        sep = [results[s][f"beta{best:g}"]["ask_rate_unknown"] - results[s][f"beta{best:g}"]["ask_rate_known"] for s in held]
        dh = [results[s][f"entropy{best_th:g}"]["reward"] for s in held]
        c_greedy = compare(de, dg, higher_is_better=True); c_entropy = compare(de, dh, higher_is_better=True)
        # pre-registered thresholds, evaluated mechanically on the held-out seeds
        a_ok = float(np.mean(sep)) > 0.2
        b_ok = c_greedy["effect_in_sd"] > 2                                 # 'by > 2 seed-SD' (no CI clause in this PREREG)
        c_ok = not (c_entropy["effect"] < 0 and c_entropy["exceeds_seed_spread"])   # at least as high within seed spread
        V = {"held_out_seeds": held, "a_separation": float(np.mean(sep)), "a_ok": a_ok, "b_reward_vs_greedy": c_greedy, "b_ok": b_ok,
             "c_reward_vs_entropy": c_entropy, "c_ok": c_ok, "supported": bool(a_ok and b_ok and c_ok),
             "note": ("epistemic value reduces to output entropy here" if (a_ok and b_ok and c_ok and not (c_entropy["effect"] > 0 and c_entropy["exceeds_seed_spread"])) else "")}
        verdict = (f"\nHeld-out seeds {held}: (a) ask-rate separation (unknown - known) = {V['a_separation']:+.3f} (> 0.2: {a_ok}); "
                   f"(b) reward efe - greedy = {c_greedy['effect']:+.3f} = {c_greedy['effect_in_sd']:+.1f} seed-SD (> 2: {b_ok}); "
                   f"(c) reward efe - entropy-threshold = {c_entropy['effect']:+.3f} (not worse beyond seed spread: {c_ok}). "
                   f"**Verdict: {'SUPPORTED at these seeds, pending replication' if V['supported'] else 'NOT SUPPORTED'}**" + (f" ({V['note']})" if V['note'] else "") + ".")
    else:
        verdict = ""
    md = f"## AIF-H5: ask-to-learn, EFE decoding vs greedy (beta_e chosen on seed {args.seeds[0]} = {best:g})\n\n" + "\n".join(rows) + verdict + "\n"
    (EXP / "SUMMARY.md").write_text(md); (EXP / "summary.json").write_text(json.dumps({"best_beta": best, "best_entropy_threshold": best_th, "verdict": V, "results": {str(k): v for k, v in results.items()}}, indent=1, default=float)); print(md)
