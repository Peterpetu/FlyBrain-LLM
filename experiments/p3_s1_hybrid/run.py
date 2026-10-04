"""P3-S1 (rung 1, v3): the two-store memory layer with vs without per-address states inside a trained sequence model,
on the dense rung-1 task where memory-only models learn reliably (see PREREG.md, "Why v3"). GPU-ready;
`--smoke` and `--probe` run in a scratch directory and never freeze the pre-registration."""
from __future__ import annotations

import argparse, json, math, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyprinc.hybrid import HybridConfig, HybridLM, LayerConfig, count_params, layer_cost, chunk_error
from flyprinc.lmtasks import LMTaskConfig, STRATA, make_batch, BatchStream
from flybrain.harness import run_grid, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
RATE_SWEEP = (0.15, 0.30)                                                        # outlier rates (the budget rule's reduction (3), applied in advance: PREREG, Cost)
CENTRAL = 0.15                                                                   # ablation and reference arms run at the central rate only
ABLATIONS = ("single_state", "state_slots_only", "state_feats_only", "gdn_x_gate", "attn_only")
RUN = {"device": "cpu", "smoke": False, "compile": False, "data_workers": 3}   # runtime facts, kept out of the stored run config
FULL = dict(d_model=128, n_layers=2, n_heads=4, d_k=32, d_v=32, d_c=8, chunk=8, sense_block=8, T=256, keys=24, p_rebind=0.2,
            B=64, steps=6000, warmup=200, eval_batches=64)                               # 64 x 64 held-out sequences: ~720 re-bound queries (8 batches gave 90)
SMOKE = dict(d_model=32, n_layers=2, n_heads=2, d_k=8, d_v=8, d_c=4, chunk=16, sense_block=16, T=128, keys=8, p_rebind=0.3,
             B=8, steps=6, warmup=2, eval_batches=1)


def arms(S):
    two = dict(n_traces=2)
    dk_single = int(round(S["d_k"] * (2 * S["d_v"] + S["d_c"] + 1) / S["d_v"]))    # one store, state-matched to the two-store state arm (2 dk dv + dk dc + dk values)
    A = {"state": dict(two, states=("nconf", "zc")), "free": dict(two, states=()), "single_wide": dict(states=(), d_k=dk_single),
         "single": dict(states=()), "hebb_column": dict(two, states=("a",), a_floor=False)}
    for b in (0.5, 1.0): A[f"best_fixed_{b}"] = dict(two, states=(), const_gate=((0.2, 0.2, 0.05), (b, b, 0.05)))   # slow store at 0.2, labile at b
    A["single_state"] = dict(states=("nconf", "zc"))                                                  # the states on one store (E2's form)
    A["state_slots_only"] = dict(two, states=("nconf", "zc"), feature_path=False); A["state_feats_only"] = dict(two, states=("nconf", "zc"), slot_path=False)
    A["gdn_x_gate"] = dict(two, states=(), x_only_gate=True)                                             # two stores with Gated DeltaNet's own input-only gate
    A["attn_only"] = dict(states=(), _attn=True)                                                         # reference: attention only (2 layers, full context)
    return A


def build(cfg, S, vocab, device):
    lk = {k: cfg[k] for k in ("states", "d_k", "a_floor", "const_gate", "slot_path", "feature_path", "x_only_gate", "n_traces") if k in cfg}
    layer = LayerConfig(d_model=S["d_model"], n_heads=S["n_heads"], d_k=lk.pop("d_k", S["d_k"]), d_v=S["d_v"], d_c=S["d_c"], chunk=S["chunk"],
                        sense_block=S["sense_block"], compile_chunk=RUN["compile"], **lk)
    hc = HybridConfig(vocab=vocab, d_model=S["d_model"], n_layers=S["n_layers"], n_heads=S["n_heads"], window=S["T"], layer=layer,
                      attn_every=1 if cfg.get("_attn") else 99, pos="rope", tied=False, emb_std=1.0)   # memory-only models (attention only in the reference arm)
    return HybridLM(hc).to(device), layer


def task_for(S, contradiction_free=False):
    return LMTaskConfig(T=S["T"], keys_per_seq=S["keys"], window=S["T"], p_rebind=0.0 if contradiction_free else S["p_rebind"],
                        p_outlier=0.0 if contradiction_free else S["rate"], far_fraction=0.0)


def evaluate(model, S, device, seed):
    rng = np.random.default_rng(seed); out = {}; model.eval()
    with torch.no_grad():
        for name, cf in (("", False), ("cf_", True)):
            task = task_for(S, cf); hits, cnt = {}, {}
            for _ in range(S["eval_batches"]):
                toks, tgt, strat, _ = make_batch(task, S["B"], rng)
                pred = model(torch.as_tensor(toks, device=device)).argmax(-1).cpu().numpy(); ok = pred == tgt; m = tgt >= 0
                for key, sel in [("all", m)] + [(s, m & (strat == i)) for i, s in enumerate(STRATA)]:
                    hits[key] = hits.get(key, 0) + int(ok[sel].sum()); cnt[key] = cnt.get(key, 0) + int(sel.sum())
            for key in hits: out[f"acc_{name}{key}"] = hits[key] / cnt[key] if cnt[key] else float("nan"); out[f"n_{name}{key}"] = cnt[key]
    out["acc_decisive"] = float(np.mean([out["acc_rebound"], out["acc_after_outlier"]]))            # PRIMARY: the strata where history decides (equal-weighted)
    out["acc_inert"] = float(np.mean([out["acc_single"], out["acc_confirmed"]]))                    # where the mechanism should be inert (no-regression)
    model.train(); return out


def dead_fractions(model, tgt):
    fr = {}; m = tgt >= 0
    for i, layer in enumerate(model.memory_layers()):
        if layer.last_feats is None: continue
        fnc, fzc = layer.last_feats; sel = m[:, None, :].expand_as(fnc)
        fr[f"dead_nconf_l{i}"] = float((fnc[sel] == 0).float().mean()); fr[f"dead_zc_l{i}"] = float((fzc[sel] == 0).float().mean()); layer.last_feats = None
    return fr


def voided_runs(res):
    """Runs whose accuracies were voided by a gate or by divergence, with the cause (PREREG.md, Correctness gates)."""
    causes = ("diverged", "dead_feature_stop", "chunk_error_gate_failed")
    return sorted(f"{r['condition']} seed {r['seed']}: " + ", ".join(c for c in causes if r.get(c)) for r in res if any(r.get(c) for c in causes))


def prereg_verdict(an, rates, seeds, res):
    """The pre-registered conjuncts, evaluated mechanically over the rates that ran (PREREG.md, Thresholds). An undefined
    comparison (a voided run) passes no conjunct; the verdict needs every pre-registered seed of every arm at every rate."""
    def cmp(tag, ctl): return (an.get(tag) or {}).get("comparisons", {}).get(f"{ctl}|{tag}")
    passA = sorted(t for t in an if (c := cmp(t, "free")) and c["effect_in_sd"] > 2 and c["ci_excludes_0"])
    passB = sorted(t for t in an if (c := cmp(t, "single_wide")) and c["effect"] > 0 and c["ci_excludes_0"])
    A = len(passA) == len(rates); Bc = len(passB) == len(rates)                         # every remaining rate (the sweep was reduced to two)
    regress = [(t, m) for t in an for m in ("acc_inert", "acc_cf_all") if (c := an[t].get(f"noreg_{m}")) and c["effect_in_sd"] < -1.0]
    Cc = not regress
    short = sorted(f"{c} ({row['n_seeds']} seeds)" for a in an.values() for c, row in a["table"].items() if sorted(row["seeds"]) != sorted(seeds))
    voided = voided_runs(res); complete = len(an) == len(rates) and not short
    return {"A_state_beats_free_gt2sd_ci_at_every_rate": A, "A_passing": passA, "B_state_beats_single_wide_ci_at_every_rate": Bc, "B_passing": passB,
            "C_no_regression_gt1sd_on_inert_or_cf": Cc, "C_regressions": regress, "settings_analysed": len(an), "arms_missing_seeds": short,
            "voided_runs": voided, "grid_complete": complete,
            "verdict": ("SUPPORTED at the discovery seeds (pending replication)" if (A and Bc and Cc) else "FALSIFIED") if complete else "INCOMPLETE (not all settings ran)"}


def train_fn(cfg, seed, run_dir):
    S = cfg["_S"]; device = RUN["device"]; torch.manual_seed(seed); np.random.seed(seed)
    task = task_for(S); model, layer = build(cfg, S, task.vocab, device); t0 = time.time(); log = open(run_dir / "log.jsonl", "w")
    if device != "cpu": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / S["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, S["steps"])))))
    diverged = False; hist = []; accs = []; step_times = []; dead = {}; dead_stop = False; to90 = None
    has_states = bool(set(cfg.get("states", ())) & {"nconf", "zc"})
    stream = iter(BatchStream(task, S["B"], seed + 1, S["steps"], workers=RUN["data_workers"]))   # batch s drawn with its own generator (seed + 1, s)
    for s in range(S["steps"]):
        toks, tgt, _, _ = next(stream); x, y = torch.as_tensor(toks, device=device), torch.as_tensor(tgt, device=device)
        mon = (s + 1) % 100 == 0 or s == 0
        for ml in model.memory_layers(): ml.monitor = mon
        ts = time.time(); logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        if not torch.isfinite(loss): diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if device != "cpu": torch.cuda.synchronize()
        step_times.append(time.time() - ts); hist.append(loss.item()); accs.append((logits[m].argmax(-1) == y[m]).float().mean().item())
        if mon:
            dead = dead_fractions(model, y); ta = float(np.mean(accs[-100:]))
            log.write(json.dumps({"step": s + 1, "loss": hist[-1], "train_acc": ta, "step_s": step_times[-1], **dead}) + "\n"); log.flush()
            if ta >= 0.9 and to90 is None and s >= 99: to90 = s + 1           # learning speed (reported): first logged step with 100-step training accuracy >= 0.9
            if has_states and s + 1 >= 500 and dead and max(dead.values()) > 0.9: dead_stop = True; break     # gate (iii): a dead state feature is a stop condition
    stream.close()                                                              # stops the data workers (also after an early stop)
    out = {"diverged": diverged, "dead_feature_stop": dead_stop, "train_loss_last": float(np.mean(hist[-100:])) if hist else float("nan"),
           "train_acc_last": float(np.mean(accs[-100:])) if accs else float("nan"), "steps_done": len(hist),
           "steps_to_0.9": float(to90) if to90 is not None else float("nan"),
           "step_s_median": float(np.median(step_times)) if step_times else float("nan"), **{k: v for k, v in (dead if hist else {}).items()}}
    out.update(evaluate(model, S, device, seed=777))
    if model.memory_layers():                                                   # gate (ii): the trained chunked layers reproduce their sequential definition
        toks, _, _, _ = make_batch(task, 4, np.random.default_rng(999)); xx = torch.as_tensor(toks, device=device); errs = []
        with torch.no_grad():
            h = model.emb(xx)
            for i, mm in enumerate(model.mix):
                hn = model.n1[i](h)
                if model.kind[i] == "attn": h = h + mm(hn)
                else: errs.append(chunk_error(mm, hn)); h = h + mm(hn)[0]
                h = h + model.mlp[i](model.n2[i](h))
        out["chunk_error_max"] = max(errs); out["chunk_error_gate_failed"] = bool(max(errs) > 0.01)
    if diverged or dead_stop or out.get("chunk_error_gate_failed"):
        for k in list(out):
            if k.startswith("acc"): out[k] = float("nan")
    if device != "cpu":
        torch.cuda.synchronize(); out["gpu_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30; out["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    cost = layer_cost(layer) if model.memory_layers() else {"state_per_layer": 0, "macs_per_head_token": 0}
    out.update({"params": count_params(model), "state_per_layer": cost["state_per_layer"], "macs_per_head_token": cost["macs_per_head_token"], "wall_s": time.time() - t0})
    log.close(); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true", help="toy size in a scratch directory; never freezes PREREG.md")
    ap.add_argument("--probe", action="store_true", help="FULL size in a scratch directory; never freezes PREREG.md")
    ap.add_argument("--compile", action="store_true", help="torch.compile each memory layer's per-chunk step (identical arithmetic; needs Triton)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14]); ap.add_argument("--only", nargs="*"); ap.add_argument("--variant", default=None)
    ap.add_argument("--rates", type=float, nargs="*", default=None); ap.add_argument("--lrs", type=float, nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--steps", type=int, default=None); ap.add_argument("--override", default=None)
    ap.add_argument("--claim", action="store_true", help="one of several parallel workers sharing the GPU; afterwards run once without it for the complete summary")
    ap.add_argument("--data_workers", type=int, default=3, help="processes generating training batches ahead (0 = in-line; the batches are identical)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads); S = dict(SMOKE if args.smoke else FULL)
    if args.steps: S["steps"] = args.steps
    if args.override: S.update(json.loads(args.override))
    exp = EXP
    if args.smoke or args.probe:
        kind = "smoke" if args.smoke else "probe"
        exp = Path(os.environ.get("SCRATCH", "/tmp")) / f"p3_s1_{kind}_{os.getpid()}"; exp.mkdir(parents=True); shutil.copy(EXP / "PREREG.md", exp / "PREREG.md"); print(f"{kind} dir", exp)
    if args.smoke: args.seeds = args.seeds[:1]
    rates = args.rates or ((CENTRAL,) if args.smoke else RATE_SWEEP); A = arms(S); C = {}
    for rate in rates:
        for name, a in A.items():
            if args.only and name not in args.only: continue
            if name in ABLATIONS and rate != CENTRAL: continue
            C[f"{name}|r{rate:g}"] = dict(a, _S=dict(S, rate=rate))
    RUN.update(device=args.device, smoke=args.smoke, compile=args.compile, data_workers=args.data_workers)
    lrs = {k: (args.lrs or [1e-3]) for k in C}
    res = run_grid(exp, C, [0.0], args.seeds, train_fn, per_condition_lrs=lrs, variant=args.variant, claim=args.claim)
    an = {}
    for rate in rates:
        tag = f"r{rate:g}"; res_t = [r for r in res if r["condition"].endswith("|" + tag)]
        if f"state|{tag}" in C:
            ctl = [f"{n}|{tag}" for n in A if n != "state" and f"{n}|{tag}" in C]
            an[tag] = analyse(res_t, "acc_decisive", f"state|{tag}", ctl, extra_metrics=["acc_inert", "acc_all", "acc_rebound", "acc_after_outlier", "acc_cf_all", "train_acc_last", "steps_to_0.9"])
            if f"free|{tag}" in C:
                for m in ("acc_inert", "acc_cf_all"): an[tag][f"noreg_{m}"] = analyse(res_t, m, f"state|{tag}", [f"free|{tag}"])["comparisons"].get(f"free|{tag}")
    verdict = prereg_verdict(an, rates, args.seeds, res)
    md = "## P3-S1 (v3): two-store memory with vs without the per-address states, memory-only models on the dense rung-1 task (primary: accuracy on the decisive strata)\n\n"
    md += "\n\n".join(f"### outlier rate {tag[1:]}\n\n" + markdown_table(a, "acc_decisive", ["acc_inert", "acc_rebound", "acc_after_outlier", "acc_cf_all", "train_acc_last", "steps_to_0.9"]) for tag, a in an.items())
    md += "\n\n### Pre-registered verdict (computed)\n\n" + "\n".join(f"- {k}: {v}" for k, v in verdict.items()) + "\n"
    top = an.get(f"r{CENTRAL:g}") or (an[next(iter(an))] if an else {"table": {}, "comparisons": {}})
    write_summary(exp, {"per_setting": an, "verdict": verdict, **top}, md, variant=args.variant); print(md)
