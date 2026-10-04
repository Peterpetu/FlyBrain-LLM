"""P3-S1b (rung 1b, v2): the Kenyon-cell sparse address inside trained memory-only sequence models, in the capacity
form of the dense recall task (P distinct keys, each presented 2-5 times and queried near a presentation, while the
other keys load the same memory). See PREREG.md. Same training loop, gates and accounting as rung 1 v3
(experiments/p3_s1_hybrid); `--smoke` and `--probe` never freeze PREREG.md."""
from __future__ import annotations

import argparse, json, math, os, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flyprinc.hybrid import HybridConfig, HybridLM, LayerConfig, count_params, layer_cost, chunk_error
from flyprinc.lmtasks import LMTaskConfig, make_batch, BatchStream
from flybrain.harness import run_grid, analyse, markdown_table, write_summary

EXP = Path(__file__).resolve().parent
P_SWEEP = (32, 64, 128)                                                         # stored keys per sequence (T = tokens_per_key x P)
CENTRAL = 64                                                                    # the secondary and reference arms run at the central P only
SECONDARY = ("sparse_learned", "expand_dense", "full_dense", "full_sparse", "attn_only")
MARGIN = 0.02                                                                   # equivalence margin (accuracy) for MATCHES in the reported comparisons
LOAD_BINS = (0, 16, 32, 64, 128, 10**9)                                         # keys already presented when a query comes (reported accuracy by load)
RUN = {"device": "cpu", "smoke": False, "compile": False, "data_workers": 3}   # runtime facts, kept out of the stored run config
FULL = dict(d_model=128, n_layers=2, n_heads=4, d_k=16, d_v=32, d_c=8, chunk=64, block=16, n_cells=256, k_active=16, fan_in=4, pk_n1=16,
            tokens_per_key=8, B=64, steps=6000, warmup=200, eval_batches=8)                # chunk: Gated DeltaNet-gated arms; block: the full-layer arms' sensing block
                                                                                              # d_k = 16 chosen by the PREREG's rule (at d_k = 32 the dense store was not the bottleneck)
SMOKE = dict(d_model=32, n_layers=2, n_heads=2, d_k=8, d_v=8, d_c=4, chunk=32, block=16, n_cells=64, k_active=8, fan_in=3, pk_n1=8,
             tokens_per_key=8, B=4, steps=6, warmup=2, eval_batches=1)


def compute_matched_dk(S):
    """The dense key width whose per-token arithmetic (Gated DeltaNet gating) equals the sparse arm's with a kernel that touches
    only the active rows and gathers the fixed expansion (layer_cost's macs_active_per_head_token; linear in d_k for dense keys)."""
    base = dict(d_model=S["d_model"], n_heads=S["n_heads"], d_v=S["d_v"], d_c=S["d_c"], states=(), x_only_gate=True, tie_gamma=True)
    target = layer_cost(LayerConfig(**base, d_k=S["d_k"], address="sparse", n_cells=S["n_cells"], k_active=S["k_active"], fan_in=S["fan_in"]))["macs_active_per_head_token"]
    m0, m1 = (layer_cost(LayerConfig(**base, d_k=d))["macs_active_per_head_token"] for d in (S["d_k"], S["d_k"] + 1))
    return int(round(S["d_k"] + (target - m0) / (m1 - m0)))


def arms(S):
    sp = dict(address="sparse", n_cells=S["n_cells"], k_active=S["k_active"], fan_in=S["fan_in"])
    gdn = dict(states=(), x_only_gate=True, tie_gamma=True, sense_block=0, chunk=S["chunk"])   # Gated DeltaNet's gating: write, erase (= write) and decay from the input
                                                                                  # only; exact at any chunk size, so the chunk is a speed choice
    full = dict(states=(), sense_block=S["block"], chunk=S["block"])              # this repository's layer: the gate senses the store as of each block start (the block
                                                                                  # is part of the model), erase decoupled; exact in one round with chunk = block
    return {
        "dense": dict(gdn),                                                        # Gated DeltaNet, d_k dense rows: the equal-active-rows, equal-parameter control
        "sparse": dict(gdn, **sp),                                                 # + the Kenyon-cell code (fixed random expansion, k-WTA): THE HYPOTHESIS ARM
        "dense_c": dict(gdn, d_k=compute_matched_dk(S)),                           # dense keys widened to the sparse arm's arithmetic (with a sparse kernel): equal compute
        "dense_wide": dict(gdn, d_k=S["n_cells"]),                                # dense keys of width n_cells: the equal-state control (every row active, wider projections)
        "product": {**gdn, **sp, "address": "product", "pk_n1": S["pk_n1"]},       # + product-key addressing (learned sub-key codebooks, top-k): the closest relative
        "sparse_learned": {**gdn, **sp, "address": "sparse_learned"},             # learned expansion + k-WTA: is the random wiring what matters?
        "expand_dense": {**gdn, **sp, "address": "expand_dense"},                 # the fixed expansion without k-WTA: is it the sparsity?
        "full_dense": dict(full),                                                  # the address comparison repeated in this repository's full layer (the one rung 1 uses)
        "full_sparse": dict(full, **sp),
        "attn_only": dict(states=(), _attn=True),                                  # attention only (2 layers, full context): reference
    }


def build(cfg, S, vocab, T, device):
    lk = {k: cfg[k] for k in ("states", "d_k", "address", "n_cells", "k_active", "fan_in", "pk_n1", "x_only_gate", "tie_gamma") if k in cfg}
    layer = LayerConfig(d_model=S["d_model"], n_heads=S["n_heads"], d_k=lk.pop("d_k", S["d_k"]), d_v=S["d_v"], d_c=S["d_c"], chunk=cfg.get("chunk", S["chunk"]),
                        sense_block=cfg.get("sense_block", 0), compile_chunk=RUN["compile"], **lk)
    hc = HybridConfig(vocab=vocab, d_model=S["d_model"], n_layers=S["n_layers"], n_heads=S["n_heads"], window=T, layer=layer,
                      attn_every=1 if cfg.get("_attn") else 99, pos="rope", tied=False, emb_std=1.0)   # memory-only models (attention only in the reference arm)
    return HybridLM(hc).to(device), layer


def rewire(model, seed):
    """Draw each memory layer's fixed random expansion from the run's seed and the layer's index (the layer's own construction:
    fan_in randomly chosen key dimensions per cell, weight 1/fan_in), so that the seeds sample wirings as well as training
    noise, as E1 drew one wiring per seed. Without it every run and both layers would share the one wiring the layer draws
    by default. A buffer, so the sequential copy of the correctness gate receives it through the state dict."""
    for i, layer in enumerate(model.memory_layers()):
        c = layer.c
        if c.address not in ("sparse", "expand_dense"): continue
        g = torch.Generator().manual_seed(1_000_003 * (seed + 1) + i); E = torch.zeros(c.n_cells, c.d_k)
        for j in range(c.n_cells): E[j, torch.randperm(c.d_k, generator=g)[:c.fan_in]] = 1.0 / c.fan_in
        with torch.no_grad(): layer.E.copy_(E.to(layer.E.device))


def task_for(S):
    """The dense recall generator in its contradiction-free capacity form: P keys, 2-5 consistent presentations each, 1-2 queries near a presentation."""
    P = S["keys"]; T = S["tokens_per_key"] * P
    return LMTaskConfig(T=T, keys_per_seq=P, window=T, p_rebind=0.0, p_outlier=0.0, far_fraction=0.0)


def query_load(toks, tgt, n_keys, Q):
    """For every query target position: the number of distinct keys presented (not queried) before the query."""
    load = np.full(tgt.shape, -1)
    for b in range(toks.shape[0]):
        seen = set(); row = toks[b]
        for p in range(toks.shape[1]):
            if tgt[b, p] >= 0: load[b, p] = len(seen)
            if row[p] < n_keys and (p == 0 or row[p - 1] != Q): seen.add(int(row[p]))
    return load


def evaluate(model, task, S, device, seed):
    rng = np.random.default_rng(seed); hits = cnt = 0; bins = {i: [0, 0] for i in range(len(LOAD_BINS) - 1)}; st = {0: [0, 0], 1: [0, 0]}; model.eval()
    with torch.no_grad():
        for _ in range(S["eval_batches"]):
            toks, tgt, strat, _ = make_batch(task, S["B"], rng)
            pred = model(torch.as_tensor(toks, device=device)).argmax(-1).cpu().numpy(); m = tgt >= 0; ok = (pred == tgt) & m
            hits += int(ok.sum()); cnt += int(m.sum()); load = query_load(toks, tgt, task.n_keys, task.Q)
            for i in bins:
                sel = m & (load >= LOAD_BINS[i]) & (load < LOAD_BINS[i + 1]); bins[i][0] += int(ok[sel].sum()); bins[i][1] += int(sel.sum())
            for s in st: sel = m & (strat == s); st[s][0] += int(ok[sel].sum()); st[s][1] += int(sel.sum())
    model.train()
    out = {"acc": hits / cnt if cnt else float("nan"), "n_queries": cnt,
           "acc_single": st[0][0] / st[0][1] if st[0][1] else float("nan"), "acc_confirmed": st[1][0] / st[1][1] if st[1][1] else float("nan")}
    for i, (h, n) in bins.items(): out[f"acc_load_{LOAD_BINS[i]}"] = h / n if n else float("nan"); out[f"n_load_{LOAD_BINS[i]}"] = n
    return out


def train_fn(cfg, seed, run_dir):
    S = cfg["_S"]; device = RUN["device"]; torch.manual_seed(seed); np.random.seed(seed)
    task = task_for(S); model, layer = build(cfg, S, task.vocab, task.T, device); rewire(model, seed); t0 = time.time(); log = open(run_dir / "log.jsonl", "w")
    if device != "cpu": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / S["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, S["steps"])))))
    diverged = False; hist = []; accs = []; step_times = []; to90 = None
    stream = iter(BatchStream(task, S["B"], seed + 1, S["steps"], workers=RUN["data_workers"]))   # batch s drawn with its own generator (seed + 1, s)
    for s in range(S["steps"]):
        toks, tgt, _, _ = next(stream); x, y = torch.as_tensor(toks, device=device), torch.as_tensor(tgt, device=device)
        ts = time.time(); logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        if not torch.isfinite(loss): diverged = True; break
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if device != "cpu": torch.cuda.synchronize()
        step_times.append(time.time() - ts); hist.append(loss.item()); accs.append((logits[m].argmax(-1) == y[m]).float().mean().item())
        if (s + 1) % 100 == 0 or s == 0:
            ta = float(np.mean(accs[-100:])); log.write(json.dumps({"step": s + 1, "loss": hist[-1], "train_acc": ta, "step_s": step_times[-1]}) + "\n"); log.flush()
            if ta >= 0.9 and to90 is None and s >= 99: to90 = s + 1           # learning speed (reported): first logged step with 100-step training accuracy >= 0.9
    stream.close()                                                              # stops the data workers (also after an early stop)
    out = {"diverged": diverged, "train_loss_last": float(np.mean(hist[-100:])) if hist else float("nan"),
           "train_acc_last": float(np.mean(accs[-100:])) if accs else float("nan"), "steps_done": len(hist),
           "steps_to_0.9": float(to90) if to90 is not None else float("nan"),
           "step_s_median": float(np.median(step_times)) if step_times else float("nan")}
    out.update(evaluate(model, task, S, device, seed=777))
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
    if diverged or out.get("chunk_error_gate_failed"):
        for k in list(out):
            if k.startswith("acc"): out[k] = float("nan")
    if device != "cpu":
        torch.cuda.synchronize(); out["gpu_peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30; out["gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    cost = layer_cost(layer) if model.memory_layers() else {}
    out.update({"params": count_params(model), "wall_s": time.time() - t0, **{k: cost[k] for k in ("state_per_layer", "active_rows", "macs_per_head_token", "macs_active_per_head_token") if k in cost}})
    log.close(); return out


def classify(c):
    """Reported comparisons (PREREG.md): BEATS / LOSES when the paired-bootstrap 95% CI excludes 0, MATCHES when it lies within +/- MARGIN."""
    if not c or c["n"] < 3: return "UNDEFINED (fewer than 3 paired seeds)"
    lo, hi = c["bootstrap"]["ci95"]
    if not (np.isfinite(lo) and np.isfinite(hi)): return "UNDEFINED (a voided run)"
    if lo > 0: return "BEATS"
    if hi < 0: return "LOSES"
    if lo >= -MARGIN and hi <= MARGIN: return "MATCHES"
    return "INCONCLUSIVE"


def voided_runs(res):
    causes = ("diverged", "chunk_error_gate_failed")
    return sorted(f"{r['condition']} seed {r['seed']}: " + ", ".join(c for c in causes if r.get(c)) for r in res if any(r.get(c) for c in causes))


def prereg_verdict(an, Ps, seeds, res):
    """Primary (PREREG.md): sparse beats dense by > 2 seed-SD with the CI excluding 0 at >= 2 of 3 P (every P if fewer ran).
    Reported beside it: sparse vs the equal-state dense store and vs product keys at every P."""
    comp = lambda tag, ctl: an.get(tag, {}).get("comparisons", {}).get(f"{ctl}|{tag}")
    passing = sorted(t for t in an if (c := comp(t, "dense")) and c["effect_in_sd"] > 2 and c["ci_excludes_0"])
    need = 2 if len(Ps) >= 3 else len(Ps)
    short = sorted(f"{c} ({row['n_seeds']} seeds)" for a in an.values() for c, row in a["table"].items() if sorted(row["seeds"]) != sorted(seeds))
    complete = len(an) == len(Ps) and not short
    return {"A_sparse_beats_dense_gt2sd_ci_at": passing, "A_needed": need,
            "B_sparse_vs_dense_c_equal_compute": {t: classify(comp(t, "dense_c")) for t in an},
            "C_sparse_vs_dense_wide_equal_memory": {t: classify(comp(t, "dense_wide")) for t in an},
            "D_sparse_vs_product_keys": {t: classify(comp(t, "product")) for t in an},
            "E_full_layer_sparse_vs_dense": {t: classify(a["full_layer"].get(f"full_dense|{t}")) for t, a in an.items() if a.get("full_layer")},
            "arms_missing_seeds": short, "voided_runs": voided_runs(res), "grid_complete": complete,
            "verdict": ("SUPPORTED at the discovery seeds (pending replication)" if len(passing) >= need else "FALSIFIED") if complete else "INCOMPLETE (not all settings ran)"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true", help="toy size in a scratch directory; never freezes PREREG.md")
    ap.add_argument("--probe", action="store_true", help="FULL size in a scratch directory; never freezes PREREG.md")
    ap.add_argument("--compile", action="store_true", help="torch.compile each memory layer's per-chunk step (identical arithmetic; needs Triton)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    ap.add_argument("--only", nargs="*"); ap.add_argument("--variant", default=None); ap.add_argument("--P", type=int, nargs="*", default=None)
    ap.add_argument("--lrs", type=float, nargs="*", default=None); ap.add_argument("--threads", type=int, default=4); ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--override", default=None)
    ap.add_argument("--claim", action="store_true", help="one of several parallel workers sharing the GPU; afterwards run once without it for the complete summary")
    ap.add_argument("--data_workers", type=int, default=3, help="processes generating training batches ahead (0 = in-line; the batches are identical)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads); S = dict(SMOKE if args.smoke else FULL)
    if args.steps: S["steps"] = args.steps
    if args.override: S.update(json.loads(args.override))
    exp = EXP
    if args.smoke or args.probe:
        kind = "smoke" if args.smoke else "probe"
        exp = Path(os.environ.get("SCRATCH", "/tmp")) / f"p3_s1b_{kind}_{os.getpid()}"; exp.mkdir(parents=True); shutil.copy(EXP / "PREREG.md", exp / "PREREG.md"); print(f"{kind} dir", exp)
    if args.smoke: args.seeds = args.seeds[:1]
    Ps = args.P or ((8,) if args.smoke else P_SWEEP); central = Ps[0] if args.smoke else CENTRAL
    C = {}
    for P in Ps:
        for name, a in arms(S).items():
            if args.only and name not in args.only: continue
            if name in SECONDARY and P != central: continue
            C[f"{name}|P{P}"] = dict(a, _S=dict(S, keys=P))
    RUN.update(device=args.device, smoke=args.smoke, compile=args.compile, data_workers=args.data_workers)
    lrs = {k: (args.lrs or [1e-3]) for k in C}
    res = run_grid(exp, C, [0.0], args.seeds, train_fn, per_condition_lrs=lrs, variant=args.variant, claim=args.claim)
    an = {}; extra = ["acc_single", "acc_confirmed", "train_acc_last", "steps_to_0.9", "step_s_median", "gpu_peak_reserved_gb"]
    for P in Ps:
        tag = f"P{P}"; res_t = [r for r in res if r["condition"].endswith("|" + tag)]
        if f"sparse|{tag}" in C:
            ctl = [f"{n}|{tag}" for n in arms(S) if n != "sparse" and f"{n}|{tag}" in C]
            an[tag] = analyse(res_t, "acc", f"sparse|{tag}", ctl, extra_metrics=extra + [f"acc_load_{b}" for b in LOAD_BINS[:-1]])
            if f"full_sparse|{tag}" in C and f"full_dense|{tag}" in C:
                an[tag]["full_layer"] = analyse(res_t, "acc", f"full_sparse|{tag}", [f"full_dense|{tag}"])["comparisons"]
    verdict = prereg_verdict(an, Ps, args.seeds, res)
    md = "## P3-S1b (v2): the Kenyon-cell sparse address in trained memory-only models, capacity form of the dense recall task (primary: held-out recall accuracy)\n\n"
    md += "\n\n".join(f"### {tag}\n\n" + markdown_table(a, "acc", extra) for tag, a in an.items())
    md += "\n\n### Pre-registered verdict (computed)\n\n" + "\n".join(f"- {k}: {v}" for k, v in verdict.items()) + "\n"
    top = an.get(f"P{central}") or (an[next(iter(an))] if an else {"table": {}, "comparisons": {}})
    write_summary(exp, {"per_setting": an, "verdict": verdict, **top}, md, variant=args.variant); print(md)
