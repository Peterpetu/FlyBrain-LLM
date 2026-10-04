#!/usr/bin/env python3
"""Assemble RESULTS.md from experiments/*/summary.json + SUMMARY.md, with each
pre-registered threshold evaluated mechanically next to its outcome."""
from __future__ import annotations

import json, re
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
E = R / "experiments"
VERSION = re.search(r'^version\s*=\s*"([^"]+)"', (R / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)


def load(name):
    p = E / name / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def md(name):
    p = E / name / "SUMMARY.md"
    return p.read_text(encoding="utf-8") if p.exists() else "_not run_\n"


def verdict(c, need_sd=2.0):
    """c = compare() dict. Returns (supported: bool, text). Needs >= 3 paired seeds, a finite effect in
    seed-SD units above `need_sd`, a bootstrap CI excluding 0 and the right sign."""
    if c is None:
        return False, "n/a"
    n = c.get("n", len(c["hyp"].get("seeds", [])) or 3); esd = c["effect_in_sd"]
    ok = n >= 3 and np.isfinite(esd) and esd > need_sd and c["ci_excludes_0"] and c["effect"] > 0
    return ok, f"effect {c['effect']:+.4f} = {esd:+.1f} seed-SD (n={n}), CI {'excludes' if c['ci_excludes_0'] else 'includes'} 0"


def control_label(c):
    if c is None: return "n/a"
    if verdict(c)[0]: return "beyond threshold"
    if c["effect"] < 0 and c["exceeds_seed_spread"]: return "CONTROL better beyond seed spread"
    if within_spread(c): return "within seed spread (null)"
    return "not beyond threshold"


def within_spread(c):
    return c is not None and not c["exceeds_seed_spread"]


out = ["# RESULTS", "",
       f"Version {VERSION} (CHANGELOG.md). "
       "Phase 1-2 experiments and the phase-3 module ladder ran on 4 CPU cores with seeds {0,1,2} unless a table says otherwise; the phase-3 "
       "system-level grids (P3-S1, P3-S1b) ran on one RTX 4080 SUPER (16 GB) with seeds 10-14. Every table row = best learning rate for that "
       "condition by seed-mean of the primary metric; mean +/- SD over seeds; the comparison column is the seed-paired difference (hypothesis minus "
       "control, or as stated) with a 20,000-sample paired bootstrap 95% CI, and the effect in units of the larger seed SD. Differences inside the "
       "seed spread are nulls. Run directories: `experiments/<exp>/runs/<condition>/lr*/seed*/` (not committed; per-run tables in `runs_table.csv` where present).",
       "", "Threshold verdicts below are computed mechanically from summary.json by `scripts/build_results.py`; interpretation is in REPORT.md.", ""]

# ---------------- H5
out += ["## H5 — connectome graph as recurrent mask (calibration)", "", "PREREG: `experiments/h5_graph_mask/PREREG.md`. Primary metric: validation bits/char (lower is better).", ""]
S = load("h5_graph_mask")
if S:
    out.append(md("h5_graph_mask"))
    out += ["### Pre-registered thresholds", ""]
    n_real_gap = 0
    for key, an in S.items():
        N = key[1:]; cmp = an["comparisons"]
        sh = cmp.get(f"shuffle_N{N}"); er = cmp.get(f"er_N{N}"); dp = cmp.get(f"dense_param_N{N}"); dn = cmp.get(f"dense_N_N{N}")
        # compare() was called with higher_is_better=False, so effect>0 means real is better
        real_gap = sh is not None and sh["effect_in_sd"] > 2 and sh["ci_excludes_0"] and sh["effect"] > 0
        n_real_gap += real_gap
        out.append(f"- N={N}: real vs shuffle {verdict(sh)[1]} -> {'real better than shuffle beyond threshold' if real_gap else 'no real-vs-shuffle gap beyond threshold'}; "
                   f"real vs ER {verdict(er)[1]}; real vs dense_param {verdict(dp)[1]}; real vs dense_N {verdict(dn)[1]}.")
    out.append(f"- **Verdict:** real-vs-shuffle gap beyond threshold at {n_real_gap} of {len(S)} sizes (pre-registered: needs >= 2 of 3). "
               + ("**A weak graph prior survives.**" if n_real_gap >= 2 else "**The graph carries no usable information beyond its degree sequence as a recurrent mask at these sizes.**"))
    dense_wins = all(an["comparisons"].get(f"dense_N_N{k[1:]}", {}).get("effect", 0) < 0 for k, an in S.items())
    out.append(f"- Dense same-N layer beats the real graph at every size: {'yes' if dense_wins else 'no'} (pre-registered expectation: yes; a 'no' unlocks H3/H4 follow-up).")
    out.append("")
else:
    out.append("_not run_\n")

# ---------------- H1
out += ["## H1 — mushroom-body memory (sparse expansion, k-WTA, error-gated delta write) on MQAR", "", "PREREG: `experiments/h1_mushroom_body/PREREG.md` (v2; v1 superseded before any cross-condition result, see its change log). Primary metric: MQAR accuracy at T=64 / 8 pairs (chance 0.048).", ""]
S = load("h1_mushroom_body")
if S:
    out.append(md("h1_mushroom_body"))
    out += ["### Pre-registered thresholds", ""]
    c = S["comparisons"]; a = S["h1a"]["comparisons"]; d = S["h1d"]["comparisons"]
    main_ok, main_txt = verdict(c.get("gdn_matched"))
    dense_exp = a.get("abl_dense_expanded"); dense_matches = within_spread(dense_exp)
    out.append(f"- H1 main: mb vs gdn_matched {main_txt}; abl_dense_expanded vs mb: {'matches within seed spread' if dense_matches else ('does not match' if dense_exp else 'n/a')}. "
               f"-> **{'SUPPORTED' if (main_ok and dense_exp and not dense_matches) else 'NOT SUPPORTED'}**.")
    for name, key, text in [("H1a sparse expansion (no-kWTA)", "abl_no_kwta", "falsified if it matches mb"), ("H1a sparse expansion (dense learned projection)", "abl_dense_expanded", "falsified if it matches mb")]:
        cc = a.get(key); out.append(f"- {name}: {verdict(cc)[1]} -> {'**falsified** (' + text + ')' if within_spread(cc) or (cc and cc['effect'] < 0) else 'not falsified (ablation is worse)'}.")
    for name, key in [("H1b error gate (vs learned gate)", "abl_learned_gate"), ("H1c delta/depression write (vs Hebb)", "abl_hebb")]:
        cc = c.get(key); out.append(f"- {name}: {verdict(cc)[1]} -> {'**falsified** (ablation matches or beats mb)' if within_spread(cc) or (cc and cc['effect'] < 0) else 'not falsified (ablation is worse; run at mb LR, so under-tuning cannot be excluded)'}.")
    dd = d.get("mb_shuffled"); ok, txt = verdict(dd)
    out.append(f"- H1d real PN->KC wiring vs degree-preserving shuffle: {txt} -> **{'SUPPORTED' if ok else 'NOT SUPPORTED: the measured pattern carries no extra information beyond its degree sequence here'}**.")
    out.append("")
else:
    out.append("_not run_\n")

# ---------------- H2
out += ["## H2 — central-complex ring attractor for state tracking with length extrapolation", "", "PREREG: `experiments/h2_ring_attractor/PREREG.md`. Primary metric: heading accuracy at 4x training length (T=128; chance 0.125).", ""]
S = load("h2_ring_attractor")
if S:
    out.append(md("h2_ring_attractor")); c = S["comparisons"]
    out += ["### Pre-registered thresholds", ""]
    ok, txt = verdict(c.get("rnn_unstructured"))
    out.append(f"- ring_connectome vs rnn_unstructured (same size, same velocity input): {txt} -> **{'SUPPORTED' if ok else 'NOT SUPPORTED (falsified: an unstructured layer of equal size matches or beats it)' }**.")
    cc = c.get("ring_canonical")
    if cc: out.append(f"- ring_connectome vs ring_canonical (idealised ring, positive control): effect {cc['effect']:+.4f} ({cc['effect_in_sd']:+.1f} seed-SD). " + ("Canonical ring is better: the mechanism works but the measured matrix under our assumptions does not implement it as well." if cc["effect"] < 0 and cc["exceeds_seed_spread"] else "Connectome matrix performs at least as well as the idealised ring."))
    for k in ("rnn_plain", "gdn", "attention"):
        cc = c.get(k)
        if cc: out.append(f"- ring_connectome vs {k}: {verdict(cc)[1]}.")
    out.append("")
else:
    out.append("_not run_\n")

# ---------------- H3 / H4
out += ["## H3 — cell-type dcSBM prior vs Erdos-Renyi", "", "PREREG: `experiments/h3_sbm_prior/PREREG.md`.", ""]
S = load("h3_sbm_prior")
if S:
    out.append(md("h3_sbm_prior")); n_ok = 0
    for key, an in S.items():
        cc = an["comparisons"].get(f"er_N{key[1:]}"); ok, txt = verdict(cc); n_ok += ok
        out.append(f"- N={key[1:]}: sbm vs er {txt}.")
    out.append(f"- **Verdict:** {'SUPPORTED' if n_ok == len(S) else 'NOT SUPPORTED'} (pre-registered: beat ER beyond threshold at both sizes).\n")
else:
    out.append("_not run_\n")
out += ["## H4 — Dale's law / E-I fraction and long-context stability", "", "PREREG: `experiments/h4_dale_stability/PREREG.md`.", ""]
S = load("h4_dale_stability")
if S:
    out.append(md("h4_dale_stability")); c = S["comparisons"]; t = S["table"]
    cu = c.get("unconstrained")
    if cu:
        match128 = within_spread(cu)
        d2048 = {k: t[k]["bpc_L2048"]["mean"] for k in t if "bpc_L2048" in t[k]}
        out.append(f"- dale_f0.3 vs unconstrained at seq 128: {verdict(cu)[1]} -> {'matches within seed spread' if match128 else 'differs'}.")
        out.append(f"- bits/char at seq 2048 by condition: " + ", ".join(f"{k} {v:.3f}" for k, v in d2048.items()) + ".")
        out.append(f"- hidden-norm ratio (last/first 128 positions) at 2048: " + ", ".join(f"{k} {t[k]['hnorm_ratio_L2048']['mean']:.2f}" for k in t if 'hnorm_ratio_L2048' in t[k]) + ".")
    out.append("")
else:
    out.append("_not run_\n")


# ---------------- Phase 2: AIF experiments
out += ["", "# Phase 2 — FlyAI active-inference model", "", "PREREG files: `experiments/aif_h*/PREREG.md`. Same reporting rules; every condition got the same 3-point LR sweep (deviation from the >= 5-point baseline rule, see LIMITATIONS.md).", ""]
def aif_block(title, name, hyp, controls, higher, primary_label, extra=None):
    """One phase-2 experiment: summary table, mechanical threshold verdict (every listed control must
    be beaten; a missing control makes the verdict INCOMPLETE), optional experiment-specific clauses,
    and the replication grid side by side when `summary_replication.json` exists."""
    global out
    out += [f"## {title}", ""]
    S = load(name)
    if not S:
        out.append("_not run_\n"); return
    out.append(md(name)); c = S.get("comparisons", {})
    out += ["### Pre-registered thresholds", ""]
    oks, missing = [], []
    for ctl in controls:
        cc = c.get(ctl)
        if cc is None:
            missing.append(ctl); continue
        ok, txt = verdict(cc); oks.append(ok)
        out.append(f"- {hyp} vs {ctl} ({primary_label}): {txt} -> {control_label(cc)}.")
    clauses = extra(S) if extra else []
    for text, ok in clauses:
        out.append(f"- {text}"); oks.append(ok)
    if missing:
        out.append(f"- **Verdict: INCOMPLETE** (pre-registered control(s) missing from the grid: {', '.join(missing)}).")
    elif oks:
        n_seeds = next((cc.get("n") for cc in c.values() if isinstance(cc, dict) and "n" in cc), "?")
        out.append(f"- **Verdict:** {f'SUPPORTED at the discovery seeds (n = {n_seeds}), PENDING the pre-registered six-fresh-seed replication before it may be reported as supported' if all(oks) else 'NOT SUPPORTED'} (needs every listed control beaten beyond threshold and every clause above true).")
    rep = E / name / "summary_replication.json"
    if rep.exists():
        Rp = json.loads(rep.read_text(encoding="utf-8")); rc = Rp.get("comparisons", {})
        out += ["", "### Replication grid (`runs_replication/`, fresh seeds; reported side by side, never merged)", "", (E / name / "SUMMARY_replication.md").read_text(encoding="utf-8") if (E / name / "SUMMARY_replication.md").exists() else ""]
        roks = []
        for ctl in controls:
            cc = rc.get(ctl)
            if cc is None: continue
            ok, txt = verdict(cc); roks.append(ok); out.append(f"- replication: {hyp} vs {ctl}: {txt} -> {control_label(cc)}.")
        if roks: out.append(f"- **Replication verdict:** {'REPLICATED' if all(roks) else 'NOT REPLICATED'} (discovery seeds {S.get('comparisons', {}).get(controls[0], {}).get('seeds', '?')}, replication seeds {rc.get(controls[0], {}).get('seeds', '?')}).")
    out.append("")


def h2_extra(S):
    pre = S.get("pre_rebind", {})
    cl = []
    for ctl in ("surprise", "learned"):
        cc = pre.get(ctl)
        if cc is None: cl.append((f"pre-rebind guard vs {ctl}: n/a (missing)", False)); continue
        worse = cc["effect"] < 0 and cc["exceeds_seed_spread"]
        cl.append((f"pre-rebind guard: precision vs {ctl} on pre-rebind accuracy {cc['effect']:+.4f} ({cc['effect_in_sd']:+.1f} seed-SD) -> {'WORSE beyond seed spread (guard fails)' if worse else 'not worse beyond seed spread (guard holds)'}", not worse))
    return cl


def h4_extra(S):
    c = S.get("comparisons", {}); cl = []
    f1 = c.get("fixed_1.0")
    if f1 is not None:
        cl.append((f"fixed_1.0 vs regulated: {verdict(f1)[1]} -> {'fixed_1.0 matches regulated: the homeostatic mechanism adds nothing over a well-chosen constant (partial null, pre-registered)' if within_spread(f1) or f1['effect'] < 0 else 'regulated better than fixed_1.0'}", True))
    fc = c.get("fixed_conv")
    if fc is not None:
        cl.append((f"post-hoc fixed_conv (gain fixed at the regulated core's converged value): {verdict(fc)[1]} -> {'regulated at least as good' if not (fc['effect'] < 0 and fc['exceeds_seed_spread']) else 'fixed converged gain BETTER beyond seed spread'}", not (fc["effect"] < 0 and fc["exceeds_seed_spread"])))
    else:
        cl.append(("post-hoc fixed_conv arm: not run", False))
    return cl


def h6_extra(S):
    t = S.get("table", {}); row = t.get("flyai_K2", {}).get("kl_bits_last2")
    if row is None: return [("AIF-H6a absolute threshold: flyai_K2 KL n/a", False)]
    ok = row["mean"] < 0.05
    return [(f"AIF-H6a (belief tracking): flyai_K2 KL = {row['mean']:.4f} +/- {row['sd']:.4f} bits, threshold < 0.05 -> {'met' if ok else 'NOT met'}", ok)]


aif_block("AIF-H1 — per-token free-energy inference vs amortised-only (val bits/char, lower better)", "aif_h1_inference", "flyai_K3", ["flyai_K0", "flyai_K0_wide"], False, "bits/char")
out += ["Notes recorded by the verification pass (docs/aif/VERIFICATION.md): `flyai_K0_wide` (n = 192) is an over-provisioned control, "
        "not a matched one (measured 1.88 vs 1.42 MFLOPs/token and 262k vs 135k parameters against `flyai_K3`); the adaptive-K rule never "
        "stopped early (K_mean = 3.00 in every run), so `flyai_K3` and `flyai_K3_fixed` are the same computation; H1 ran on the code before "
        "the precision bounds and the dithered gain controller were added (both inert or absent in H1's conditions, see VERIFICATION.md).", ""]
aif_block("AIF-H2 — inferred-precision gate under non-stationarity (post-rebind recall)", "aif_h2_precision", "precision", ["surprise", "learned"], True, "post-rebind accuracy", extra=h2_extra)
aif_block("AIF-H3 — arousal from running surprise on topic-switch streams (val bits/char)", "aif_h3_arousal", "arousal", ["no_arousal", "arousal_learned"], False, "bits/char")
aif_block("AIF-H4 — Lyapunov-regulated core gain vs fixed gains (retention accuracy)", "aif_h4_criticality", "regulated", ["fixed_0.5", "fixed_2.0"], True, "accuracy at 64", extra=h4_extra)
aif_block("AIF-H6 — Bayesian wind tunnel: KL to the exact HMM posterior predictive (bits, lower better)", "aif_h6_windtunnel", "flyai_K2", ["flyai_K0"], False, "KL bits", extra=h6_extra)
out += ["## AIF-H5 — expected-free-energy decoding on ask-to-learn", "", "PREREG: `experiments/aif_h5_efe/PREREG.md`. Thresholds (a)-(c) are evaluated mechanically by the driver on the held-out seeds; the verdict line is in the summary below.", ""]
p5 = E / "aif_h5_efe" / "SUMMARY.md"
out.append(p5.read_text(encoding="utf-8") if p5.exists() else "_not run_\n")

# ---------------- Phase 3: principle ladder (module level)
out += ["", "# Phase 3 — principle ladder (module level)", "", "PREREG files: `experiments/p3_*/PREREG.md`; plan: docs/phase3/EXPERIMENT_PLAN.md. The memory module is measured directly on explicit write/query streams; learned controls are trained by backpropagation through the module. Compute is matched by active rows per event; state sizes are in every table.", ""]

def diagnostic_block(title, name, caption):
    """A pre-registered diagnostic (no hypothesis verdict): its SUMMARY.md is embedded verbatim."""
    global out
    out += [f"## {title}", "", caption, ""]
    p = E / name / "SUMMARY.md"
    out.append(p.read_text(encoding="utf-8").split("\n", 1)[1].strip() + "\n" if p.exists() else "_not run_\n")

diagnostic_block("P3-E0 — signal availability of candidate per-address states (diagnostic, no training)", "p3_e0_signal",
                 "PREREG: `experiments/p3_e0_signal/PREREG.md`. Which candidate states carry information about an address's teaching history that a learned gate could not compute from the delta rule's own intermediates. Pre-registered admission rule: incremental 3-way accuracy > 0.05 on sparse keys.")
diagnostic_block("P3-D1 (v2) — persistent re-binding of key-like tokens in text (diagnostic; v1 retracted as an instrument defect, kept in `experiments/p3_d1_corpus/`)", "p3_d1_corpus",
                 "PREREG: `experiments/p3_d1_corpus/PREREG.md`. Does text contain the regime the memory principles serve (an address re-bound beyond the 512-token attention window, with the new binding persisting), against a within-key permutation null.")

def e1_extra(S):
    c = S.get("comparisons", {}); cl = []
    de = c.get("dense_expanded")
    if de is not None: cl.append((f"equal-state control `dense_expanded`: {verdict(de)[1]} -> {'sparse_random better beyond seed spread (the gain is addressing, not state)' if (de['effect'] > 0 and de['exceeds_seed_spread']) else 'matches sparse_random: the gain is state size, not sparse addressing'}", bool(de["effect"] > 0 and de["exceeds_seed_spread"])))
    sl = c.get("sparse_learned")
    if sl is not None: cl.append((f"descriptive `sparse_learned`: {verdict(sl)[1]} -> {'the fixed random expansion is NOT necessary (learned expansion better beyond threshold)' if (sl['effect'] < 0 and sl['exceeds_seed_spread']) else 'a learned expansion does not beat the fixed random one beyond threshold'}", True))
    for k, lab in (("sparse_nonorm", "P9 normalisation ablation"), ("sparse_hebb", "P2 Hebb row")):
        cc = c.get(k)
        if cc is not None: cl.append((f"{lab}: sparse_random vs {k} {verdict(cc)[1]}", True))
    return cl

def e2_extra(S):
    c = S.get("comparisons", {}); cl = []; best = S.get("best_const"); tab = S.get("table", {})
    for ctl, lab in (("precision", "learned_state vs precision (reference)"), ("two_trace", "learned_state vs two_trace (reference)")):
        cc = c.get(ctl)
        if cc is not None: cl.append((f"{lab}: {verdict(cc)[1]} -> {control_label(cc)}", True))
    if best and best in c: cl.append((f"learned_state vs the best fixed rate (`{best}`, reference): {verdict(c[best])[1]} -> {control_label(c[best])}", True))
    pv = S.get("precision_vs_fixed", {}).get(best) if best else None
    if pv is not None: ok, txt = verdict(pv); cl.append((f"P3 precision vs the best fixed rate (`{best}`): {txt} -> {'SUPPORTED' if ok else 'not supported'}", True))
    tt = S.get("two_trace_vs_precision", {}).get("precision")
    if tt is not None: ok, txt = verdict(tt); cl.append((f"P4 two_trace vs precision: {txt} -> {'SUPPORTED' if ok else 'not supported'}", True))
    lf = tab.get("learned_free", {}).get("acc_mean", {}).get("mean"); ls = tab.get("learned_state", {}).get("acc_mean", {}).get("mean")
    if lf is not None and ls is not None and best in tab:
        bc = tab[best]["acc_mean"]["mean"]; cl.append((f"seed means: learned_free {lf:.4f}, learned_state {ls:.4f}, best fixed rate {bc:.4f} (a learned gate below the best fixed rate is reported as an optimisation failure of the arm, not as evidence about the state)", True))
    return cl

aif_block("P3-E1 — sparse random addressing vs dense keys at equal active rows (recall accuracy at P = 128)", "p3_e1_addressing", "sparse_random", ["dense"], True, "accuracy at P = 128", extra=e1_extra)
aif_block("P3-E2 (v2) — K0: learned write gate with vs without per-address states (overall accuracy, mean of 4 regimes)", "p3_e2_tracking", "learned_state", ["learned_free"], True, "overall accuracy", extra=e2_extra)

def e3_extra(S):
    c = S.get("comparisons", {}); sub = S.get("substrate", {}); cl = []
    for ctl, lab in (("learned_free", "tt_learned_state vs learned_free (E2 reference)"), ("two_trace", "tt_learned_state vs the fixed two_trace (E2 reference)")):
        if ctl in c: cl.append((f"{lab}: {verdict(c[ctl])[1]} -> {control_label(c[ctl])}", True))
    for ctl, lab in (("single_free_wide", "SUBSTRATE: tt_learned_free vs the state-matched single store"), ("two_trace", "tt_learned_free vs the fixed two_trace")):
        if ctl in sub: ok, txt = verdict(sub[ctl]); cl.append((f"{lab}: {txt} -> {'the substrate with a learned readout wins' if ok else 'not beyond threshold'} (reported, not part of the K0-TT verdict)", True))
    orc = E / "p3_e3_two_trace" / "oracle_ceiling.json"
    if orc.exists():
        o = json.loads(orc.read_text(encoding="utf-8")); cl.append((f"post-hoc oracle ceiling (exact Bayesian filtering with the generator's rates, `scripts/e2_oracle_ceiling.py`): class-balanced {o['acc_classes']:.4f}, natural mix {o['acc_mean']:.4f}; the learned arms sit about 9 and 3 points below it, so the stream has headroom and the co-location of every learned mechanism is not an information ceiling", True))
    return cl

aif_block("P3-E3 — K0 on the two-trace substrate: learned readout and gates with vs without the states; state-matched single store (class-balanced accuracy)", "p3_e3_two_trace", "tt_learned_state", ["tt_learned_free", "single_free_wide"], True, "class-balanced accuracy", extra=e3_extra)
_l8 = E / "p3_e3_two_trace" / "summary_long800.json"
if _l8.exists():
    L8 = json.loads(_l8.read_text(encoding="utf-8")); t = L8.get("table", {}); c8 = L8.get("comparisons", {})
    out += ["### Exploratory variant `long800` (800 steps, lr 3e-3, seeds 0-2; NOT pre-registered as such, reported beside the grid, never merged)", "",
            "- " + ", ".join(f"{k}: {v['acc_classes']['mean']:.4f}" for k, v in t.items() if 'acc_classes' in v) + ".",
            "- " + "; ".join(f"tt_learned_state vs {k}: {verdict(v)[1]}" for k, v in c8.items()) + ". Motivates the pre-registered E4 (fresh seeds).", ""]
aif_block("P3-E4 — E3's nested test at 800 training steps, fresh seeds (class-balanced accuracy)", "p3_e4_two_trace_long", "tt_learned_state", ["tt_learned_free", "single_free_wide"], True, "class-balanced accuracy", extra=e3_extra)


def s1_block():
    """Rung 1 (GPU, two-store form): the pre-registered verdict is computed by the experiment script itself over all settings."""
    global out
    out += ["", "# Phase 3 — system level (GPU)", "", "## P3-S1 (v3) — the two-store memory layer with vs without per-address states in trained memory-only sequence models (rung 1)", "",
            "PREREG: `experiments/p3_s1_hybrid/PREREG.md`; runbook: docs/phase3/GPU_RUNBOOK.md. Primary metric: held-out accuracy on the decisive strata (re-bound, after-outlier; equal-weighted). The verdict below is computed by `run.py` from the pre-registered conjuncts over every outlier rate.", ""]
    S = load("p3_s1_hybrid")
    if not S: out.append("_not run_\n"); return
    out.append(md("p3_s1_hybrid"))                                                     # SUMMARY.md already ends with the computed verdict
    rep = E / "p3_s1_hybrid" / "summary_replication.json"
    if rep.exists():
        Rv = json.loads(rep.read_text(encoding="utf-8")).get("verdict", {}); out += ["### Replication grid (fresh seeds; side by side, never merged)", ""] + [f"- replication {k}: {val}" for k, val in Rv.items()] + [""]

s1_block()


def s1b_block():
    """Rung 1b (GPU): the Kenyon-cell address against Gated DeltaNet. The verdict (A) and the fixed-rule comparisons (B-E) are computed by the experiment script."""
    global out
    out += ["", "## P3-S1b (v2) — the Kenyon-cell sparse address against Gated DeltaNet in trained memory-only models (rung 1b)", "",
            "PREREG: `experiments/p3_s1b_sparse/PREREG.md`. Primary metric: held-out recall accuracy on the capacity form of the dense recall task "
            "(P = 32 / 64 / 128 keys, key width 16, one wiring per seed and layer). A is the verdict (sparse against Gated DeltaNet at equal active rows "
            "and parameters); B (equal compute), C (equal memory), D (product keys) and E (this repository's full layer) are classified by fixed rules "
            "(BEATS / LOSES when the paired-bootstrap CI excludes 0, MATCHES when it lies within +/- 0.02). Per-run values: `runs_table.csv`.", ""]
    S = load("p3_s1b_sparse")
    if not S: out.append("_not run_\n"); return
    out.append(md("p3_s1b_sparse"))                                                    # SUMMARY.md already ends with the computed verdict and comparisons
    rep = E / "p3_s1b_sparse" / "summary_replication.json"
    if rep.exists():
        Rv = json.loads(rep.read_text(encoding="utf-8")).get("verdict", {}); out += ["### Replication grid (fresh seeds; side by side, never merged)", ""] + [f"- replication {k}: {val}" for k, val in Rv.items()] + [""]

s1b_block()
(R / "RESULTS.md").write_text("\n".join(out), encoding="utf-8", newline="\n")   # explicit: Windows would otherwise write its legacy code page
print("\n".join(out)[:3000])
