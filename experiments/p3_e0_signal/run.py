"""P3-E0: signal availability of candidate per-address states (numpy, no training). See PREREG.md.
Hardened from the red team's scratch scripts (docs/phase3/DESIGN_PANEL.md)."""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flybrain.harness.prereg import Prereg

EXP = Path(__file__).resolve().parent
N_EACH, N_DIS, D_V = 32, 96, 16
BETAS = (0.25, 0.5, 0.8, 1.0)
KEYCODES = ("sparse", "dense", "dense_aniso0.5", "dense_aniso1.0")
MODES = ("rebind", "cancel")
CONTROL = ("e1", "vhat1", "vhat2", "cos", "kMk")     # what a learned gate already has from the delta rule's own intermediates
STATES = ("a", "zc", "nconf", "logage")             # candidate per-address states
LOGF = {"e1", "vhat1", "vhat2", "kMk", "a", "zraw", "zc"}
RHO_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 59)])   # oracle floor sweep for the contradiction mass (60 values)


def auroc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x, kind="mergesort"); xs, ys = x[o], y[o]; r = np.empty(len(xs)); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]: j += 1
        r[i:j + 1] = 0.5 * (i + j) + 1.0; i = j + 1
    npos, nneg = ys.sum(), (1 - ys).sum()
    return float((r[ys == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def make_keys(m, keycode, rng):
    if keycode == "sparse":
        n, kap = 2048, 32; K = np.zeros((m, n))
        for i in range(m): K[i, rng.choice(n, kap, replace=False)] = 1 / np.sqrt(kap)
        return K, kap / n
    d = 128; K = rng.standard_normal((m, d))
    if keycode.startswith("dense_aniso"): K = K * np.arange(1, d + 1) ** (-float(keycode.split("aniso")[1]))
    return K / np.linalg.norm(K, axis=1, keepdims=True), 1 / np.sqrt(d)


def unit(x): return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def stream(seed, keycode, beta, mode):
    """One stream; returns per-address features at the probe event and class labels (0 novel, 1 clean, 2 contradicted)."""
    rng = np.random.default_rng(seed); nk = 3 * N_EACH; sparse = keycode == "sparse"
    K, rho = make_keys(nk, keycode, rng); Kd, _ = make_keys(N_DIS, keycode, rng); dim = K.shape[1]
    nov, cln, con = np.arange(N_EACH), np.arange(N_EACH, 2 * N_EACH), np.arange(2 * N_EACH, 3 * N_EACH)
    V = unit(rng.standard_normal((nk, D_V))); Vc = -V if mode == "cancel" else unit(rng.standard_normal((nk, D_V)))
    Vp = unit(rng.standard_normal((nk, D_V)))                          # fresh probe values: the same current event for every class
    # the store and the candidate states
    S = np.zeros((dim, D_V)); Z = np.zeros(dim); sig = 0.0; Zc = np.zeros(dim); Cv = np.zeros((dim, D_V))
    Kw = np.zeros((4 * nk + N_DIS, dim)); bw = np.zeros(4 * nk + N_DIS); nw = 0    # written keys (for k^T M k)
    ts = np.full(dim, -1e9); last_addr = {}                            # row timestamps (sparse) / oracle per-address last touch (dense)
    def probe_feats(i, k, v, t):
        vhat = S.T @ k; e = v - vhat; zraw = float(Z @ k)
        cread = Cv.T @ k; nconf = float(cread @ unit(vhat))
        age = t - (np.median(ts[k > 0]) if sparse else last_addr.get(i, -1e9))
        return {"e1": float(np.abs(e).sum()), "vhat1": float(np.abs(vhat).sum()), "vhat2": float(np.linalg.norm(vhat)),
                "cos": float(unit(v) @ unit(vhat)), "kMk": float((bw[:nw] * (Kw[:nw] @ k) ** 2).sum()),
                "a": max(0.0, zraw - rho * sig), "zraw": zraw, "sig": sig, "zc": float(Zc @ k), "nconf": nconf,
                "logage": float(np.log1p(min(max(age, 0.0), 1e6)))}
    def write(i, k, v, b, t):
        nonlocal S, Z, sig, Zc, Cv, nw
        vhat = S.T @ k; e = v - vhat; ev = float(np.abs(e).mean())
        familiar = np.linalg.norm(vhat) >= 0.5 * b                       # the store knows its own write rate; one write leaves ||vhat|| ~ b
        contradict = familiar and float(unit(v) @ unit(vhat)) <= 0.5      # content-based: the observation disagrees with what is stored
        S = S + b * np.outer(k, e)                                        # gated delta rule, decay 1: S <- S + beta k (v - S^T k)^T
        Z = Z + b * k * ev; sig = sig + b * ev                            # ATTEST contradiction mass (d_z = 1, |written residual|) and its floor mass
        Zc = Zc + b * k * ev * contradict                                 # familiarity-gated contradiction mass (only disagreements at known addresses)
        cread = Cv.T @ k; target = np.zeros(D_V) if contradict else cread + v      # confirmation vector: +v on agreement / novelty, reset on contradiction
        Cv = Cv + np.outer(k, target - cread)
        Kw[nw] = k; bw[nw] = b; nw += 1
        if sparse: ts[k > 0] = t
        else: last_addr[i] = t
    events = []
    for i in cln: events += [("w", i, V[i]), ("w", i, V[i]), ("p", i, Vp[i])]
    for i in con: events += [("w", i, V[i]), ("w", i, Vc[i]), ("p", i, Vp[i])]
    for i in nov: events += [("p", i, Vp[i])]
    for j in range(N_DIS): events += [("d", j, unit(rng.standard_normal(D_V)))]
    # shuffle while keeping each address's own events in order
    order = rng.permutation(len(events)); per_addr = {}
    for pos in order:
        kind, i, v = events[pos]; per_addr.setdefault((kind == "d", i), []).append(pos)
    slots = sorted(order); seq = [None] * len(events); ptr = {key: 0 for key in per_addr}
    addr_of_pos = {}
    for key, poss in per_addr.items():
        for p in poss: addr_of_pos[p] = key
    own = {key: sorted(poss) for key, poss in per_addr.items()}         # each address's events in their original (correct) order
    for slot, pos in zip(slots, order):
        key = addr_of_pos[pos]; seq[slot] = events[own[key][ptr[key]]]; ptr[key] += 1
    feats, labels = {}, {}
    for t, (kind, i, v) in enumerate(seq):
        if kind == "d": write(("d", i), Kd[i], v, beta, t); continue
        k = K[i]
        if kind == "p": feats[i] = probe_feats(i, k, v, t); labels[i] = 0 if i in nov else (1 if i in cln else 2)
        write(i, k, v, beta, t)
    idx = sorted(feats); X = {f: np.array([feats[i][f] for i in idx]) for f in feats[idx[0]]}; y = np.array([labels[i] for i in idx])
    return X, y


def softmax_fit(Xtr, ytr, Xte, yte, steps=3000, lr=0.1):
    """Multinomial logistic regression (numpy), standardised features; returns test accuracy."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    A = np.concatenate([A, np.ones((len(A), 1))], 1); B = np.concatenate([B, np.ones((len(B), 1))], 1)
    W = np.zeros((A.shape[1], 3)); Y = np.eye(3)[ytr]
    for _ in range(steps):
        L = A @ W; P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        W -= lr * (A.T @ (P - Y) / len(A) + 1e-3 * W)
    return float(((B @ W).argmax(1) == yte).mean())


def hb_test():
    """||S^T k|| after +v then -v at the same address: gated linear attention vs the true gated delta rule."""
    rng = np.random.default_rng(0); d_k, d_v = 64, 8; k = rng.standard_normal(d_k); k /= np.linalg.norm(k); v = rng.standard_normal(d_v); out = {}
    for beta in BETAS:
        S = np.zeros((d_k, d_v))
        for vt in (v, -v): S = S + beta * np.outer(k, vt)
        gla = float(np.linalg.norm(S.T @ k)); S = np.zeros((d_k, d_v))
        for vt in (v, -v): e = vt - S.T @ k; S = S + beta * np.outer(k, e)
        out[f"beta{beta:g}"] = {"gla": gla, "gdn": float(np.linalg.norm(S.T @ k)), "v_norm": float(np.linalg.norm(v))}
    return out


def feat(x, f): return np.log1p(np.abs(x)) if f in LOGF else x


def stack(rs, feats):
    return (np.concatenate([np.stack([feat(r[0][f], f) for f in feats], 1) for r in rs]), np.concatenate([r[1] for r in rs]))


def main(seeds):
    t0 = time.time(); results = {"hb": hb_test()}; lines = []
    for keycode in KEYCODES:
        for mode in MODES:
            for beta in BETAS:
                runs = [stream(s, keycode, beta, mode) for s in seeds]
                tr = [r for i, r in enumerate(runs) if i % 2 == 0]; te = [r for i, r in enumerate(runs) if i % 2 == 1]
                Xtr, ytr = stack(tr, CONTROL); Xte, yte = stack(te, CONTROL); acc_ctl = softmax_fit(Xtr, ytr, Xte, yte)
                row = {"keycode": keycode, "mode": mode, "beta": beta, "acc_control": acc_ctl}
                for st in STATES:
                    Xtr2, _ = stack(tr, CONTROL + (st,)); Xte2, _ = stack(te, CONTROL + (st,)); row[f"acc_control+{st}"] = softmax_fit(Xtr2, ytr, Xte2, yte)
                Xtr3, _ = stack(tr, CONTROL + STATES); Xte3, _ = stack(te, CONTROL + STATES); row["acc_control+all_states"] = softmax_fit(Xtr3, ytr, Xte3, yte)
                Xtr4, _ = stack(tr, STATES); Xte4, _ = stack(te, STATES); row["acc_states_only"] = softmax_fit(Xtr4, ytr, Xte4, yte)
                allX = {f: np.concatenate([r[0][f] for r in runs]) for f in runs[0][0]}; ally = np.concatenate([r[1] for r in runs])
                for f in CONTROL + STATES + ("zraw",):
                    x = allX[f]; row[f"auroc_NvC_{f}"] = auroc(x[ally == 2], x[ally == 0]); row[f"auroc_CvC_{f}"] = auroc(x[ally == 2], x[ally == 1])
                row["dead_fraction_a"] = float((allX["a"] == 0).mean())
                # oracle floor for the contradiction mass (most charitable, label-aware): best rho over the grid, per-address floor rho * sigma
                best, best_rho = -1.0, 0.0
                for rho_f in RHO_GRID:
                    aa = np.maximum(0.0, allX["zraw"] - rho_f * allX["sig"]); au = auroc(aa[ally == 2], aa[ally == 1])
                    if au > best: best, best_rho = au, float(rho_f)
                row["auroc_CvC_a_oracle_floor"] = best; row["oracle_rho"] = best_rho
                results[f"{keycode}|{mode}|{beta:g}"] = row
                lines.append(f"| {keycode} | {mode} | {beta:g} | {acc_ctl:.3f} | " + " | ".join(f"{row[f'acc_control+{s}'] - acc_ctl:+.3f}" for s in STATES)
                             + f" | {row['acc_control+all_states'] - acc_ctl:+.3f} | {row['auroc_CvC_cos']:.2f} / {row['auroc_CvC_vhat2']:.2f} | {row['auroc_CvC_a']:.2f} | {row['auroc_CvC_a_oracle_floor']:.2f} | {row['auroc_CvC_zc']:.2f} | {row['auroc_CvC_nconf']:.2f} | {row['dead_fraction_a']:.2f} |")
    # verdicts (pre-registered)
    rows = {k: r for k, r in results.items() if k != "hb"}
    inc = {k: {st: r[f"acc_control+{st}"] - r["acc_control"] for st in STATES} for k, r in rows.items()}
    K1 = [k for k, r in rows.items() if max(inc[k].values()) <= 0.02]
    K2 = [k for k, r in rows.items() if r["keycode"] != "sparse" and r["auroc_CvC_a"] < 0.7]
    K3 = [k for k, r in rows.items() if r["keycode"] != "sparse" and r["dead_fraction_a"] > 0.5]
    K4 = [b for b, h in results["hb"].items() if h["gdn"] > 0.1 * h["v_norm"]]
    sparse_inc = {st: float(np.mean([inc[k][st] for k in rows if k.startswith("sparse")])) for st in STATES}
    dense_inc = {st: float(np.mean([inc[k][st] for k in rows if not k.startswith("sparse")])) for st in STATES}
    admitted = [st for st, v in sparse_inc.items() if v > 0.05]
    verdict = {"K1_fires_in": K1, "K2_fires_in": K2, "K3_fires_in": K3, "K4_fires_for_betas": K4, "sparse_incremental_accuracy": sparse_inc,
               "dense_incremental_accuracy": dense_inc, "states_admitted_to_E2": admitted}
    results["verdict"] = verdict
    md = [f"## P3-E0: signal availability of candidate per-address states (no training; {len(seeds)} seeds, even train / odd test)", "",
          "Columns: control logistic 3-way accuracy (novel / clean / contradicted history; the novel class is trivially separable, so the "
          "informative contrast is clean-vs-contradicted, CvC); incremental accuracy from adding each state; CvC AUROC of the control's own "
          "cos(v, vhat) / ||vhat||; CvC AUROC of the contradiction mass a at the stated floor, at the oracle floor, of the familiarity-gated "
          "mass zc, of the confirmation count; dead fraction of a.", "",
          "| key code | mode | beta | control acc | +a | +zc | +nconf | +log age | +all | AUROC CvC cos / vhat | a (stated floor) | a (oracle floor) | zc | nconf | dead a |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"] + lines
    md += ["", "H_B test (||S^T k|| after +v, -v; ||v|| = %.3f): " % results["hb"]["beta1"]["v_norm"] + ", ".join(f"beta {b[4:]}: GLA {h['gla']:.4f} / GDN {h['gdn']:.3f}" for b, h in results["hb"].items()), "",
           f"**Pre-registered kill conditions.** K1 (best single state adds <= 0.02 accuracy) fires in {len(K1)} of {len(rows)} configurations"
           f" ({sum(k.startswith('sparse') for k in K1)} of {sum(k.startswith('sparse') for k in rows)} sparse); K2 (dense contradiction-mass AUROC < 0.7 at the stated floor) fires in {len(K2)} of {sum(not k.startswith('sparse') for k in rows)} dense; "
           f"K3 (dense dead fraction > 0.5) fires in {len(K3)}; K4 (H_B residual > 0.1 ||v|| under the gated delta rule) fires for betas {[b[4:] for b in K4]}.",
           f"**Incremental 3-way accuracy on sparse keys (mean over beta and mode):** " + ", ".join(f"{s} {v:+.3f}" for s, v in sparse_inc.items())
           + "; on dense keys: " + ", ".join(f"{s} {v:+.3f}" for s, v in dense_inc.items()) + f". **States admitted to E2 (> 0.05 on sparse):** {admitted or 'none'}.", ""]
    (EXP / "SUMMARY.md").write_text("\n".join(md)); (EXP / "summary.json").write_text(json.dumps(results, indent=1, default=float))
    print("\n".join(md)); print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=list(range(8))); args = ap.parse_args()
    Prereg(EXP).freeze_or_check(); main(args.seeds)
