#!/usr/bin/env python3
"""Genomic-bottleneck measurements on MaleCNS v1.0.

Question (phase-3 brief: GENOME, DEVELOPMENT AND INNATENESS): how many bits does a
*type-level* ("genome-like") description of the fly connectome cost, how much of the
neuron-level graph does it predict, and how much of what is left is developmental noise
rather than anything a genome could specify?

Outputs five blocks:
  1. type statistics and columnar repetition (weight sharing)
  2. description length: raw graph vs degree model vs type-level SBM
  3. naive low-rank ("g-network") factorisation of the type x type matrix
  4. hemilineage -> neurotransmitter (block-structured Dale's law)
  5. bilateral reproducibility within one animal (the developmental noise floor)

Every "weight" is an anatomical synaptic-contact count; signs are neurotransmitter
predictions. See LIMITATIONS.md.
"""
from __future__ import annotations
import math, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from flybrain.connectome.io import load_connectome, DATA_DIR

ANN = DATA_DIR / "malecns" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"


def log2binom(n: float, k: float) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def entropy(labels) -> float:
    _, c = np.unique(labels, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main() -> None:
    C = load_connectome()
    N, E = C.n, C.n_edges
    ann = pd.read_feather(ANN).set_index("bodyId").reindex(pd.Index(C.body_id.astype(np.int64)))
    ty = ann["type"].fillna("__NA__").astype(str).values
    sc = ann["superclass"].fillna("NA").astype(str).values
    side = ann["somaSide"].fillna("?").astype(str).values
    nt = pd.Series(C.nt).fillna("unknown").astype(str).values
    u, inv = np.unique(ty, return_inverse=True)
    cnt_all = np.bincount(inv, minlength=len(u))

    print(f"# MaleCNS v1.0: N={N} neurons, E={E} edges (>=3 contacts), "
          f"density={E/(N*(N-1)):.3e}")

    # ---- 1. types and columnar repetition -----------------------------------
    print("\n## 1. cell types and repetition")
    print(f"distinct type labels: {int(ann['type'].notna().sum())} annotated neurons, "
          f"{len(np.unique(ty[ann['type'].notna().values]))} types; "
          f"neurons/type mean {cnt_all.mean():.1f} median {np.median(cnt_all):.0f} max {cnt_all.max()}")
    for th in (8, 50, 200, 500, 800):
        s = cnt_all[cnt_all >= th]
        print(f"  neurons in types with >= {th:3d} members: {int(s.sum()):>6} "
              f"({100*s.sum()/N:4.1f}%) in {int((cnt_all>=th).sum())} types")
    for g in ("ol_intrinsic", "visual_projection", "cb_intrinsic", "vnc_intrinsic"):
        m = sc == g
        sz = np.bincount(inv[m]); sz = sz[sz > 0]
        print(f"  {g:18s} n={int(m.sum()):>6} types={len(sz):>5} "
              f"median={np.median(sz):>5.0f} mean={sz.mean():>7.1f} max={sz.max()}")

    # ---- 2. description length ----------------------------------------------
    print("\n## 2. description length of the topology")
    dl_raw = log2binom(N * (N - 1), E)
    print(f"raw (uniform ER code at matched density): {dl_raw:.4g} bits = {dl_raw/8e6:.2f} MB "
          f"= {dl_raw/N:.0f} bits/neuron")

    A = (C.W > 0).tocsr()
    kout = np.asarray(A.sum(1)).ravel().astype(np.float64)
    kin = np.asarray(A.sum(0)).ravel().astype(np.float64)
    rng = np.random.default_rng(0)
    M = 4_000_000
    p = np.clip(kout[rng.integers(0, N, M)] * kin[rng.integers(0, N, M)] / E, 1e-12, 1 - 1e-12)
    dl_deg = float((-(p*np.log2(p) + (1-p)*np.log2(1-p))).mean()) * N * (N - 1)
    print(f"residual | degree sequence (Chung-Lu, Monte-Carlo): {dl_deg/8e6:.2f} MB")

    keep = np.flatnonzero(cnt_all >= 3)
    tmap = -np.ones(len(u), np.int64); tmap[keep] = np.arange(len(keep))
    tc = tmap[inv]; m = tc >= 0; T = len(keep); n_t = cnt_all[keep].astype(np.float64)
    coo = A.tocoo(); sel = m[coo.row] & m[coo.col]
    K = np.zeros((T, T), np.float64)
    np.add.at(K, (tc[coo.row[sel]], tc[coo.col[sel]]), 1.0)
    Np = np.outer(n_t, n_t); np.fill_diagonal(Np, n_t * (n_t - 1))

    def bits(Q):
        Q = np.clip(Q, 1e-12, 1 - 1e-12)
        return float((-(K*np.log2(Q)) - (Np - K)*np.log2(1 - Q)).sum())

    Pf = np.divide(K, Np, out=np.zeros_like(K), where=Np > 0)
    nz = int((K > 0).sum()); res = bits(Pf)
    par = log2binom(T*T, nz) + 8*nz
    print(f"types with >=3 neurons: {T} covering {int(m.sum())} neurons ({100*m.mean():.1f}%), "
          f"{int(K.sum())} edges, {nz} non-empty type-pair blocks ({100*nz/T**2:.2f}%)")
    print(f"  global density only           : {bits(np.full_like(Pf, K.sum()/Np.sum()))/8e6:6.2f} MB")
    print(f"  residual | free type-pair table: {res/8e6:6.2f} MB  ({res/K.sum():.2f} bits/edge)")
    print(f"  that table's own cost          : {par/8e6:6.2f} MB")
    print(f"  TOTAL type-level model+residual: {(res+par)/8e6:6.2f} MB  "
          f"-> {dl_raw/(res+par):.2f}x vs raw")
    print(f"  type label per neuron          : {N*math.log2(len(u))/8e6:.2f} MB "
          f"({math.log2(len(u)):.1f} bits/neuron)")

    # ---- 3. naive low-rank g-network ----------------------------------------
    print("\n## 3. is the type x type matrix low rank?  (naive SVD fits)")
    X = np.log(np.maximum(Pf, 1e-6)).astype(np.float32)
    mu = float(X.mean()); Xc = X - mu
    Om = rng.standard_normal((T, 208)).astype(np.float32)
    Y = Xc @ Om
    for _ in range(2):
        Y = Xc @ (Xc.T @ Y)
    Qq, _ = np.linalg.qr(Y)
    Ub, S, Vt = np.linalg.svd(Qq.T @ Xc, full_matrices=False)
    U = Qq @ Ub; tot = float((S**2).sum())
    print("  cumulative Frobenius energy of centred log-density:",
          {r: round(float((S[:r]**2).sum()/tot), 3) for r in (1, 4, 16, 32, 64, 128, 192)})
    for r in (8, 32, 64, 128, 192):
        low = (U[:, :r] * S[:r]) @ Vt[:r]
        best = min((bits(np.exp(np.clip(mu + s*low, -30, -1e-6))), s) for s in (0.6, 0.8, 1.0))
        print(f"  rank {r:3d} (scale {best[1]}): residual {best[0]/8e6:6.2f} MB "
              f"+ {2*T*r*16/8e6:.2f} MB fp16 params = {(best[0]+2*T*r*16)/8e6:6.2f} MB")

    # ---- 4. hemilineage -> neurotransmitter ---------------------------------
    print("\n## 4. hemilineage determines neurotransmitter (block-structured Dale)")
    for col in ("itoleeHl", "trumanHl"):
        hl = ann[col].fillna("NA").astype(str).values
        mm = (hl != "NA") & (hl != "putative_primary")
        lab, ntm, tym = hl[mm], nt[mm], ty[mm]
        ul = np.unique(lab)
        cond = sum(len(ntm[lab == l])/len(ntm) * entropy(ntm[lab == l]) for l in ul)
        H0 = entropy(ntm)
        pure = sum(1 for l in ul
                   if np.unique(ntm[lab == l], return_counts=True)[1].max() /
                      (lab == l).sum() >= 0.9)
        tph = [len(np.unique(tym[lab == l])) for l in ul]
        span = pd.DataFrame({"t": tym, "l": lab}).groupby("t")["l"].nunique()
        print(f"  {col}: {int(mm.sum())} neurons, {len(ul)} hemilineages")
        print(f"    H(NT)={H0:.3f} b  H(NT|hemilineage)={cond:.3f} b  "
              f"gain={H0-cond:.3f} b ({100*(H0-cond)/H0:.1f}%)")
        print(f"    >=90% one transmitter: {pure}/{len(ul)} ({100*pure/len(ul):.0f}%); "
              f"types/hemilineage mean {np.mean(tph):.1f} median {np.median(tph):.0f}; "
              f"types spanning >1 hemilineage {int((span>1).sum())}/{len(span)} "
              f"({100*(span>1).mean():.1f}%)")

    # ---- 5. bilateral reproducibility ---------------------------------------
    print("\n## 5. bilateral reproducibility within one animal (developmental noise floor)")
    Wc = C.W.tocoo()
    KL = np.zeros((T, T), np.float32); KR = np.zeros((T, T), np.float32)
    for S_, dst in (("L", KL), ("R", KR)):
        s_ = side == S_
        q = m[Wc.row] & m[Wc.col] & s_[Wc.row] & s_[Wc.col]
        np.add.at(dst, (tc[Wc.row[q]], tc[Wc.col[q]]), Wc.data[q].astype(np.float32))
    nL = np.bincount(tc[m & (side == "L")], minlength=T).astype(np.float32)
    nR = np.bincount(tc[m & (side == "R")], minlength=T).astype(np.float32)
    both = np.flatnonzero((nL >= 2) & (nR >= 2))
    pL = KL[np.ix_(both, both)] / np.maximum(nL[both][:, None], 1)
    pR = KR[np.ix_(both, both)] / np.maximum(nR[both][:, None], 1)
    print(f"  {len(both)} types with >=2 neurons on each side")
    nrmL = np.linalg.norm(pL, axis=1); nrmR = np.linalg.norm(pR, axis=1)
    ok = (nrmL > 0) & (nrmR > 0)
    cs = (pL[ok]*pR[ok]).sum(1) / (nrmL[ok]*nrmR[ok])
    perm = np.random.default_rng(1).permutation(ok.sum())
    cs0 = (pL[ok]*pR[ok][perm]).sum(1) / (nrmL[ok]*nrmR[ok][perm])
    print(f"  cosine(L,R) of type-level out-profile: median {np.median(cs):.3f} "
          f"IQR [{np.quantile(cs,.25):.3f},{np.quantile(cs,.75):.3f}] n={ok.sum()}; "
          f"mismatched-type control median {np.median(cs0):.3f}")
    for thr in (0, 2, 5, 10, 20, 50):
        BL, BR = pL > thr, pR > thr
        print(f"  Jaccard(L,R) of type->type edges at > {thr:2d} contacts/source neuron: "
              f"{(BL & BR).sum()/max(1,(BL | BR).sum()):.3f}  (|L|={int(BL.sum())})")
    msk = (pL > 0) | (pR > 0)
    print(f"  Pearson r of log1p(contacts/source neuron) on union-present pairs: "
          f"{np.corrcoef(np.log1p(pL[msk]), np.log1p(pR[msk]))[0,1]:.3f} (n={int(msk.sum())})")


if __name__ == "__main__":
    main()
