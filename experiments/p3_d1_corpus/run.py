"""P3-D1 (v2): persistent re-binding of key-like tokens in text, by gap, against a permutation null. See PREREG.md."""
from __future__ import annotations

import argparse, json, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flybrain.harness.prereg import Prereg

EXP = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[2] / "data" / "text"
WINDOWS = (512, 2048, 8192); FAR = 512; N_PERM = 20


def load(name):
    return (DATA / name).read_text(errors="ignore").split()


def strata(tokens):
    freq = Counter(tokens); cap, rare = set(), set()
    for t, c in freq.items():
        if not t.isalpha(): continue
        if 3 <= c <= 100: rare.add(t)
        if c >= 3 and t[0].isupper() and freq.get(t.lower(), 0) < c: cap.add(t)
    return {"cap": cap, "rare": rare}


def occurrences(tokens, keyset):
    """Arrays sorted by (key id, position): key id, position, next-token id."""
    ids = {t: i for i, t in enumerate(sorted(keyset))}; vocab = {}
    key, pos, nxt = [], [], []
    for i, t in enumerate(tokens[:-1]):
        if t in ids: key.append(ids[t]); pos.append(i); nxt.append(vocab.setdefault(tokens[i + 1], len(vocab)))
    key, pos, nxt = map(np.asarray, (key, pos, nxt)); o = np.lexsort((pos, key))
    return key[o], pos[o], nxt[o]


def stats(key, pos, nxt, W):
    same = key[1:] == key[:-1]; gap = pos[1:] - pos[:-1]; rep = same & (gap <= W)             # pair (j-1, j)
    cons = rep & (nxt[1:] == nxt[:-1]); reb = rep & (nxt[1:] != nxt[:-1])
    pers = reb[:-1] & rep[1:] & (nxt[2:] == nxt[1:-1])                                        # pair j re-binds and pair j+1 keeps the new binding
    far = pers & (gap[:-1] > FAR); g = gap[:-1][pers]
    return {"repeats": int(rep.sum()), "consistent": int(cons.sum()), "rebind": int(reb.sum()), "persistent": int(pers.sum()), "persistent_far": int(far.sum()),
            "gap_median": float(np.median(g)) if len(g) else float("nan"), "gap_p90": float(np.percentile(g, 90)) if len(g) else float("nan")}


def permuted(key, nxt, rng):
    r = rng.random(len(key)); o = np.lexsort((r, key)); return nxt[o]                       # within-key shuffle of the next tokens (key is sorted)


def main():
    t0 = time.time(); rng = np.random.default_rng(0); res = {}; lines = []
    for corpus in ("wikitext2.train.txt", "ptb.train.txt"):
        toks = load(corpus); n = len(toks); per1k = 1000.0 / n
        for name, keyset in strata(toks).items():
            if not keyset or (name == "cap" and corpus.startswith("ptb")): continue     # PTB is lower-cased: no cap stratum (PREREG)
            key, pos, nxt = occurrences(toks, keyset)
            nulls = [permuted(key, nxt, rng) for _ in range(N_PERM)]
            for W in WINDOWS:
                s = stats(key, pos, nxt, W); ns = [stats(key, pos, nx, W) for nx in nulls]
                null = {m: {"mean": float(np.mean([x[m] for x in ns])), "sd": float(np.std([x[m] for x in ns], ddof=1))} for m in ("persistent", "persistent_far", "consistent")}
                z = {m: (s[m] - null[m]["mean"]) / max(null[m]["sd"], 1e-9) for m in null}
                ratio = {m: s[m] / max(null[m]["mean"], 1e-9) for m in null}
                row = {"corpus": corpus, "stratum": name, "W": W, "n_tokens": n, "n_key_types": len(keyset), "keys_per_1k": len(key) * per1k,
                       "repeats_per_1k": s["repeats"] * per1k, "consistent_fraction": s["consistent"] / max(s["repeats"], 1),
                       "persistent_per_1k": s["persistent"] * per1k, "persistent_far_per_1k": s["persistent_far"] * per1k,
                       "gap_median": s["gap_median"], "gap_p90": s["gap_p90"], "observed": s, "null": null, "z": z, "ratio": ratio}
                res[f"{corpus}|{name}|W{W}"] = row
                lines.append(f"| {corpus.split('.')[0]} | {name} | {W} | {row['keys_per_1k']:.1f} | {row['repeats_per_1k']:.1f} | {row['consistent_fraction']:.2f} | "
                             f"{row['persistent_per_1k']:.2f} | {row['persistent_far_per_1k']:.3f} | {s['gap_median']:.0f} / {s['gap_p90']:.0f} | "
                             f"{ratio['persistent']:.2f}x (z {z['persistent']:+.1f}) | {ratio['persistent_far']:.2f}x (z {z['persistent_far']:+.1f}) | {ratio['consistent']:.2f}x (z {z['consistent']:+.1f}) |")
    ok = {k: (v["persistent_far_per_1k"] > 1000 / 2048) and (v["z"]["persistent"] > 3) for k, v in res.items() if k.endswith("W8192")}
    verdict = {"rule": "far persistent re-bindings > 1 per 2048 tokens at W=8192 AND persistent count exceeds the permutation null by z > 3",
               "passes_in": [k for k, v in ok.items() if v], "language_rung_in_scope": any(ok.values())}
    res["verdict"] = verdict
    md = ["## P3-D1 (v2): persistent re-binding of key-like tokens in text, against a within-key permutation null", "",
          "| corpus | stratum | window | keys / 1k | repeats / 1k | consistent fraction | persistent re-bindings / 1k | far (> 512) persistent / 1k | gap median / p90 | persistent vs null | far persistent vs null | consistent vs null |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"] + lines
    md += ["", f"**Decision rule.** {verdict['rule']}. Passes in: {verdict['passes_in'] or 'none'}. "
           f"**Language rung in scope under the pre-registered rule: {verdict['language_rung_in_scope']}.**", ""]
    (EXP / "SUMMARY.md").write_text("\n".join(md)); (EXP / "summary.json").write_text(json.dumps(res, indent=1, default=float)); print("\n".join(md)); print(f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args(); Prereg(EXP).freeze_or_check(); main()
