"""Rung 2 recall probes: the D1 regime's events in tokenized WikiText-103, scored where a language model predicts the value.

The word-level definitions are D1's own (imported from experiments/p3_d1_corpus/run.py, not re-implemented): keys are the
`rare` (alphabetic, 3-100 occurrences) and `cap` strata of the split, the value of an occurrence is the next word, and two
consecutive occurrences of a key form a repeat (consistent if the values agree, a re-binding if not; a re-binding is
persistent if the next occurrence keeps the new value). A probe is the GPT-2 token that starts the value word, predicted from
the token before it. Types:
  recall_far / recall_near   a consistent repeat: the value was shown last time, more / less than FAR tokens earlier
  rebind_far / rebind_near   the confirming occurrence of a persistent re-binding: the NEW value was shown last time,
                             more / less than FAR tokens earlier, while the old value was shown before that
"far" means beyond a sliding-window attention of FAR tokens, so only a memory can supply the value.
Usage: python experiments/p3_s2_lm/probes.py [--split validation] -> data/text/wikitext103/probes_<split>.npz and a summary."""
from __future__ import annotations

import argparse, importlib.util, json, re, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _d1():
    """D1's run.py, loaded by path under its own module name (several experiments have a `run.py`, so a bare
    `import run` would return whichever was imported first in the process)."""
    spec = importlib.util.spec_from_file_location("p3_d1_corpus_run", ROOT / "experiments" / "p3_d1_corpus" / "run.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    return mod


_D1 = _d1(); strata, occurrences = _D1.strata, _D1.occurrences      # D1's definitions, unchanged

DATA = ROOT / "data" / "text" / "wikitext103"
FAR = 512
TYPES = ("recall_near", "recall_far", "rebind_near", "rebind_far")


def build(text: str, ids: np.ndarray, offsets: np.ndarray, far: int = FAR, max_gap: int = 8192) -> dict:
    """Probe arrays for one split: `pos` (index of the predicting token), `target` (the value's first token id), `type`
    (index into TYPES), `gap` (tokens since the key's previous occurrence), `key_type` (0 rare, 1 cap), `w_gap` (words)."""
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]; words = [text[s:e] for s, e in spans]
    tok_at = np.full(len(text) + 1, -1, dtype=np.int64)                   # character -> index of the token covering it
    for t, (s, e) in enumerate(offsets):
        tok_at[s:e] = t
    out = {k: [] for k in ("pos", "target", "type", "gap", "back", "key_type", "w_gap")}      # back: tokens from the predicting position to the previous occurrence's first token
    for key_type, (name, keyset) in enumerate(strata(words).items()):
        key, wpos, nxt = occurrences(words, keyset)
        tpos = np.array([tok_at[spans[p][0]] for p in wpos])                # token index of each key occurrence's first character
        vtok = np.array([tok_at[spans[p + 1][0]] for p in wpos])            # token index of the value word's first character
        same = key[1:] == key[:-1]; gap = tpos[1:] - tpos[:-1]; wgap = wpos[1:] - wpos[:-1]
        rep = same & (gap <= max_gap)                                          # D1's repeat, with the window in tokens (the model's context)
        cons = rep & (nxt[1:] == nxt[:-1]); reb = rep & (nxt[1:] != nxt[:-1])
        pers = np.zeros(len(key) - 1, bool); pers[1:] = reb[:-1] & rep[1:] & (nxt[2:] == nxt[1:-1])    # D1's persistent re-binding, indexed by its confirming pair
        for kind, mask in (("recall", cons & ~pers), ("rebind", pers)):         # a confirming pair is also consistent: it is counted once, as a re-binding
            for j in np.nonzero(mask)[0]:
                v = vtok[j + 1]
                if v <= 0 or (offsets[v][0] not in (spans[wpos[j + 1] + 1][0], spans[wpos[j + 1] + 1][0] - 1)): continue   # the value word must start a token
                out["pos"].append(v - 1); out["target"].append(int(ids[v])); out["gap"].append(int(gap[j])); out["w_gap"].append(int(wgap[j]))
                out["back"].append(int(v - 1 - tpos[j]))
                out["type"].append(TYPES.index(f"{kind}_{'far' if gap[j] > far else 'near'}")); out["key_type"].append(key_type)
    P = {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}
    _, first = np.unique(np.stack([P["pos"], P["type"]]), axis=1, return_index=True)   # a key in both strata (a name with 3-100 occurrences) is one event, not two
    first = np.sort(first); P = {k: v[first] for k, v in P.items()}
    o = np.argsort(P["pos"], kind="stable")
    return {k: v[o] for k, v in P.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="validation"); ap.add_argument("--max_gap", type=int, default=8192); a = ap.parse_args()
    from transformers import AutoTokenizer
    text = (DATA / f"{a.split}.txt").read_text(encoding="utf-8"); ids = np.fromfile(DATA / f"{a.split}.bin", dtype=np.uint16)
    enc = AutoTokenizer.from_pretrained("gpt2")(text, return_offsets_mapping=True)
    assert np.array_equal(np.asarray(enc["input_ids"], dtype=np.uint16), ids), "tokenization differs from the stored split"
    P = build(text, ids, np.asarray(enc["offset_mapping"]), max_gap=a.max_gap)
    np.savez(DATA / f"probes_{a.split}.npz", **P)
    summary = {t: {"n": int((P["type"] == i).sum()), "gap_median": float(np.median(P["gap"][P["type"] == i])) if (P["type"] == i).any() else None}
               for i, t in enumerate(TYPES)}
    print(json.dumps({"split": a.split, "tokens": int(len(ids)), "far": FAR, "max_gap": a.max_gap, "probes": summary}, indent=1))
