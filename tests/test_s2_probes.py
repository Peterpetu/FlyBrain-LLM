"""Rung-2 recall probes (experiments/p3_s2_lm/probes.py) on a constructed text whose events are known: consistent repeats
and a persistent re-binding of one key, near and far, scored at the token that predicts the value; a key in two D1 strata
yields one probe per event. Needs the GPT-2 tokenizer (downloaded once into the Hugging Face cache)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "p3_s2_lm"))
transformers = pytest.importorskip("transformers")


_TOK = []


def tokenizer():
    """Loaded once, from the local cache only (each network check of the hub can take minutes)."""
    if not _TOK:
        try: _TOK.append(transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True))
        except Exception as e: pytest.skip(f"GPT-2 tokenizer not in the local cache (run scripts/fetch_wikitext103.py once): {e}")
    return _TOK[0]


def test_probes_mark_the_known_events():
    import probes
    fill = lambda n: " ".join(str(1000 + i) for i in range(n))              # numbers are never D1 keys (not alphabetic)
    # Zorblat (capitalised, 5 occurrences: in both the rare and the cap strata) with values red7 red7 blue9 blue9 blue9 (values are
    # not alphabetic, so they are never keys themselves), then a far consistent repeat of a second key, Quendor
    parts = ["Zorblat red7", fill(20), "Zorblat red7", fill(400), "Zorblat blue9", fill(20), "Zorblat blue9", fill(20), "Zorblat blue9",
             fill(5), "Quendor gold3", fill(5), "Quendor gold3", fill(300), "Quendor gold3", fill(5)]
    text = " " + " ".join(parts) + "\n"
    enc = tokenizer()(text, return_offsets_mapping=True); ids = np.asarray(enc["input_ids"]); off = np.asarray(enc["offset_mapping"])
    P = probes.build(text, ids, off, far=200)
    kinds = [probes.TYPES[t] for t in P["type"]]
    # Zorblat pairs: (1,2) red7-red7 consistent, near; (2,3) red7->blue9 re-binding (400 numbers apart); (3,4) blue9-blue9 confirms it:
    # one re-binding probe, near, not also a recall probe; (4,5) blue9-blue9 consistent, near. Quendor: a near, then a far repeat.
    # One probe per event although both keys are in two strata.
    assert kinds == ["recall_near", "rebind_near", "recall_near", "recall_near", "recall_far"], kinds
    first = lambda w: tokenizer()(w)["input_ids"][0]
    assert list(P["target"]) == [first(" red7"), first(" blue9"), first(" blue9"), first(" gold3"), first(" gold3")]
    assert all(ids[p + 1] == t for p, t in zip(P["pos"], P["target"])) and P["gap"][-1] > 200 and (P["gap"][:-1] <= 200).all()
    import re
    starts = [m.start() for m in re.finditer(r"Zorblat|Quendor", text)]                                   # characters where a key occurrence begins
    zq = {t for c in starts for t in range(len(off)) if off[t][0] <= c < off[t][1]}                      # the tokens covering them
    assert (P["back"] >= P["gap"]).all() and all(p - P["back"][k] in zq for k, p in enumerate(P["pos"]))   # back reaches the previous key's first token
