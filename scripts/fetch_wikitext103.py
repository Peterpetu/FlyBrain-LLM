"""Rung 2 data: WikiText-103 (raw) tokenized with the GPT-2 BPE tokenizer, one uint16 token file per split, plus the raw
validation and test text (for the word-level recall probes) and a manifest (token counts, tokenizer, SHA-256 of every file).
Writes data/text/wikitext103/ (never committed). Tokenizer threads are limited so that a grid running on the GPU keeps its CPU.
Usage: python scripts/fetch_wikitext103.py [--threads 2]"""
import argparse, hashlib, json, os, time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "text" / "wikitext103"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads); os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from datasets import load_dataset
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tok = AutoTokenizer.from_pretrained("gpt2"); eot = tok.eos_token_id
    manifest = {"source": "Salesforce/wikitext, wikitext-103-raw-v1", "tokenizer": "gpt2 (BPE, 50,257 tokens)", "splits": {}}
    for split in ("train", "validation", "test"):
        text = "".join(ds[split]["text"])                               # the raw lines already end in newlines; articles start with " = Title = "
        pieces, i = [], 0                                               # ~1 MB pieces, each cut just before a heading line (" = ... = "), where
        while i < len(text):                                            # the tokenization cannot depend on the cut (checked below on the small splits)
            j = min(len(text), i + (1 << 20))
            if j < len(text):
                k = text.rfind("\n = ", i, j)
                if k > i: j = k + 1
            pieces.append(text[i:j]); i = j
        ids = [t for enc in tok(pieces)["input_ids"] for t in enc]
        if split != "train": assert ids == tok(text)["input_ids"], f"{split}: piecewise tokenization differs from whole-text tokenization"
        arr = np.asarray(ids, dtype=np.uint16); assert arr.max() < 65536 and len(arr) == len(ids)
        (OUT / f"{split}.bin").write_bytes(arr.tobytes())
        if split != "train": (OUT / f"{split}.txt").write_text(text, encoding="utf-8")
        manifest["splits"][split] = {"tokens": int(len(arr)), "characters": len(text),
                                     "bin_sha256": hashlib.sha256(arr.tobytes()).hexdigest(), "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        print(f"{split}: {len(text):,} characters -> {len(arr):,} tokens ({time.time() - t0:.0f} s)", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1)); print(json.dumps(manifest, indent=1))
