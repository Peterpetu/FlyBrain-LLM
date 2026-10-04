"""TinyStories for the exploratory language-model prototype: the first `--train_tokens` tokens of the training split (streamed,
so the whole 2 GB set is not downloaded) and the validation split, GPT-2 BPE, stories separated by the end-of-text token,
one uint16 token file per split and a manifest with hashes. Writes data/text/tinystories/ (never committed).
Usage: python scripts/fetch_tinystories.py [--train_tokens 100000000] [--threads 2]"""
import argparse, hashlib, json, os, time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "text" / "tinystories"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train_tokens", type=int, default=100_000_000); ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(a.threads); os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from datasets import load_dataset
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time(); tok = AutoTokenizer.from_pretrained("gpt2"); eot = tok.eos_token_id
    manifest = {"source": "roneneldan/TinyStories", "tokenizer": "gpt2 (BPE, 50,257 tokens)", "separator": "end-of-text token after every story", "splits": {}}
    for split, limit in (("validation", a.val_tokens), ("train", a.train_tokens)):
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True); parts, n, stories, batch = [], 0, 0, []
        def flush():
            global n, stories
            for ids in tok(batch)["input_ids"]:
                parts.append(np.asarray(ids + [eot], dtype=np.uint16)); n += len(ids) + 1; stories += 1
            batch.clear()
        for ex in ds:
            text = ex["text"].strip()
            if text: batch.append(text)
            if len(batch) == 2000:
                flush()
                if n >= limit: break
        if batch and n < limit: flush()
        arr = np.concatenate(parts); (OUT / f"{split}.bin").write_bytes(arr.tobytes())
        manifest["splits"][split] = {"tokens": int(len(arr)), "stories": stories, "bin_sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
        print(f"{split}: {stories:,} stories -> {len(arr):,} tokens ({time.time() - t0:.0f} s)", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1)); print(json.dumps(manifest, indent=1))
