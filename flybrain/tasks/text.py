"""Character-level corpora (bits per character)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..connectome.io import DATA_DIR


class CharCorpus:
    def __init__(self, name: str = "tinyshakespeare", data_dir: Path | None = None, val_frac: float = 0.1):
        d = Path(data_dir or DATA_DIR) / "text"
        if name == "tinyshakespeare":
            txt = (d / "tinyshakespeare.txt").read_text(encoding="utf-8")
            n = int(len(txt) * (1 - val_frac)); tr, va = txt[:n], txt[n:]
        elif name == "wikitext2":
            tr = (d / "wikitext2.train.txt").read_text(encoding="utf-8"); va = (d / "wikitext2.valid.txt").read_text(encoding="utf-8")
        elif name == "ptb":
            tr = (d / "ptb.train.txt").read_text(encoding="utf-8"); va = (d / "ptb.valid.txt").read_text(encoding="utf-8")
        else:
            raise ValueError(name)
        chars = sorted(set(tr) | set(va)); self.stoi = {c: i for i, c in enumerate(chars)}; self.vocab = len(chars)
        self.train = torch.tensor([self.stoi[c] for c in tr], dtype=torch.long)
        self.val = torch.tensor([self.stoi[c] for c in va], dtype=torch.long)

    def batch(self, split: str, batch: int, seq_len: int, rng: np.random.Generator):
        data = self.train if split == "train" else self.val
        ix = rng.integers(0, len(data) - seq_len - 1, size=batch)
        x = torch.stack([data[i:i + seq_len] for i in ix]); y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
        return x, y

    @torch.no_grad()
    def eval_bpc(self, model, seq_len: int, batch: int = 32, max_chars: int | None = 200_000) -> float:
        model.eval(); data = self.val if max_chars is None else self.val[:max_chars]
        n = (len(data) - 1) // seq_len; total, count = 0.0, 0
        for s in range(0, n, batch):
            idx = torch.arange(s, min(s + batch, n)) * seq_len
            x = torch.stack([data[i:i + seq_len] for i in idx]); y = torch.stack([data[i + 1:i + seq_len + 1] for i in idx])
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
            total += float(loss); count += y.numel()
        model.train()
        return total / count / np.log(2)
