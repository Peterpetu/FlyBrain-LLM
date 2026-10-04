"""Rung 2 building blocks (no pre-registration yet; nothing here has been trained): language-model data, the model, and
the two evaluations (perplexity and the D1-regime recall probes of probes.py) for `flyprinc.hybrid.HybridLM` on
WikiText-103 (scripts/fetch_wikitext103.py). The arms and budgets are decided by rung 2's PREREG after rungs 1 and 1b report.

Evaluation windows: the split is cut into windows of T tokens with stride T/2 and only each window's second half is scored
(the first window is scored whole), so every scored token sees at least T/2 tokens of history. A probe counts only where its
previous occurrence starts inside the window that scores it (`back`, the distance from the predicting position to that
occurrence's first token, is at most the position within the window); `evaluate` reports how many probes each type had and
how many were covered."""
from __future__ import annotations

import math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flyprinc.hybrid import HybridConfig, HybridLM, LayerConfig  # noqa: E402

DATA = ROOT / "data" / "text" / "wikitext103"
VOCAB = 50257                                                                    # GPT-2 BPE
TYPES = ("recall_near", "recall_far", "rebind_near", "rebind_far")               # as probes.TYPES


def set_corpus(name: str) -> None:
    """Point the data functions at data/text/<name>/ (wikitext103 by default; tinystories for the fluency prototype)."""
    global DATA
    DATA = ROOT / "data" / "text" / name


def tokens(split: str) -> np.ndarray:
    return np.memmap(DATA / f"{split}.bin", dtype=np.uint16, mode="r")


def train_batch(ids: np.ndarray, B: int, T: int, rng: np.random.Generator):
    """B random windows of T + 1 tokens: inputs and next-token targets (int64)."""
    s = rng.integers(0, len(ids) - T - 1, size=B); w = np.stack([np.asarray(ids[i:i + T + 1], dtype=np.int64) for i in s])
    return torch.from_numpy(w[:, :-1]), torch.from_numpy(w[:, 1:])


def windows(n: int, T: int):
    """Evaluation windows over n tokens (inputs [s, s + T), predicting [s + 1, s + T + 1)): (start, first scored offset)."""
    out, s = [], 0
    while s + T + 1 <= n:
        out.append((s, 0 if s == 0 else T // 2)); s += T // 2
    return out


def build_model(d_model: int, n_layers: int, n_heads: int, window: int, attn_every: int, layer: dict, tied: bool = True, emb_std: float = 0.02,
                checkpoint: bool = False) -> HybridLM:
    lc = LayerConfig(d_model=d_model, n_heads=n_heads, **layer)
    return HybridLM(HybridConfig(vocab=VOCAB, d_model=d_model, n_layers=n_layers, n_heads=n_heads, window=window, layer=lc,
                                 attn_every=attn_every, pos="rope", tied=tied, emb_std=emb_std, checkpoint=checkpoint))


BACK_BINS = (0, 64, 256, 512, 1024, 2048)                                        # distance to the previous occurrence, for the by-distance report


@torch.no_grad()
def evaluate(model: HybridLM, split: str, T: int, B: int, device, probes: dict | None = None, max_windows: int | None = None, by_distance: bool = False) -> dict:
    """Perplexity over the scored halves, and per probe type: covered probes, top-1 accuracy and mean log-probability of the value
    (with `by_distance`, also per bin of the distance to the previous occurrence, recall and re-binding probes pooled)."""
    ids = tokens(split); W = windows(len(ids), T)[:max_windows]; nll = 0.0; count = 0
    hits = {t: 0 for t in TYPES}; logp = {t: 0.0 for t in TYPES}; cov = {t: 0 for t in TYPES}
    dh = {b: 0 for b in BACK_BINS[:-1]}; dc = {b: 0 for b in BACK_BINS[:-1]}
    if probes is not None:
        ppos, ptgt, ptyp, pback = (probes[k] for k in ("pos", "target", "type", "back"))
    model.eval()
    for b0 in range(0, len(W), B):
        chunk = W[b0:b0 + B]; x = torch.stack([torch.from_numpy(np.asarray(ids[s:s + T], dtype=np.int64)) for s, _ in chunk]).to(device)
        y = torch.stack([torch.from_numpy(np.asarray(ids[s + 1:s + T + 1], dtype=np.int64)) for s, _ in chunk]).to(device)
        lp = F.log_softmax(model(x).float(), dim=-1)
        for r, (s, first) in enumerate(chunk):
            tl = lp[r, first:].gather(-1, y[r, first:, None])[:, 0]; nll -= float(tl.sum()); count += len(tl)
            if probes is None: continue
            lo, hi = np.searchsorted(ppos, [s + first, s + T])                    # probes predicted from a scored position of this window
            for k in range(lo, hi):
                off = int(ppos[k]) - s
                if pback[k] > off: continue                                       # its previous occurrence does not start inside this window
                t = TYPES[int(ptyp[k])]; cov[t] += 1; row = lp[r, off]; hit = int(int(row.argmax()) == int(ptgt[k]))
                hits[t] += hit; logp[t] += float(row[int(ptgt[k])])
                if by_distance and t.startswith("recall"):
                    b = max(x for x in BACK_BINS[:-1] if x <= pback[k]); dc[b] += 1; dh[b] += hit
    model.train()
    out = {"val_nll": nll / max(count, 1), "val_ppl": math.exp(nll / max(count, 1)), "val_tokens": count}
    if probes is not None:
        for t in TYPES:
            out[f"probe_{t}_n"] = int((probes["type"] == TYPES.index(t)).sum()); out[f"probe_{t}_covered"] = cov[t]
            out[f"probe_{t}_acc"] = hits[t] / cov[t] if cov[t] else float("nan"); out[f"probe_{t}_logp"] = logp[t] / cov[t] if cov[t] else float("nan")
        if by_distance:
            for lo, hi in zip(BACK_BINS[:-1], BACK_BINS[1:]):
                out[f"recall_back_{lo}_{hi}_covered"] = dc[lo]; out[f"recall_back_{lo}_{hi}_acc"] = dh[lo] / dc[lo] if dc[lo] else float("nan")
    return out


def load_probes(split: str) -> dict:
    return dict(np.load(DATA / f"probes_{split}.npz"))
