"""Rung-2 building blocks (experiments/p3_s2_lm/lm.py): evaluation windows score every token exactly once with at least T/2
of history; a tiny model trains a few steps and evaluates perplexity and probes on a synthetic split (no downloaded data)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "p3_s2_lm"))
import lm  # noqa: E402

torch.set_num_threads(1)


def test_windows_score_every_token_once_with_half_a_window_of_history():
    n, T = 1000, 128; W = lm.windows(n, T); scored = []
    for s, first in W: scored += list(range(s + first, s + T))
    assert scored == list(range(len(scored))) and len(scored) >= n - T                  # contiguous, no double counting
    assert all(first == T // 2 for s, first in W[1:]) and W[0] == (0, 0)


def test_tiny_model_trains_and_evaluates_on_a_synthetic_split(tmp_path, monkeypatch):
    rng = np.random.default_rng(0); ids = rng.integers(0, 500, size=4000).astype(np.uint16)
    ids[1000:1010] = ids[100:110]                                                           # a repeat the probes point at
    for split in ("train", "validation"): ids.tofile(tmp_path / f"{split}.bin")
    monkeypatch.setattr(lm, "DATA", tmp_path); monkeypatch.setattr(lm, "VOCAB", 500)
    model = lm.build_model(32, 2, 2, 64, 2, dict(d_k=8, d_v=8, d_c=4, states=(), x_only_gate=True, tie_gamma=True, chunk=16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3); r = np.random.default_rng(1)
    for _ in range(3):
        x, y = lm.train_batch(lm.tokens("train"), 2, 64, r); loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    # probe 1's previous occurrence is 50 tokens back (inside its scoring window: 128 tokens, stride 64, so 64-127 tokens of
    # history); probe 2's is 3000 back (inside no window)
    probes = {"pos": np.array([1004, 1005]), "target": ids[[1005, 1006]].astype(np.int64), "type": np.array([1, 1]),
              "back": np.array([50, 3000]), "gap": np.array([48, 3000])}
    out = lm.evaluate(model, "validation", 128, 4, "cpu", probes=probes)
    assert np.isfinite(out["val_ppl"]) and out["val_tokens"] >= 4000 - 128
    assert out["probe_recall_far_n"] == 2 and out["probe_recall_far_covered"] == 1
