"""Synthetic tasks that target what active inference adds: non-stationarity, precision,
and long-range retention. Char-level corpora are reused from flybrain.tasks.text."""
from __future__ import annotations

import numpy as np
import torch

from flybrain.tasks.text import CharCorpus


def nonstationary_mqar(batch: int, seq_len: int, n_pairs: int, vocab: int, rng: np.random.Generator, n_rebind: int = 4):
    """MQAR where `n_rebind` keys are re-bound to NEW values midway; queries after the
    re-binding must return the new value, queries before it the old one. Tests whether a
    memory can overwrite (precision-gated depression) without corrupting other pairs."""
    nk = vocab // 3; keys_v = np.arange(0, nk); vals_v = np.arange(nk, 2 * nk); fill_v = np.arange(2 * nk, vocab)
    x = np.zeros((batch, seq_len), dtype=np.int64); y = np.full((batch, seq_len), -100, dtype=np.int64)
    half = seq_len // 2
    for b in range(batch):
        ks = rng.choice(keys_v, size=n_pairs, replace=False); vs = rng.choice(vals_v, size=n_pairs)
        x[b, 0:2 * n_pairs:2] = ks; x[b, 1:2 * n_pairs:2] = vs
        rest1 = np.arange(2 * n_pairs, half); x[b, rest1] = rng.choice(fill_v, size=len(rest1))
        q1 = rng.choice(rest1, size=min(n_pairs, len(rest1)), replace=False); qi = rng.integers(0, n_pairs, size=len(q1))
        x[b, q1] = ks[qi]; y[b, q1] = vs[qi]
        # re-bind n_rebind keys at the midpoint
        rb = rng.choice(n_pairs, size=n_rebind, replace=False); new_vs = vs.copy(); new_vs[rb] = rng.choice(vals_v, size=n_rebind)
        x[b, half:half + 2 * n_rebind:2] = ks[rb]; x[b, half + 1:half + 2 * n_rebind:2] = new_vs[rb]
        rest2 = np.arange(half + 2 * n_rebind, seq_len); x[b, rest2] = rng.choice(fill_v, size=len(rest2))
        q2 = rng.choice(rest2, size=min(n_pairs, len(rest2)), replace=False); qi2 = rng.integers(0, n_pairs, size=len(q2))
        x[b, q2] = ks[qi2]; y[b, q2] = new_vs[qi2]
    return torch.from_numpy(x), torch.from_numpy(y)


class TopicSwitchStream:
    """Concatenates segments from different corpora (Shakespeare / PTB / WikiText-2) with
    a shared character vocabulary; each training window contains 1-3 topic switches.
    Targets AIF-H3 (arousal / precision reset at surprising boundaries)."""

    def __init__(self, names=("tinyshakespeare", "ptb", "wikitext2"), data_dir=None):
        self.corpora = [CharCorpus(n, data_dir) for n in names]
        chars = sorted(set().union(*[set(c.stoi) for c in self.corpora])); self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.vocab = len(chars)
        self.train = [self._remap(c, c.train, c.stoi) for c in self.corpora]
        self.val = [self._remap(c, c.val, c.stoi) for c in self.corpora]

    def _remap(self, c, data, stoi):
        itos = {i: ch for ch, i in stoi.items()}
        lut = torch.tensor([self.stoi[itos[i]] for i in range(len(itos))], dtype=torch.long)
        return lut[data]

    def batch(self, split, batch, seq_len, rng: np.random.Generator, min_seg: int = 24):
        srcs = self.train if split == "train" else self.val
        x = torch.zeros(batch, seq_len + 1, dtype=torch.long); sw = torch.zeros(batch, seq_len + 1, dtype=torch.bool)
        for b in range(batch):
            pos = 0
            while pos < seq_len + 1:
                c = int(rng.integers(0, len(srcs))); L = int(rng.integers(min_seg, max(min_seg + 1, seq_len // 2)))
                L = min(L, seq_len + 1 - pos); i = int(rng.integers(0, len(srcs[c]) - L - 1))
                x[b, pos:pos + L] = srcs[c][i:i + L]; sw[b, pos] = pos > 0; pos += L
        return x[:, :-1], x[:, 1:], sw[:, 1:]        # inputs, targets, switch-position mask (target-aligned)


def retention_batch(batch: int, seq_len: int, vocab: int, rng: np.random.Generator, n_items: int = 4):
    """Long-range retention: `n_items` (cue, value) pairs at the start, a long filler, then
    cues at the end; target = value. Distance grows with seq_len. Targets AIF-H4."""
    nk = vocab // 3; cue_v = np.arange(0, nk); val_v = np.arange(nk, 2 * nk); fill_v = np.arange(2 * nk, vocab)
    x = np.zeros((batch, seq_len), dtype=np.int64); y = np.full((batch, seq_len), -100, dtype=np.int64)
    for b in range(batch):
        cs = rng.choice(cue_v, size=n_items, replace=False); vs = rng.choice(val_v, size=n_items)
        x[b, 0:2 * n_items:2] = cs; x[b, 1:2 * n_items:2] = vs
        x[b, 2 * n_items:seq_len - n_items] = rng.choice(fill_v, size=seq_len - 3 * n_items)
        order = rng.permutation(n_items); x[b, seq_len - n_items:] = cs[order]; y[b, seq_len - n_items:] = vs[order]
    return torch.from_numpy(x), torch.from_numpy(y)


def same_index_logits(model, x: torch.Tensor, fill_token: int):
    """Logits aligned with SAME-INDEX targets (MQAR / retention: the value is the target at the
    query position). `FlyAI` emits logits[:, t] BEFORE observing x[:, t], so the prediction that
    may use the query token x[:, t] is logits[:, t + 1]; run one extra filler step and drop the
    first prediction. (SeqModel baselines already include x[:, t] in logits[:, t]; do not shift them.)"""
    x_ext = torch.cat([x, torch.full_like(x[:, :1], fill_token)], 1)
    logits, st = model(x_ext)
    return logits[:, 1:], st


# ---------------------------------------------------------------------------------------
# Bayesian wind tunnel: HMM sequences with a closed-form posterior predictive
# ---------------------------------------------------------------------------------------
class HMMWindTunnel:
    """Random HMMs (n_states hidden, vocab observations). For each sequence the exact
    forward-algorithm posterior predictive p(o_{t+1} | o_{<=t}) is computed, so a model's
    predictive distribution can be scored by KL to the true Bayesian predictive
    (Agarwal, Dalal & Misra 2025 'Bayesian wind tunnel'). Each sequence uses a freshly
    sampled HMM from a fixed family so the model must infer the latent state in context.
    Transition rows ~ Dirichlet(alpha_T), emission rows ~ Dirichlet(alpha_E)."""

    def __init__(self, n_states: int = 6, vocab: int = 16, alpha_T: float = 0.3, alpha_E: float = 0.3):
        self.S, self.V, self.aT, self.aE = n_states, vocab, alpha_T, alpha_E

    def sample(self, batch: int, seq_len: int, rng: np.random.Generator):
        x = np.zeros((batch, seq_len), dtype=np.int64); post = np.zeros((batch, seq_len, self.V), dtype=np.float64)
        for b in range(batch):
            T = rng.dirichlet(np.full(self.S, self.aT), size=self.S); E = rng.dirichlet(np.full(self.V, self.aE), size=self.S)
            pi = np.full(self.S, 1.0 / self.S); s = rng.choice(self.S, p=pi); alpha = pi.copy()
            for t in range(seq_len):
                post[b, t] = alpha @ E                          # predictive for o_t given o_<t
                o = rng.choice(self.V, p=E[s]); x[b, t] = o
                alpha = alpha * E[:, o]; alpha = alpha / alpha.sum(); alpha = alpha @ T
                s = rng.choice(self.S, p=T[s])
        return torch.from_numpy(x), torch.from_numpy(post.astype(np.float32))


def predictive_kl(logits: torch.Tensor, post: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """KL(true predictive || model predictive) in bits at every position: tensor (B, T) (average it yourself)."""
    logp = torch.log_softmax(logits, -1)
    return (post * (torch.log(post + eps) - logp)).sum(-1) / np.log(2)
