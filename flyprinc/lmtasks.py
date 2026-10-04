"""Token-level streams for the system-level rungs (docs/phase3/DESIGN.md section 4): multi-query
associative recall with silent re-binding and one-off outliers, with the gaps controlled so that a
pre-registered fraction of the decisive writes falls BEYOND the sliding-window attention.

Format (all integers): write = [key, value]; query = [Q, key, value] where the loss is on the
prediction of `value` from the position of `key`; filler tokens elsewhere.

Each key has a current value. Presentations show the current value; with p_rebind per ELIGIBLE
presentation (not the first, not the last, not one followed by a query, not a forced confirmation)
a presentation silently changes the current value, and the next presentation always confirms it
before any query; with p_outlier a presentation shows a wrong value once without changing the
current value. Effective rates per placed presentation are therefore below the nominal ones and
are reported by `rates()`. The target of a query is the current value at query time. Strata of a
query: 'single' (one presentation so far), 'confirmed' (>= 2 consistent presentations, no change),
'rebound' (a change followed by >= 1 confirming presentation), 'after_outlier' (the last
presentation was an outlier); 'far' when the gap from the last presentation to the query exceeds
the window."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STRATA = ("single", "confirmed", "rebound", "after_outlier")


@dataclass
class LMTaskConfig:
    n_keys: int = 512
    n_vals: int = 512
    n_fill: int = 256
    T: int = 2048
    keys_per_seq: int = 48
    presentations: tuple = (2, 5)        # per key, uniform inclusive
    queries_per_key: tuple = (1, 2)
    p_rebind: float = 0.15
    p_outlier: float = 0.10
    window: int = 512
    far_fraction: float = 0.5             # fraction of queries whose gap to the last presentation is forced beyond the window

    @property
    def vocab(self): return self.n_keys + self.n_vals + self.n_fill + 2
    @property
    def Q(self): return self.vocab - 2
    @property
    def PAD(self): return self.vocab - 1


def make_stream(cfg: LMTaskConfig, rng: np.random.Generator, contradiction_free: bool = False):
    """One sequence: tokens (T,), targets (T,) with -1 where no loss, stratum (T,) in 0..3 or -1, far (T,) bool."""
    c = cfg; T = c.T; toks = rng.integers(c.n_keys + c.n_vals, c.n_keys + c.n_vals + c.n_fill, size=T); tgt = np.full(T, -1); strat = np.full(T, -1); far = np.zeros(T, bool)
    used = bytearray(T)                                                  # occupancy as plain bytes: the same decisions as a numpy mask, without a numpy call per try
    def place(length, lo, hi):
        lo = max(lo, 0); hi = min(hi, T - length)
        if hi < lo: return None
        for _ in range(64):
            p = int(rng.integers(lo, hi + 1))
            if not any(used[p:p + length]): used[p:p + length] = b"\x01" * length; return p
        return None
    for key in rng.choice(c.n_keys, c.keys_per_seq, replace=False):
        n_pres = int(rng.integers(c.presentations[0], c.presentations[1] + 1)); n_q = int(rng.integers(c.queries_per_key[0], c.queries_per_key[1] + 1))
        q_after = set(rng.integers(1, n_pres + 1, size=n_q).tolist())                 # query after the i-th presentation (i >= 1)
        cur = int(rng.integers(c.n_vals)); pending = False; changed = False; n_since = 0; last_outlier = False
        pos = int(rng.integers(0, T // 4)); last_pos = None
        for i in range(1, n_pres + 1):
            kind = "cons"; r = rng.random()
            if pending: kind = "confirm"; pending = False
            elif not contradiction_free and i > 1:
                can_rebind = i < n_pres and (i not in q_after)                     # a change is always confirmed by the next presentation before any query
                if r < c.p_rebind and can_rebind: kind = "rebind"; pending = True
                elif r < c.p_rebind + c.p_outlier: kind = "outlier"
            if kind == "rebind": cur = (cur + int(rng.integers(1, c.n_vals))) % c.n_vals; changed = True; n_since = 1; last_outlier = False
            elif kind == "outlier": last_outlier = True
            else: n_since += 1; last_outlier = False
            shown = cur if kind != "outlier" else (cur + int(rng.integers(1, c.n_vals))) % c.n_vals
            p = place(2, pos + int(rng.integers(2, 64)), pos + 320)
            if p is None: break
            toks[p], toks[p + 1] = key, c.n_keys + shown; pos = p + 2; last_pos = pos
            if i in q_after:
                if rng.random() < c.far_fraction: lo, hi = pos + c.window + 8, pos + c.window + 512
                else: lo, hi = pos + 4, pos + min(c.window - 8, 256)
                q = place(3, lo, hi)
                if q is None: continue
                toks[q], toks[q + 1], toks[q + 2] = c.Q, key, c.n_keys + cur; tgt[q + 1] = c.n_keys + cur
                strat[q + 1] = 3 if last_outlier else (2 if (changed and n_since >= 2) else (0 if n_since == 1 else 1)); far[q + 1] = (q - last_pos) > c.window
                pos = q + 3
    return toks, tgt, strat, far


def make_batch(cfg: LMTaskConfig, B: int, rng: np.random.Generator, contradiction_free: bool = False):
    out = [make_stream(cfg, rng, contradiction_free) for _ in range(B)]
    return tuple(np.stack([o[i] for o in out]) for i in range(4))


def batch_at(cfg: LMTaskConfig, B: int, seed: int, step: int):
    """Training batch `step` of a run: drawn with its own generator seeded by (seed, step), so batches can be made ahead,
    by any number of workers, and the stream is the same."""
    return make_batch(cfg, B, np.random.default_rng([seed, step]))


class BatchStream:
    """Batches 0..steps-1 of a run in order, generated `ahead` batches in advance by `workers` processes (the generator
    is pure Python, so at long T it would otherwise take longer than the GPU step). workers = 0 generates in-line;
    the batches are identical either way. Workers are spawned processes that re-import the calling script, so a script
    using workers must keep its work under `if __name__ == "__main__":`."""
    def __init__(self, cfg: LMTaskConfig, B: int, seed: int, steps: int, workers: int = 3, ahead: int | None = None):
        self.args = (cfg, B, seed); self.steps = steps; self.workers = workers; self.ahead = ahead or 4 * max(1, workers)

    def __iter__(self):
        if self.workers <= 0:
            for s in range(self.steps): yield batch_at(*self.args, s)
            return
        from collections import deque
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(self.workers)
        try:
            q, nxt = deque(), 0
            for _ in range(self.steps):
                while nxt < self.steps and len(q) < self.ahead:
                    q.append(pool.submit(batch_at, *self.args, nxt)); nxt += 1
                yield q.popleft().result()
        finally:                                                          # also when the consumer stops early (divergence, a stop gate)
            pool.shutdown(wait=False, cancel_futures=True)


def rates(cfg: LMTaskConfig, B: int = 64, seed: int = 0) -> dict:
    """Measured event rates per placed presentation and the query composition (reported in the pre-registration)."""
    rng = np.random.default_rng(seed); toks, tgt, strat, far = make_batch(cfg, B, rng); m = tgt >= 0
    comp = {f"{s}_{d}": int(((strat == i) & m & (far if d == "far" else ~far)).sum()) for i, s in enumerate(STRATA) for d in ("far", "near")}
    return {"queries_per_seq": float(m.sum(1).mean()), "far_fraction": float(far[m].mean()), "composition": comp}
