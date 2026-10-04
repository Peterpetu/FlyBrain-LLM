"""flyprinc.lmtasks: the token format, the strata as defined in the module docstring (checked from the tokens alone),
the contradiction-free stream, and the reproducible batch stream (identical with and without worker processes)."""
import numpy as np
import pytest

from flyprinc.lmtasks import LMTaskConfig, STRATA, make_batch, batch_at, BatchStream


def presentations_and_queries(cfg, toks, tgt, strat):
    """Walk one sequence: per query, the values shown by the key's earlier presentations, the target and the stratum."""
    shown, out = {}, []
    for p in range(len(toks)):
        t = int(toks[p])
        if t == cfg.Q:
            key, val = int(toks[p + 1]), int(toks[p + 2]) - cfg.n_keys
            assert tgt[p + 1] == toks[p + 2] and strat[p + 1] in range(len(STRATA))       # the target is the value token that follows
            out.append((shown.get(key, []).copy(), val, int(strat[p + 1])))
        elif t < cfg.n_keys and (p == 0 or toks[p - 1] != cfg.Q):
            shown.setdefault(t, []).append(int(toks[p + 1]) - cfg.n_keys)
    return out


@pytest.mark.parametrize("T,keys", [(256, 24), (512, 64)])
def test_format_and_strata_follow_their_definitions(T, keys):
    cfg = LMTaskConfig(T=T, keys_per_seq=keys, window=T, p_rebind=0.2, p_outlier=0.15, far_fraction=0.0)
    toks, tgt, strat, _ = make_batch(cfg, 32, np.random.default_rng(3)); seen = set(); n = 0
    assert toks.min() >= 0 and toks.max() < cfg.vocab and ((tgt >= 0) == (strat >= 0)).all()
    for b in range(32):
        for vals, target, s in presentations_and_queries(cfg, toks[b], tgt[b], strat[b]):
            n += 1; seen.add(s); assert vals, "a query before any presentation of its key"
            assert (vals[-1] != target) == (s == 3)                                            # after_outlier <=> the last shown value is not the current one
            if s == 0: assert vals == [target]                                                  # single: one presentation, showing the target
            if s == 1: assert len(vals) >= 2 and vals[0] == target and vals[-1] == target       # confirmed: never changed
            if s == 2: assert vals[-1] == target and any(v != target for v in vals[:-1])        # rebound: changed, the new value shown last
    assert n > 100 and seen == {0, 1, 2, 3}


def test_contradiction_free_stream_is_consistent():
    cfg = LMTaskConfig(T=512, keys_per_seq=64, window=512, p_rebind=0.0, p_outlier=0.0, far_fraction=0.0)
    toks, tgt, strat, _ = make_batch(cfg, 16, np.random.default_rng(4))
    for b in range(16):
        for vals, target, s in presentations_and_queries(cfg, toks[b], tgt[b], strat[b]):
            assert set(vals) == {target} and s in (0, 1)


def test_batch_stream_is_reproducible_and_independent_of_workers():
    cfg = LMTaskConfig(T=256, keys_per_seq=24, window=256, p_rebind=0.2, p_outlier=0.15, far_fraction=0.0)
    inline = list(BatchStream(cfg, 4, 11, 6, workers=0)); parallel = list(BatchStream(cfg, 4, 11, 6, workers=2, ahead=3))
    assert len(inline) == len(parallel) == 6
    for a, b in zip(inline, parallel):
        for x, y in zip(a, b): assert np.array_equal(x, y)
    assert all(np.array_equal(x, y) for x, y in zip(inline[4], batch_at(cfg, 4, 11, 4)))              # batch s is batch_at(seed, s)
    assert not np.array_equal(inline[0][0], inline[1][0]) and not np.array_equal(inline[0][0], batch_at(cfg, 4, 12, 0)[0])
