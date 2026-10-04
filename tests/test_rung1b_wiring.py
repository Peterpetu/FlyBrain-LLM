"""Rung 1b's per-seed wiring (experiments/p3_s1b_sparse/run.py: rewire): the fixed expansion keeps its construction
(fan_in inputs per cell, weight 1/fan_in), differs between seeds and between layers, is reproducible for a seed, reaches
the sequential copy used by the correctness gate, and leaves every other arm untouched."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "p3_s1b_sparse"))
import run  # noqa: E402
from flyprinc.hybrid import chunk_error  # noqa: E402


def model_for(arm, seed):
    S = dict(run.SMOKE, keys=8); run.RUN.update(device="cpu", compile=False); torch.manual_seed(seed)
    model, _ = run.build(dict(run.arms(S)[arm]), S, 1282, 64, "cpu"); run.rewire(model, seed); return model


def test_wiring_is_drawn_per_seed_and_layer_with_the_same_construction():
    a, b, a2 = model_for("sparse", 10), model_for("sparse", 11), model_for("sparse", 10)
    Ea = [l.E for l in a.memory_layers()]; Eb = [l.E for l in b.memory_layers()]; c = a.memory_layers()[0].c
    for E in Ea + Eb:
        assert ((E > 0).sum(1) == c.fan_in).all() and torch.allclose(E[E > 0], torch.full_like(E[E > 0], 1.0 / c.fan_in))
    assert not torch.equal(Ea[0], Ea[1]) and not torch.equal(Ea[0], Eb[0])                         # layers differ, seeds differ
    assert all(torch.equal(x.E, y.E) for x, y in zip(a.memory_layers(), a2.memory_layers()))         # reproducible for a seed
    x = torch.randn(2, 40, run.SMOKE["d_model"]); assert chunk_error(a.memory_layers()[0], x) < 1e-4  # the gate's sequential copy receives the wiring


def test_other_arms_are_untouched():
    for arm in ("dense", "product", "sparse_learned"):
        m1 = model_for(arm, 10); S = dict(run.SMOKE, keys=8); torch.manual_seed(10); m2, _ = run.build(dict(run.arms(S)[arm]), S, 1282, 64, "cpu")
        assert all(torch.equal(p, q) for p, q in zip(m1.state_dict().values(), m2.state_dict().values()))
