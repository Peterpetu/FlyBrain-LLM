"""flyprinc: config validation, the capacity ordering, the two-trace mechanism, stream semantics, training."""
import numpy as np
import pytest
import torch

from flyprinc import MemConfig, FastMemory, Probe, Vocab, evaluate, train_probe, capacity_stream, rebind_stream, load_stream, consolidation_stream, reversal_stream
from flyprinc.tasks import WRITE, QUERY, SLEEP

torch.set_num_threads(1)
VOC = Vocab(n_keys=512, n_vals=32, d_key=32, d_val=8, seed=3)
SP = dict(d_key=32, d_val=8, address="sparse_random", n_cells=256, k_active=16, fan_in=5)


def test_config_validation():
    with pytest.raises(ValueError): MemConfig(address="nope")
    with pytest.raises(ValueError): MemConfig(n_compartments=2, decays=(0.9,))
    with pytest.raises(ValueError): MemConfig(readout="confirm", n_compartments=1)
    with pytest.raises(ValueError): MemConfig(route="confirmation")
    c = MemConfig(**SP); assert c.rows == 256 and c.active_rows == 16 and c.state_numel == 256 * 8


def test_code_is_kwta_unit_norm_and_dense_is_unit_norm():
    m = FastMemory(MemConfig(**SP)); k = m.code(torch.randn(3, 32))
    assert k.shape == (3, 256) and ((k > 0).sum(-1) == 16).all() and torch.allclose(k.norm(dim=-1), torch.ones(3), atol=1e-5)
    d = FastMemory(MemConfig(d_key=32, d_val=8, address="dense")).code(torch.randn(3, 32))
    assert d.shape == (3, 32) and torch.allclose(d.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_capacity_ordering_sparse_beats_dense_at_high_load():
    rng = np.random.default_rng(0); streams = [capacity_stream(4, 48, VOC, rng) for _ in range(2)]     # 48 pairs >> 16 dense rows
    sparse = evaluate(Probe(MemConfig(**SP), VOC), streams)["acc"]
    dense = evaluate(Probe(MemConfig(d_key=32, d_val=8, address="dense"), VOC), streams)["acc"]
    hebb = evaluate(Probe(MemConfig(**SP, write="hebb"), VOC), streams)["acc"]
    assert sparse > dense + 0.2 and sparse > hebb


def test_two_trace_ignores_one_outlier_and_adopts_a_confirmed_change():
    K, V = VOC.tensors(); m = FastMemory(MemConfig(**SP, n_compartments=2, betas=(0.2, 1.0), readout="confirm"))
    st = m.init_state(1, K); x = K[1:2]; v, out, new = V[3:4], V[9:10], V[5:6]
    with torch.no_grad():
        for _ in range(6): st = m.write(st, x, v)
        st = m.write(st, x, out); assert int((m.read(st, x) @ V.t()).argmax()) == 3          # one outlier: still the old value
        for _ in range(3): st = m.write(st, x, new)
        assert int((m.read(st, x) @ V.t()).argmax()) == 5                                       # three confirmations: the change is adopted


def test_confirmation_count_is_content_based_under_sharing():
    """Another key's confirmed writes on shared rows must not count as confirmations for this key."""
    voc = Vocab(n_keys=512, n_vals=64, d_key=64, d_val=16, seed=3); K, V = voc.tensors()     # the E1/E2 geometry
    m = FastMemory(MemConfig(d_key=64, d_val=16, address="sparse_random", n_cells=1024, k_active=64, fan_in=7, n_compartments=2, betas=(0.2, 1.0), readout="confirm"))
    st = m.init_state(1, K); x, y = K[1:2], K[2:3]
    with torch.no_grad():
        for _ in range(6): st = m.write(st, x, V[3:4])
        st = m.write(st, x, V[9:10])                                       # outlier on x: confirmations reset
        for _ in range(20): st = m.write(st, y, V[7:8])                    # many confirmed writes of ANOTHER key
        k = m.code(x); conf = torch.einsum("br,brv->bv", k, st["conf"]); reads = torch.einsum("br,bcrv->bcv", k, st["M"])
        n_conf = float((conf * torch.nn.functional.normalize(reads[:, 1], dim=-1)).sum())
    assert n_conf < 1.0 and int((m.read(st, x) @ V.t()).argmax()) == 3   # leakage through shared rows stays below one confirmation


def test_stream_semantics():
    rng = np.random.default_rng(1)
    s = capacity_stream(2, 10, VOC, rng); assert (s["kinds"][:10] == WRITE).all() and (s["kinds"][10:] == QUERY).all()
    for b in range(2):
        bind = dict(zip(s["keys"][b, :10].tolist(), s["vals"][b, :10].tolist()))
        assert all(bind[int(k)] == int(v) for k, v in zip(s["keys"][b, 10:], s["vals"][b, 10:]))
    r = rebind_stream(2, 8, 300, VOC, rng, p_change=0.01, p_outlier=0.2); q = r["kinds"] == QUERY
    assert q.any() and set(np.unique(r["tags"][q])) <= {0, 1, 2}
    # a query's answer is the CURRENT binding: the value of the last non-outlier write of that key
    ld = load_stream(2, VOC, rng, n_long=2, n_short=2, interval_long=16, interval_short=4, T=64, p_distractor=0.2)
    assert (ld["kinds"] == QUERY).sum() > 0 and ld["meta"]["distractors"] > 0
    c = consolidation_stream(1, VOC, rng, P_total=60, sleep_every=20); assert (c["kinds"] == SLEEP).sum() == 3
    v = reversal_stream(1, VOC, rng, P=8, n_rev=2, reps=2); assert set(np.unique(v["tags"][v["kinds"] == QUERY])) == {1, 2, 9}


def test_training_moves_learned_parameters_and_fixed_modules_skip():
    rng = np.random.default_rng(2)
    pr = Probe(MemConfig(**SP, gate="learned"), VOC); w0 = pr.mem.gate_mlp[0].weight.clone()
    out = train_probe(pr, lambda r: rebind_stream(2, 8, 400, VOC, r, p_change=0.01), steps=2, lr=1e-2, sigma=0.3)
    assert out["trained"] and not out["diverged"] and not torch.equal(w0, pr.mem.gate_mlp[0].weight)
    assert train_probe(Probe(MemConfig(**SP), VOC), lambda r: capacity_stream(2, 8, VOC, r), steps=2, lr=1e-2)["trained"] is False


def test_learned_gate_state_inputs_equal_params_and_semantics():
    F = torch.nn.functional
    free = FastMemory(MemConfig(**SP, gate="learned")); stm = FastMemory(MemConfig(**SP, gate="learned", gate_inputs=("nconf", "a", "zc")))
    assert sum(p.numel() for p in free.parameters()) == sum(p.numel() for p in stm.parameters())     # nested control: equal parameters
    x = torch.randn(4, 32); v = F.normalize(torch.randn(4, 8), dim=-1); k = stm.code(x)
    st = stm.init_state(4, x); assert "Z" in st and "Zc" in st and "Z" not in free.init_state(4, x)   # fixed modules keep their old state dict
    read = lambda s: torch.einsum("br,brv->bv", k, s["M"][:, 0])
    for _ in range(3): st = stm.write(st, x, v)
    s3 = stm.state_inputs(st, k, read(st))
    assert (s3[:, 0] > 2.5).all() and (s3[:, 2] == 0).all()                                          # three confirmations counted, no contradiction
    st = stm.write(st, x, -v); s4 = stm.state_inputs(st, k, read(st))
    assert (s4[:, 0].abs() < 0.5).all() and (s4[:, 2] > 0).all() and (s4[:, 1] > s3[:, 1]).all()      # reset by the contradiction; zc and a register it
    assert (free.state_inputs(free.init_state(4, x), k, v) == 0).all() and s3.shape == (4, 4)          # the free arm's slots read 0
    with pytest.raises(ValueError): MemConfig(**SP, gate="const", gate_inputs=("nconf",))
    with pytest.raises(ValueError): MemConfig(**SP, gate="learned", gate_inputs=("bogus",))


def test_learned_readout_over_two_traces_equal_params_and_trains():
    F = torch.nn.functional
    base = dict(SP, gate="learned", n_compartments=2, readout="learned")
    free = FastMemory(MemConfig(**base)); stm = FastMemory(MemConfig(**base, gate_inputs=("nconf", "zc")))
    assert sum(p.numel() for p in free.parameters()) == sum(p.numel() for p in stm.parameters())
    with pytest.raises(ValueError): MemConfig(**SP, n_compartments=1, readout="learned")
    x = torch.randn(4, 32); v = F.normalize(torch.randn(4, 8), dim=-1); st = stm.init_state(4, x)
    for _ in range(3): st = stm.write(st, x, v)
    y = stm.read(st, x); assert y.shape == (4, 8) and torch.isfinite(y).all()
    (y - v).pow(2).sum().backward(); assert stm.readout_mlp[0].weight.grad is not None and stm.gate_mlp[0].weight.grad is not None
