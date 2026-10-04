"""flyprinc.hybrid: exactness of the chunked form, reduction to Gated DeltaNet, nested control, state semantics, training."""
import torch
import torch.nn.functional as F
import pytest

from flyprinc.hybrid import LayerConfig, StateMemoryLayer, HybridConfig, HybridLM, count_params, textbook_gdn, layer_cost

torch.set_num_threads(1)
B, T, D = 2, 64, 32
SMALL = dict(d_model=D, n_heads=2, d_k=8, d_v=8, d_c=4)


def pair(states, chunk, refine=None, const=None, **kw):
    torch.manual_seed(0); cs = LayerConfig(**SMALL, states=states, const_gate=const, chunk=0, **kw)
    seq = StateMemoryLayer(cs); ch = StateMemoryLayer(LayerConfig(**{**cs.__dict__, "chunk": chunk, "refine": refine})); ch.load_state_dict(seq.state_dict())
    return seq, ch


@pytest.mark.parametrize("states", [(), ("nconf", "zc"), ("nconf", "zc", "a")])
def test_chunked_equals_sequential_with_constant_gates(states):
    torch.manual_seed(1); x = torch.randn(B, T, D); seq, ch = pair(states, 8, const=(0.6, 0.4, 0.3))
    ys, sts = seq(x); yc, stc = ch(x)
    assert (ys - yc).abs().max() < 1e-4 and (sts["S"] - stc["S"]).abs().max() < 1e-5          # content path exact (UT transform with decays, decoupled erase)
    for n in ("C", "Zc", "Z", "sig", "ta", "tb"): assert (sts[n] - stc[n]).abs().max() < 1e-4, n   # state paths exact given the gates


def test_reduction_to_textbook_gated_deltanet():
    torch.manual_seed(2); x = torch.randn(B, T, D); cs = LayerConfig(**SMALL, states=(), const_gate=(0.6, 0.4, 0.3)); m = StateMemoryLayer(cs)
    q = F.normalize(m.cq(m.wq(x)).view(B, T, 2, 8).transpose(1, 2), dim=-1); k = F.normalize(m.ck(m.wk(x)).view(B, T, 2, 8).transpose(1, 2), dim=-1)
    v = m.cv(m.wv(x)).view(B, T, 2, 8).transpose(1, 2); alpha = torch.exp(-0.3 / m.tau)[None, :, None].expand(B, 2, T)
    y_ref, S_ref = textbook_gdn(q, k, v, torch.full((B, 2, T), 0.6), torch.full((B, 2, T), 0.4), alpha)
    y_seq, _, _, st = m._forward_sequential(q, k, v, m.wg(x).view(B, T, 2, 16).transpose(1, 2), m.init_state(B, x))
    assert (y_ref - y_seq).abs().max() < 1e-5 and (S_ref - st["S"]).abs().max() < 1e-5



@pytest.mark.parametrize("states", [(), ("nconf", "zc"), ("nconf", "zc", "a")])
def test_chunked_learned_gates_reproduce_sequential_after_refinement(states):
    torch.manual_seed(3); x = torch.randn(B, 61, D)                                                # T not a multiple of the chunk
    errs = {}
    for refine in (0, None):
        seq, ch = pair(states, 8, refine=refine); ys, sts = seq(x); yc, stc = ch(x); errs[refine] = ((ys - yc).norm() / ys.norm()).item()
    assert errs[None] < 1e-4 and errs[None] < errs[0]                                              # the iteration is exact within chunk - 1 passes
    seq, ch = pair(states, 8); ys, _ = seq(x); yc, _ = ch(x)
    seq.zero_grad(); ch.zero_grad(); ys.pow(2).sum().backward(); yc.pow(2).sum().backward()
    g = max((ps.grad - pc.grad).abs().max().item() / (ps.grad.abs().max().item() + 1e-6) for ps, pc in zip(seq.parameters(), ch.parameters()) if ps.grad is not None)
    assert g < 1e-4                                                                                # gradients agree too (measured ~1e-5)
    seq, ch = pair(states, 1); assert ((seq(x)[0] - ch(x)[0]).norm() / seq(x)[0].norm()).item() < 1e-4   # chunk 1 is the sequential form


def test_chunked_refinement_survives_a_sharp_trained_gate():
    """After training the gate can be sharp; the Picard iteration must still converge to the sequential form (verification finding 3)."""
    torch.manual_seed(7); x = torch.randn(B, 128, D); seq, ch = pair(("nconf", "zc"), 16)
    with torch.no_grad():
        for m in (seq, ch):
            for p in m.gate.parameters(): p.mul_(4.0)                                             # a sharp gate
    ys, _ = seq(x); yc, _ = ch(x); assert ((ys - yc).norm() / ys.norm()).item() < 1e-4 and ch.last_passes == 16     # fixed budget: chunk solve rounds


def test_numerics_large_forget_drive_keeps_gradients_finite():
    torch.manual_seed(8); x = torch.randn(B, T, D); m = StateMemoryLayer(LayerConfig(**SMALL, states=("nconf", "zc"), const_gate=(0.6, 0.4, 13.0), chunk=64))
    y, st = m(x); assert torch.isfinite(y).all() and torch.isfinite(st["S"]).all()
    m2 = StateMemoryLayer(LayerConfig(**SMALL, states=("nconf", "zc"), chunk=64))
    with torch.no_grad(): m2.gate[2].bias[2] = 12.0                                               # forget drive far beyond the overflow threshold of a naive exp
    y2, _ = m2(x); y2.pow(2).sum().backward(); assert all(torch.isfinite(p.grad).all() for p in m2.parameters() if p.grad is not None)


def test_tie_gamma_and_non_fresh_state():
    torch.manual_seed(9); x = torch.randn(B, T, D); seq, ch = pair((), 8, tie_gamma=True); _, st = seq(x[:, :20]); st2 = {k: v.clone() for k, v in st.items()}
    ys, _ = seq(x[:, 20:], st); yc, _ = ch(x[:, 20:], st2); assert ((ys - yc).norm() / ys.norm()).item() < 1e-3


def test_nested_control_equal_params_and_ignores_states():
    torch.manual_seed(4); x = torch.randn(B, T, D)
    free = StateMemoryLayer(LayerConfig(**SMALL, states=())); full = StateMemoryLayer(LayerConfig(**SMALL, states=("nconf", "zc")))
    assert count_params(free) == count_params(full)
    y1, st = free(x); st2 = {k: (v + 3.0 if k in ("C", "Zc", "Z") else v) for k, v in free.init_state(B, x).items()}
    y2, _ = free(x, st2); assert torch.allclose(y1, y2)                                             # the free arm cannot see the states
    assert count_params(StateMemoryLayer(LayerConfig(**SMALL, states=("a",), a_floor=False))) == count_params(free)


def test_confirmation_count_and_contradiction_mass_in_the_layer():
    torch.manual_seed(5); m = StateMemoryLayer(LayerConfig(**SMALL, states=("nconf", "zc"), const_gate=(1.0, 1.0, 0.0)))
    k = F.normalize(torch.randn(1, 2, 8), dim=-1); v = F.normalize(torch.randn(1, 2, 8), dim=-1); q = k; xg = torch.zeros(1, 2, 16)
    def run(vals):
        st = m.init_state(1, k); reads = []
        for vt in vals:
            _, nc, zc, st = m._forward_sequential(q[:, :, None], k[:, :, None], vt[:, :, None], xg[:, :, None], st); reads.append((nc[0, 0, 0].item(), zc[0, 0, 0].item()))
        return reads, st
    reads, st = run([v, v, v, -v])
    assert abs(reads[1][0] - 1) < 0.05 and abs(reads[2][0] - 2) < 0.05 and abs(reads[3][0] - 3) < 0.05   # consecutive confirmations counted (read before the write)
    assert all(r[1] == 0 for r in reads)                                                           # no contradiction yet at any read
    _, nc, zc, _ = m._forward_sequential(q[:, :, None], k[:, :, None], v[:, :, None], xg[:, :, None], st)
    assert abs(nc[0, 0, 0].item()) < 0.5 and zc[0, 0, 0].item() > 0                                # the contradiction reset the count and registered in zc


def test_hybrid_lm_trains_and_counts():
    torch.manual_seed(6); lc = LayerConfig(d_model=32, n_heads=2, d_k=8, d_v=8, d_c=4, chunk=8)
    lm = HybridLM(HybridConfig(vocab=50, d_model=32, n_layers=4, n_heads=2, window=8, layer=lc))
    tok = torch.randint(0, 50, (B, T)); logits = lm(tok); loss = F.cross_entropy(logits[:, :-1].reshape(-1, 50), tok[:, 1:].reshape(-1)); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in lm.parameters()) and lm.kind == ["mem", "mem", "mem", "attn"]
    c = layer_cost(lc); cf = layer_cost(LayerConfig(**{**lc.__dict__, "states": ()}))
    assert c["params"] == cf["params"] and c["state_per_head"] > cf["state_per_head"] and c["macs_per_head_token"] > cf["macs_per_head_token"]


def test_x_only_gate_is_chunkable_without_refinement():
    torch.manual_seed(10); x = torch.randn(B, T, D); seq, ch = pair((), 64, x_only_gate=True)
    assert ch.c.passes_max == 0
    ys, _ = seq(x); yc, _ = ch(x); assert ((ys - yc).norm() / ys.norm()).item() < 1e-5 and ch.last_passes == 1


@pytest.mark.parametrize("address", ["dense", "sparse", "product", "sparse_learned", "expand_dense"])
def test_gated_deltanet_gating_is_exact_at_large_chunks_for_every_address(address):
    """Rung 1b's main arms: Gated DeltaNet's gating (input-only gate, erase = write) is exact in one round at any chunk size."""
    kw = {} if address == "dense" else dict(address=address, n_cells=96, pk_n1=8, k_active=6, fan_in=3)
    torch.manual_seed(11); x = torch.randn(B, 150, D)                                               # 150 tokens: two full chunks of 64 and a partial one
    seq, ch = pair((), 64, x_only_gate=True, tie_gamma=True, **kw); ys, _ = seq(x); yc, _ = ch(x)
    assert ch.c.passes_max == 0 and ch.last_passes == 1 and ((ys - yc).norm() / ys.norm()).item() < 1e-5


REF = "tests/data/hybrid_ref_single_trace.pt"


def test_single_store_form_is_frozen_against_the_reference_snapshot():
    """The two-store generalisation must leave the single-store layer bit-for-bit unchanged (parameters, outputs, states)."""
    import os; ref = torch.load(REF if os.path.exists(REF) else os.path.join(os.path.dirname(__file__), "data", "hybrid_ref_single_trace.pt"))
    for states in [(), ("nconf", "zc"), ("nconf", "zc", "a")]:
        for chunk in (0, 8):
            m = StateMemoryLayer(LayerConfig(d_model=32, n_heads=2, d_k=8, d_v=8, d_c=4, states=states, chunk=chunk)); r = ref[f"{states}|{chunk}"]
            assert set(m.state_dict()) == set(r["sd"]) and all(m.state_dict()[k].shape == v.shape for k, v in r["sd"].items())   # same parameter layout
            m.load_state_dict(r["sd"])                                   # the snapshot's weights: fresh initialisation under a seed differs across PyTorch versions and platforms
            torch.manual_seed(1); x = torch.randn(2, 61, 32); y, st = m(x)
            assert (y - r["y"]).abs().max() < 2e-6 and (st["S"] - r["S"]).abs().max() < 2e-6 and (st["C"] - r["C"]).abs().max() < 2e-6
    lm = HybridLM(HybridConfig(vocab=50, d_model=32, n_layers=4, n_heads=2, window=8, layer=LayerConfig(d_model=32, n_heads=2, d_k=8, d_v=8, d_c=4, chunk=8)))
    lm.load_state_dict(ref["lm"]["sd"])
    assert (lm(ref["lm"]["tok"]) - ref["lm"]["logits"]).abs().max() < 2e-6


@pytest.mark.parametrize("states", [(), ("nconf", "zc"), ("nconf", "zc", "a")])
def test_two_store_form_is_exact_and_nested(states):
    torch.manual_seed(11); x = torch.randn(B, 61, D)
    seq, ch = pair(states, 8, n_traces=2); ys, sts = seq(x); yc, stc = ch(x)
    assert sts["S"].shape == (B, 2, 2, 8, 8) and ((ys - yc).norm() / ys.norm()).item() < 1e-4 and (sts["S"] - stc["S"]).abs().max() < 1e-4
    seqc, chc = pair(states, 8, n_traces=2, const=(0.6, 0.4, 0.3)); assert (seqc(x)[0] - chc(x)[0]).abs().max() < 1e-4
    free = StateMemoryLayer(LayerConfig(**SMALL, states=(), n_traces=2)); full = StateMemoryLayer(LayerConfig(**SMALL, states=("nconf", "zc"), n_traces=2))
    assert count_params(free) == count_params(full) and count_params(full) > count_params(StateMemoryLayer(LayerConfig(**SMALL, states=("nconf", "zc"))))
    y1, _ = free(x); st2 = {k: (v + 3.0 if k in ("C", "Zc", "Z") else v) for k, v in free.init_state(B, x).items()}; y2, _ = free(x, st2); assert torch.allclose(y1, y2)
    ys.pow(2).sum().backward(); assert seq.readout[0].weight.grad is not None and seq.gate[0].weight.grad is not None


def test_two_store_lm_trains():
    torch.manual_seed(12); lc = LayerConfig(d_model=32, n_heads=2, d_k=8, d_v=8, d_c=4, chunk=8, n_traces=2)
    lm = HybridLM(HybridConfig(vocab=50, d_model=32, n_layers=4, n_heads=2, window=8, layer=lc)); tok = torch.randint(0, 50, (B, T))
    logits = lm(tok); loss = F.cross_entropy(logits[:, :-1].reshape(-1, 50), tok[:, 1:].reshape(-1)); loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None for p in lm.parameters())


@pytest.mark.parametrize("states,n_traces", [((), 1), (("nconf", "zc"), 1), ((), 2), (("nconf", "zc"), 2), (("nconf", "zc", "a"), 2)])
def test_block_delayed_sensing_is_exact_in_one_round(states, n_traces):
    """sense_block = b: the gate senses the store and states as of each b-token block start. The chunked form with chunk = b
    needs no refinement and must equal its own sequential definition; b = 1 must be the per-token model."""
    torch.manual_seed(16); x = torch.randn(B, 61, D)                                                 # 61: a short last block
    seq, ch = pair(states, 16, n_traces=n_traces, sense_block=16)
    assert ch.c.passes_max == 0
    ys, sts = seq(x); yc, stc = ch(x)
    assert ((ys - yc).norm() / ys.norm()).item() < 1e-5 and all((sts[n] - stc[n]).abs().max() < 1e-4 for n in ("S", "C", "Zc", "ta", "tb")) and ch.last_passes == 1
    seq.zero_grad(); ch.zero_grad(); ys.pow(2).sum().backward(); yc.pow(2).sum().backward()
    g = max((ps.grad - pc.grad).abs().max().item() / (ps.grad.abs().max().item() + 1e-6) for ps, pc in zip(seq.parameters(), ch.parameters()) if ps.grad is not None)
    assert g < 1e-4
    torch.manual_seed(0); exact = StateMemoryLayer(LayerConfig(**SMALL, states=states, n_traces=n_traces)); one = StateMemoryLayer(LayerConfig(**SMALL, states=states, n_traces=n_traces, sense_block=1))
    one.load_state_dict(exact.state_dict()); assert torch.allclose(exact(x)[0], one(x)[0], atol=1e-6)
    with pytest.raises(ValueError): LayerConfig(**SMALL, chunk=16, sense_block=8)


def test_product_keys_equal_the_two_stage_search_and_learn():
    """Product-key addressing: the exact top-k of the summed sub-key scores is the set product-key memory's two-stage search
    finds (top-k per half, then top-k of the k x k candidates); the codebooks receive gradients; the cost table counts active rows."""
    kw = dict(address="product", n_cells=96, pk_n1=8, k_active=6)
    m = StateMemoryLayer(LayerConfig(**SMALL, **kw)); torch.manual_seed(21); x = F.normalize(torch.randn(3, 2, 7, 8), dim=-1)
    z = m.code(x); assert torch.allclose(z.norm(dim=-1), torch.ones(3, 2, 7), atol=1e-5) and ((z > 0).sum(-1) == 6).all()
    s1, s2 = x[..., :4] @ m.C1.t(), x[..., 4:] @ m.C2.t(); t1, i1 = s1.topk(6, -1); t2, i2 = s2.topk(6, -1)
    cand = (t1[..., :, None] + t2[..., None, :]).flatten(-2); cells = (i1[..., :, None] * 12 + i2[..., None, :]).flatten(-2)
    two_stage = cells.gather(-1, cand.topk(6, -1).indices).sort(-1).values
    assert torch.equal(two_stage, z.gt(0).nonzero()[:, -1].view(3, 2, 7, 6).sort(-1).values)
    torch.manual_seed(5); xx = torch.randn(B, 40, D)
    layer = StateMemoryLayer(LayerConfig(**SMALL, **kw, chunk=8)); layer(xx)[0].pow(2).mean().backward()
    assert layer.C1.grad.abs().sum() > 0 and layer.C2.grad.abs().sum() > 0
    for a in ("sparse", "product"):
        cost = layer_cost(LayerConfig(**SMALL, **{**kw, "address": a})); assert cost["active_rows"] == 6 and cost["macs_active_per_head_token"] < cost["macs_per_head_token"]
    dense = layer_cost(LayerConfig(**SMALL)); assert dense["active_rows"] == 8 and dense["macs_active_per_head_token"] == dense["macs_per_head_token"]
    with pytest.raises(ValueError): LayerConfig(**SMALL, address="product", n_cells=96, pk_n1=7, k_active=6)


@pytest.mark.parametrize("address", ["sparse", "sparse_learned", "expand_dense", "product"])
def test_sparse_address_code_and_exactness(address):
    """The Kenyon-cell address: k_active non-negative cells of n_cells, unit norm; the memory stores against it and the
    chunked form stays exact (per-token sensing with refinement, and block-delayed sensing in one round)."""
    kw = dict(address=address, n_cells=128, k_active=8, fan_in=3)
    m = StateMemoryLayer(LayerConfig(**SMALL, **kw)); x = F.normalize(torch.randn(2, 2, 5, 8), dim=-1); z = m.code(x)
    assert z.shape == (2, 2, 5, 128) and torch.allclose(z.norm(dim=-1), torch.ones(2, 2, 5), atol=1e-5)
    if address != "expand_dense": assert ((z > 0).sum(-1) == 8).all() and (z >= 0).all()
    torch.manual_seed(17); xx = torch.randn(B, 45, D)
    for states, extra in (((), {}), (("nconf", "zc", "a"), {"n_traces": 2}), (("nconf", "zc"), {"n_traces": 2, "sense_block": 16})):
        seq, ch = pair(states, 16, **kw, **extra); ys, sts = seq(xx); yc, stc = ch(xx)
        assert sts["S"].shape[-2] == 128 and ((ys - yc).norm() / ys.norm()).item() < 1e-4
    dense = StateMemoryLayer(LayerConfig(**SMALL)); fixed = StateMemoryLayer(LayerConfig(**SMALL, address="sparse", n_cells=128, k_active=8, fan_in=3))
    assert count_params(dense) == count_params(fixed)                                              # the fixed expansion is a buffer
    with pytest.raises(ValueError): LayerConfig(**SMALL, address="sparse", n_cells=16, k_active=32)


def test_two_store_fixed_gates_are_per_store_and_the_stores_differ():
    torch.manual_seed(13); x = torch.randn(B, T, D)
    m = StateMemoryLayer(LayerConfig(**SMALL, states=(), n_traces=2, const_gate=((0.2, 0.2, 0.05), (0.7, 0.7, 0.05)), chunk=8))
    y, st = m(x); assert not torch.allclose(st["S"][:, :, 0], st["S"][:, :, 1]) and torch.isfinite(y).all()
    same = StateMemoryLayer(LayerConfig(**SMALL, states=(), n_traces=2, const_gate=(0.5, 0.5, 0.05), chunk=8)); _, st2 = same(x)
    assert torch.equal(st2["S"][:, :, 0], st2["S"][:, :, 1])                                       # one triple for both stores collapses them: never used as an arm
    with pytest.raises(ValueError): StateMemoryLayer(LayerConfig(**SMALL, n_traces=2, const_gate=((0.2, 0.2, 0.05),)))(x)


def test_checkpointed_model_equals_plain_model_bitwise():
    for states in [(), ("nconf", "zc")]:
        outs = []
        for ck in (False, True):
            torch.manual_seed(14); lc = LayerConfig(d_model=32, n_heads=2, d_k=8, d_v=8, d_c=4, chunk=8, n_traces=2, states=states)
            lm = HybridLM(HybridConfig(vocab=50, d_model=32, n_layers=4, n_heads=2, window=8, layer=lc, checkpoint=ck)); torch.manual_seed(15); tok = torch.randint(0, 50, (B, T))
            for m in lm.memory_layers(): m.monitor = True
            logits = lm(tok); loss = F.cross_entropy(logits[:, :-1].reshape(-1, 50), tok[:, 1:].reshape(-1)); loss.backward()
            outs.append((logits.detach(), [p.grad.clone() if p.grad is not None else None for p in lm.parameters()], [m.last_feats for m in lm.memory_layers()]))
        assert torch.equal(outs[0][0], outs[1][0]) and all((a is None and b is None) or torch.equal(a, b) for a, b in zip(outs[0][1], outs[1][1]))
        assert all(f is not None for f in outs[1][2])                                              # the monitor survives the recomputed forward
