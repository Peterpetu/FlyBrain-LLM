import torch
from flyai.lm import make_lm
from flyai.block import FlyAIBlock, BlockConfig


def _blk(**kw):
    base = dict(d=32, n_heads=4, n_groups=4, n_kc=64, n_pn=12, d_v=8, n_rings=2, n_ring_units=8, core_fan_in=8, fan_in=4, chunk=4)
    base.update(kw); return FlyAIBlock(BlockConfig(**base))


def test_hybrid_lms_train_step():
    torch.manual_seed(0); x = torch.randint(0, 100, (2, 12))
    for mixer in ("flyai", "gdn", "attention"):
        m = make_lm("tiny", mixer, vocab=100)
        loss = m.loss(x); loss.backward(); assert torch.isfinite(loss), mixer
        if mixer != "attention":
            assert [l.kind for l in m.layers] == [mixer, mixer, mixer, "attention"]


def test_block_arms_run_and_backprop():
    x = torch.randn(2, 9, 32)
    for kw in [dict(fusion="closed"), dict(fusion="learned"), dict(fusion="loop", K_max=2), dict(use_attractor=False), dict(use_memory=False),
               dict(arousal="none"), dict(pi3_source="gamma"), dict(memory_gate="learned")]:
        b = _blk(**kw); y, st = b(x); y.sum().backward(); assert torch.isfinite(y).all(), kw


def test_chunked_checkpoint_matches_unchunked():
    torch.manual_seed(0); x = torch.randn(2, 10, 32)
    b1 = _blk(chunk=4, regulate_core=False); b2 = _blk(chunk=100, regulate_core=False); b2.load_state_dict(b1.state_dict())
    y1, _ = b1(x); y2, _ = b2(x); assert torch.allclose(y1, y2, atol=1e-5)
    (y1.sum()).backward(); (y2.sum()).backward()
    assert torch.allclose(b1.td.weight.grad, b2.td.weight.grad, atol=1e-4)


def test_flyai_block_causal_and_state_carry():
    torch.manual_seed(0)
    b = _blk(regulate_core=False); b.eval(); x = torch.randn(1, 10, 32); y = x.clone(); y[0, 6] += 1.0
    with torch.no_grad():
        lx, _ = b(x); ly, _ = b(y); assert torch.allclose(lx[:, :6], ly[:, :6], atol=1e-5)
        full, _ = b(x); a, st = b(x[:, :5]); c, _ = b(x[:, 5:], st)
    assert torch.allclose(full, torch.cat([a, c], 1), atol=1e-5)


def test_closed_form_is_parameter_matched_vs_gdn():
    f = make_lm("125m", "flyai", vocab=50257); g = make_lm("125m", "gdn", vocab=50257)
    nf, ng = f.n_params(), g.n_params()
    assert abs(nf - ng) / ng < 0.08, (nf, ng)


def test_delayed_write_chunk1_equals_sequential_and_chunk64_runs():
    torch.manual_seed(0); x = torch.randn(2, 9, 32)
    seq = _blk(chunk=100, regulate_core=False); d1 = _blk(chunk=1, delayed_write=True, regulate_core=False); d1.load_state_dict(seq.state_dict())
    seq.eval(); d1.eval()
    with torch.no_grad():
        y_seq, st_seq = seq(x); y_d1, st_d1 = d1(x)
    assert torch.allclose(y_seq, y_d1, atol=1e-5) and torch.allclose(st_seq["mem"]["M"], st_d1["mem"]["M"], atol=1e-5)
    d8 = _blk(chunk=4, delayed_write=True); y8, _ = d8(x); y8.sum().backward(); assert torch.isfinite(y8).all()
