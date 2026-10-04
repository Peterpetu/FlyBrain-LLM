import numpy as np
import torch

from flybrain.modules.fast_weights import FastWeightMemory, FWConfig, fast_weight_flops, matched_dense_dkey
from flybrain.modules.ring_attractor import RingAttractor
from flybrain.modules.masked_rnn import MaskedDaleRNN
from flybrain.models import SeqModel, ModelConfig
from flybrain.tasks.mqar import mqar_batch
from flybrain.tasks.compass import compass_batch


def test_fast_weight_variants_run_and_backprop():
    x = torch.randn(2, 10, 32)
    for km in ["dense", "sparse_random", "dense_expanded"]:
        for w in ["delta", "hebb"]:
            for g in ["learned", "error", "const"]:
                m = FastWeightMemory(FWConfig(d_model=32, n_heads=2, d_value=8, key_mode=km, n_pn=12, n_kc=64, write=w, gate=g))
                y, M = m(x); y.sum().backward()
                assert y.shape == (2, 10, 32) and torch.isfinite(y).all()


def test_kwta_sparsity():
    m = FastWeightMemory(FWConfig(d_model=32, n_heads=1, d_value=8, key_mode="sparse_random", n_pn=12, n_kc=100, kwta_frac=0.05))
    k, q = m._codes(torch.randn(3, 5, 32))
    assert ((k != 0).sum(-1) <= 5).all()


def test_read_before_write_cannot_copy_own_value():
    # at t the read uses M_{t-1}; a fresh key must read ~0
    m = FastWeightMemory(FWConfig(d_model=16, n_heads=1, d_value=4, key_mode="dense", d_key=8, gate="const", beta_const=1.0, decay="none"))
    y, M = m(torch.randn(1, 1, 16))
    assert torch.allclose(m.o_proj(m.norm(torch.zeros(1, 1, 1, 4)).reshape(1, 1, 4)), y, atol=1e-5)


def test_compute_matching_monotone():
    cfg = FWConfig(d_model=64, n_heads=2, d_value=16, key_mode="sparse_random", n_pn=60, n_kc=512, fan_in=5, kwta_frac=0.05)
    f = fast_weight_flops(cfg); dk = matched_dense_dkey(cfg)
    assert f["recurrent_effective"] < f["recurrent_dense"] and dk >= 4


def test_ring_modes():
    x = torch.randn(2, 7, 16)
    for mode in ["canonical", "unstructured"]:
        r = RingAttractor(16, mode=mode, n_units=48); y, h = r(x); y.sum().backward()
        assert y.shape == (2, 7, r.out_dim)


def test_masked_rnn_dale_sign():
    import scipy.sparse as sp
    A = sp.random(20, 20, density=0.3, format="csr", random_state=0); A.setdiag(0); A.eliminate_zeros()
    sign = np.where(np.arange(20) < 6, -1, 1)
    m = MaskedDaleRNN(8, A, sign, 20); W = m.effective_W().detach().numpy()
    assert (W[sign < 0] <= 0).all() and (W[sign > 0] >= 0).all()
    assert ((W != 0) <= (A.toarray() > 0)).all()


def test_tasks_and_model():
    rng = np.random.default_rng(0)
    x, y = mqar_batch(4, 32, 4, 24, rng); assert x.shape == (4, 32) and (y != -100).sum() > 0
    x2, y2 = compass_batch(4, 32, rng); assert y2.max() < 8
    m = SeqModel(ModelConfig(vocab=24, d_model=32, n_layers=1, mixer="fastweight", mixer_kwargs=dict(n_heads=1, d_value=8, key_mode="dense", d_key=8)))
    assert m(x).shape == (4, 32, 24)


def test_lowrank_path_matches_recurrent():
    torch.manual_seed(0)
    for km in ["dense", "sparse_random"]:
        for g in ["error", "learned"]:
            m = FastWeightMemory(FWConfig(d_model=16, n_heads=2, d_value=4, key_mode=km, d_key=8, n_pn=8, n_kc=40, kwta_frac=0.1, gate=g, decay="learned"))
            x = torch.randn(2, 40, 16)
            y1, M1 = m._forward_lowrank(x); y2, M2 = m(x)
            assert (y1 - y2).abs().max() < 1e-4 and (M1 - M2).abs().max() < 1e-4
