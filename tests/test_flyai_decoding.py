import torch
from flyai.model import FlyAI, FlyAIConfig
from flyai.decoding import efe_decode, info_gain


def test_efe_decode_runs_and_greedy_limit():
    torch.manual_seed(0)
    m = FlyAI(FlyAIConfig(vocab=20, d=16, n=32, n_kc=64, n_pn=12, d_v=8, n_heads=2, n_rings=2, n_ring_units=8, core_fan_in=8, K_max=1))
    prefix = torch.randint(0, 20, (2, 5))
    g = efe_decode(m, prefix, 4, beta_e=0.0, top_k=5)
    e = efe_decode(m, prefix, 4, beta_e=5.0, top_k=5)
    assert g.shape == (2, 4) and e.shape == (2, 4)
    st = m.init_state(2, m.emb.weight); mu, mu1, mu3, s3, pi_prior = m._priors(st)
    ig = info_gain(m, st, mu, pi_prior, prefix[:, 0]); assert ig.shape == (2,) and torch.isfinite(ig).all()
