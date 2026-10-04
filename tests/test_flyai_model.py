import torch
from flyai.model import FlyAI, FlyAIConfig, lm_loss


def _cfg(**kw):
    base = dict(vocab=20, d=16, n=32, n_kc=64, n_pn=12, d_v=8, n_heads=2, n_rings=2, n_ring_units=8, core_fan_in=8, n_groups=2)
    base.update(kw); return FlyAIConfig(**base)


def test_forward_backward_all_features():
    torch.manual_seed(0)
    m = FlyAI(_cfg(K_max=2)); x = torch.randint(0, 20, (3, 9))
    loss = lm_loss(m, x); loss.backward()
    assert torch.isfinite(loss) and m.emb.weight.grad is not None and m.core.V.grad is not None
    assert 0 <= m.last_diag["K_mean"] <= 2 and 0.25 <= m.last_diag["gamma_mean"] <= 4.0


def test_K0_is_amortised_only_and_differs_from_K2():
    torch.manual_seed(0)
    x = torch.randint(0, 20, (2, 7))
    m0 = FlyAI(_cfg(K_max=0)); m2 = FlyAI(_cfg(K_max=2, adaptive_K=False)); m2.load_state_dict(m0.state_dict())
    l0, _ = m0(x); l2, _ = m2(x)
    assert torch.allclose(l0[:, 0], l2[:, 0]) and not torch.allclose(l0[:, 1:], l2[:, 1:])
    assert m2.last_diag["K_mean"] == 2.0


def test_ablation_arms_run():
    x = torch.randint(0, 20, (2, 6))
    for kw in [dict(use_memory=False), dict(use_attractor=False), dict(memory_gate="learned"), dict(memory_gate="surprise"), dict(memory_gate="const"),
               dict(arousal="none"), dict(arousal="pi0"), dict(arousal="beta"), dict(arousal="learned"), dict(pi3_source="gamma"), dict(regulate_core=False),
               dict(n_groups=1), dict(shared_code=False)]:
        m = FlyAI(_cfg(K_max=1, **kw)); l = lm_loss(m, x); l.backward(); assert torch.isfinite(l), kw


def test_prediction_is_causal():
    torch.manual_seed(0)
    m = FlyAI(_cfg(K_max=1)); m.eval()
    x = torch.randint(0, 20, (1, 8)); y = x.clone(); y[0, 5] = (y[0, 5] + 1) % 20
    with torch.no_grad():
        lx, _ = m(x); ly, _ = m(y)
    assert torch.allclose(lx[:, :6], ly[:, :6], atol=1e-5)


def test_arousal_changes_inference_not_inert():
    """gamma must change the inferred latent (it scales pi_0 relative to the prior), unlike v0."""
    torch.manual_seed(0)
    m = FlyAI(_cfg(K_max=2, adaptive_K=False, arousal="pi0")); m.eval()
    x = torch.randint(0, 20, (2, 6))
    with torch.no_grad():
        m.ar_b.fill_(float(torch.log(torch.tensor(0.3)))); la, _ = m(x)
        m.ar_b.fill_(float(torch.log(torch.tensor(3.0)))); lb, _ = m(x)
    assert not torch.allclose(la[:, 1:], lb[:, 1:], atol=1e-4)


def test_lyapunov_is_estimated_but_gain_untouched_when_regulation_is_off():
    torch.manual_seed(0)
    m = FlyAI(FlyAIConfig(vocab=24, d=16, n=32, K_max=1, n_kc=64, n_pn=8, fan_in=3, d_v=4, n_heads=2, n_rings=2, n_groups=2, regulate_core=False, regulate_every=4)); m.train()
    m.core.gain.fill_(0.5); lm_loss(m, torch.randint(0, 24, (2, 12)))
    assert float(m.core.lam) != 0.0 and float(m.core.gain) == 0.5
