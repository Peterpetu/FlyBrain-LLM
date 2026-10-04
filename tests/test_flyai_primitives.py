import torch, numpy as np
from flyai.primitives import DivisiveNorm, GammaPrecision, DaleCore


def test_divisive_norm_bounded():
    dn = DivisiveNorm(); z = torch.randn(4, 100) * 50
    y = dn(z); assert (y >= 0).all() and y.mean(-1).max() < 1.5


def test_gamma_precision_tracks_variance():
    gp = GammaPrecision(rho=0.95); st = gp.init(torch.zeros(1))
    for scale in (1.0, 0.1):
        for _ in range(300):
            st, pi = gp.update(st, torch.randn(1, 16) * scale)
        assert abs(float(pi) * scale ** 2 - 1.0) < 0.35


def test_dale_core_signs_and_gradients():
    core = DaleCore(64, 8, inh_frac=0.3, fan_in=8)
    W = core.W_eff().detach()
    assert (W[core.sign < 0] <= 0).all() and (W[core.sign > 0] >= 0).all()
    assert int((core.sign < 0).sum()) == 19
    x = torch.randn(2, 10, 8); y, h = core(x); y.sum().backward()
    assert core.V.grad is not None and torch.isfinite(y).all()


def test_lyapunov_monotone_in_gain_and_balanced_init():
    torch.manual_seed(0)
    core = DaleCore(128, 8, fan_in=16); x = torch.randn(1, 200, 8) * 0.1
    assert core.ei_column_imbalance() < 0.05 and abs(core.spectral_radius() - 1.0) < 1e-3
    lams = []
    for gain in (0.3, 1.0, 6.0):
        core.gain.fill_(gain); core.lam.zero_(); h = torch.zeros(1, 128)
        with torch.no_grad():
            for t in range(200):
                core.lyapunov_step(h, x[:, t]); h = core.step(h, x[:, t])
        lams.append(float(core.lam))
    assert lams[0] < lams[1] < lams[2] and lams[0] < 0 < lams[2]


def test_homeostasis_converges_from_both_sides():
    """The dithered extremum-seeking controller brings the EMA Lyapunov estimate into a tight band
    around the target from a contractive and from an expansive start, and the base gains it settles
    at agree (the crossing of lambda(g) is unique in this regime)."""
    torch.manual_seed(0)
    x = torch.randn(1, 3000, 8) * 0.1
    gains = []
    for g0 in (0.2, 4.0):
        core = DaleCore(128, 8, fan_in=16, init_gain=g0)
        with torch.no_grad():
            core(x, regulate=True)
        assert abs(float(core.lam) - core.target_lambda) < 0.06, (g0, float(core.lam), float(core.base_gain))
        gains.append(float(core.base_gain))
    assert abs(gains[0] - gains[1]) < 0.35 * max(gains), gains
