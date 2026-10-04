import torch, numpy as np
from flyai.memory import MBMemory
from flyai.attractor import AttractorBank
from flyai.primitives import NoveltyMBON, param_groups, after_optimizer_step, DaleCore


def test_mb_memory_hetero_association_and_gate_direction():
    torch.manual_seed(0)
    m = MBMemory(16, n_pn=12, n_kc=200, d_v=8, n_heads=4, gate="precision")
    a, b = torch.randn(3, 16), torch.randn(3, 16)
    st = m.init_state(3, a)
    r0 = m.read(st, a)
    st = m.write(st, a, b)                          # pair key(a) with content(b)
    r1 = m.read(st, a)                              # reading with a now predicts b's content
    assert (r1 - r0).abs().sum() > 0 and torch.isfinite(r1).all()
    assert st["beta"].shape == (3, 4) and st["nov"].shape == (3,) and (st["nov"] <= 1).all()
    # gate rises with surprise: a larger error must give a larger beta (ga > 0)
    st2 = m.write(st, a, b); st3 = m.write(st, a, b * 10)
    assert (st3["beta"] >= st2["beta"]).all()
    (r1.sum()).backward(); assert m.u_proj.weight.grad is not None
    # decay priors are log-spaced across compartments
    alphas = torch.sigmoid(m.alpha_proj.bias).detach()
    assert alphas[0] < alphas[1] < alphas[2] < alphas[3]


def test_shared_code_is_identical_across_heads():
    m = MBMemory(16, n_pn=12, n_kc=100, d_v=8, n_heads=3, shared_code=True)
    v, i = m.code(torch.randn(2, 16)); assert torch.equal(i[:, 0], i[:, 2])
    m2 = MBMemory(16, n_pn=12, n_kc=100, d_v=8, n_heads=3, shared_code=False)
    v2, i2 = m2.code(torch.randn(2, 16)); assert not torch.equal(i2[:, 0], i2[:, 2])


def test_novelty_mbon_decreases_on_repeat():
    nov = NoveltyMBON(); w = nov.init(1, 50, torch.zeros(1)); idx = torch.tensor([[1, 2, 3]])
    n0 = nov.read(w, idx); w = nov.update(w, idx); n1 = nov.read(w, idx)
    assert float(n0) == 1.0 and float(n1) < float(n0)


def test_attractor_bump_moves_and_amplitude_precision():
    torch.manual_seed(0)
    a = AttractorBank(8, n_rings=2, n_units=8)
    h = a.init_state(1, torch.zeros(1)); x = torch.zeros(1, 8)
    with torch.no_grad():
        a.vel.weight.zero_(); a.vel.bias.fill_(3.0); a.drive.weight.zero_(); a.drive.bias.zero_()
        angs = []
        for t in range(40):
            h = a.step(h, x); c, s, amp = a.polar(h); angs.append(float(torch.atan2(s[0, 0], c[0, 0])))
    assert abs(angs[-1] - angs[0]) > 0.3 and torch.isfinite(h).all()
    assert a.readout(h).shape == (1, a.out_dim) and a.precision(h).shape == (1, 2) and (a.precision(h) > 0).all()


def test_param_groups_and_rebalance():
    core = DaleCore(64, 8, fan_in=8)
    groups = param_groups(core, 0.1); names = {id(p) for p in groups[1]["params"]}
    assert id(core.V) in names and id(core.logit_tau) in names and id(core.W_in.weight) not in names
    with torch.no_grad(): core.V.add_(torch.randn_like(core.V) * 0.5)
    after_optimizer_step(core); assert core.ei_column_imbalance() < 1e-4
