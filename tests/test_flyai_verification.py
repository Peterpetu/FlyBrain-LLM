"""Verification tests: each one asserts a v0.2 spec equation or invariant that the earlier smoke tests
did not pin (docs/aif/VERIFICATION.md). Tiny sizes, one thread."""
import itertools

import numpy as np
import pytest
import torch

from flyai.primitives import DaleCore, GammaPrecision, NoveltyMBON, param_groups
from flyai.memory import MBMemory
from flyai.block import FlyAIBlock, BlockConfig
from flyai.model import FlyAI, FlyAIConfig, lm_loss
from flyai.decoding import efe_decode
from flyai.tasks import HMMWindTunnel, predictive_kl, nonstationary_mqar, retention_batch
from flybrain.harness.report import analyse
from flybrain.harness.stats import compare

torch.set_num_threads(1)
TINY_BLOCK = dict(d=16, n_groups=2, n_heads=2, n_kc=64, n_pn=8, fan_in=3, d_v=4, n_rings=2, chunk=4, regulate_core=False, seed=0)
TINY_MODEL = dict(vocab=24, d=16, n=32, n_kc=64, n_pn=8, fan_in=3, d_v=4, n_heads=2, n_rings=2, n_groups=2, regulate_core=False)


# ---------------------------------------------------------------- primitives
def test_dale_core_spec_constants_and_no_self_loops():
    core = DaleCore(200, 16, inh_frac=0.308, fan_in=32, seed=1)
    assert int((core.sign < 0).sum()) == round(0.308 * 200)
    fan = (core.mask != 0).sum(0); inh = ((core.mask != 0) & (core.sign[:, None] < 0)).sum(0)
    assert (fan == 32).all() and (inh == round(0.308 * 32)).all()          # exact stratified fan-in
    assert float(core.mask.diag().sum()) == 0.0                              # no self-loops
    assert abs(core.spectral_radius() - 1.0) < 1e-4 and core.ei_column_imbalance() < 1e-6
    We = core.W_eff()
    for i in range(200):                                                    # Dale: one sign per SOURCE row
        row = We[i][We[i] != 0]; assert (row > 0).all() or (row < 0).all()


def test_gamma_precision_is_bounded_and_never_overflows():
    gp = GammaPrecision(rho=0.98); st = gp.init(torch.ones(2, 3))
    for _ in range(20000):                                                  # zero error for 20k steps: b would underflow without the floor
        st, pi = gp.update(st, torch.zeros(2, 3, 8))
    assert torch.isfinite(pi).all() and (pi == GammaPrecision.PI_MAX).all()
    st = gp.init(torch.ones(1, 1)); st, pi = gp.update(st, torch.full((1, 1, 8), 1e4))
    assert (pi >= GammaPrecision.PI_MIN).all() and torch.isfinite(pi).all()


def test_novelty_recovers_only_where_silent():
    nv = NoveltyMBON(delta=0.5, tick=0.1); w = torch.full((1, 6), 0.5); idx = torch.tensor([[0, 1]])
    w2 = nv.update(w, idx)
    assert torch.allclose(w2[0, 2:], torch.full((4,), 0.55)) and torch.allclose(w2[0, :2], torch.full((2,), 0.25))


def test_param_groups_on_a_real_block():
    blk = FlyAIBlock(BlockConfig(**TINY_BLOCK)); names = {id(p): n for n, p in blk.named_parameters()}
    g = param_groups(blk, 0.1); nod = {names[id(p)] for p in g[1]["params"]}; dec = {names[id(p)] for p in g[0]["params"]}
    assert {"c_level", "core.V", "core.logit_tau", "attractor.log_c", "memory.ga", "memory.gb", "ar_w", "memory.theta"} <= nod
    assert {"td.weight", "core.W_in.weight", "memory.pn.weight", "g3.weight"} <= dec and not any("bias" in n for n in dec)


# ---------------------------------------------------------------- memory
def test_memory_code_is_kwta_unit_norm_and_gate_ladder_semantics():
    torch.manual_seed(0)
    m = MBMemory(16, n_pn=12, n_kc=200, fan_in=3, active_frac=0.05, d_v=8, n_heads=2, gate="precision")
    v, i = m.code(torch.randn(3, 16))
    assert v.shape == (3, 2, 10) and torch.allclose(v.norm(dim=-1), torch.ones(3, 2), atol=1e-5) and (v >= 0).all()
    a, b = torch.randn(3, 16), torch.randn(3, 16); st = m.init_state(3, a)
    with torch.no_grad():
        st1 = m.write(st, a, b); st2 = m.write(st, a, b * 10)               # same expected error, larger current error
    assert (st2["beta"] > st1["beta"]).all()
    # precision vs raw-surprise gate: only the precision gate reacts to a change of the EXPECTED error (pi_slow)
    st_hi = dict(st); st_hi["prec"] = (st["prec"][0] * 100, st["prec"][1])   # 100x larger pi_slow
    with torch.no_grad():
        b_prec = m.write(st_hi, a, b)["beta"]
        ms = MBMemory(16, n_pn=12, n_kc=200, fan_in=3, active_frac=0.05, d_v=8, n_heads=2, gate="surprise"); ms.load_state_dict(m.state_dict(), strict=False)
        b_sur_lo = ms.write(st, a, b)["beta"]; b_sur_hi = ms.write(st_hi, a, b)["beta"]
    assert (b_prec > st1["beta"]).all() and torch.allclose(b_sur_lo, b_sur_hi)
    mc = MBMemory(16, n_pn=12, n_kc=200, fan_in=3, d_v=8, n_heads=2, gate="const")
    with torch.no_grad():
        assert torch.equal(mc.write(mc.init_state(3, a), a, b, gamma=torch.full((3,), 3.0))["beta"], torch.full((3, 2), 0.5))
    with pytest.raises(ValueError):
        MBMemory(16, gate="bogus")


def test_delayed_write_chunk1_equals_sequential_with_gradients():
    def run(delayed):
        torch.manual_seed(0); blk = FlyAIBlock(BlockConfig(**{**TINY_BLOCK, "chunk": 1 if delayed else 8, "delayed_write": delayed})); blk.train()
        torch.manual_seed(1); x = torch.randn(2, 8, 16); y, _ = blk(x); (y ** 2).mean().backward()
        return y.detach(), {n: p.grad.clone() for n, p in blk.named_parameters()}
    y1, g1 = run(True); y2, g2 = run(False)
    assert torch.allclose(y1, y2, atol=1e-6) and all(torch.allclose(g1[k], g2[k], atol=1e-6) for k in g1)


# ---------------------------------------------------------------- block
def test_closed_form_fusion_equation():
    torch.manual_seed(0); blk = FlyAIBlock(BlockConfig(**TINY_BLOCK)); blk.eval()
    with torch.no_grad():
        blk.c_level.copy_(torch.randn(3, 2))
        x, mu1, mu3 = torch.randn(2, 16), torch.randn(2, 16), torch.randn(2, 16)
        pi0, pi1, pi3 = torch.rand(2, 2) * 5, torch.rand(2, 2) * 5, torch.rand(2, 2) * 5
        s = blk._fuse(x, mu1, mu3, pi0, pi1, pi3)
        logw = torch.stack([torch.log(pi0 + 1e-6) + blk.c_level[0], torch.log(pi1 + 1e-6) + blk.c_level[1], torch.log(pi3 + 1e-6) + blk.c_level[2]])
        w = torch.softmax(logw, 0).repeat_interleave(8, -1)                  # (3, B, d)
        ref = w[0] * x + w[1] * mu1 + w[2] * mu3
    assert torch.allclose(s, ref, atol=1e-6)


def test_checkpoint_recompute_uses_the_chunk_gain_under_regulation():
    """With regulation on, the controller moves the gain at every chunk boundary; the recomputed
    forward in backward must use the gain the chunk was computed with, so checkpointed gradients
    equal those of the same chunked loop run without checkpointing."""
    from flyai.block import _flatten, _unflatten
    torch.manual_seed(0); blk = FlyAIBlock(BlockConfig(**{**TINY_BLOCK, "chunk": 4, "regulate_core": True})); blk.train()
    torch.manual_seed(1); x = torch.randn(2, 12, 16)
    y, _ = blk(x); (y ** 2).mean().backward(); g_ckpt = {n: p.grad.clone() for n, p in blk.named_parameters()}
    assert float(blk.core.base_gain) != 1.0                                # the controller did move during the forward
    torch.manual_seed(0); blk2 = FlyAIBlock(BlockConfig(**{**TINY_BLOCK, "chunk": 4, "regulate_core": True})); blk2.train()
    flat, spec = _flatten(blk2.init_state(2, x)); ys = []
    for s0 in range(0, 12, 4):                                              # the same loop without checkpoint
        gain = blk2.core.gain.detach().clone(); out = blk2._run_chunk(x[:, s0:s0 + 4], spec, gain, *flat); ys.append(out[0]); flat = list(out[1:])
        blk2.regulate(_unflatten(flat, spec))
    y2 = torch.cat(ys, 1); (y2 ** 2).mean().backward(); g_plain = {n: p.grad.clone() for n, p in blk2.named_parameters()}
    assert torch.equal(y, y2) and all(torch.allclose(g_ckpt[k], g_plain[k], atol=1e-6) for k in g_ckpt)


def test_block_config_rejects_non_string_modes():
    with pytest.raises(ValueError):
        BlockConfig(**{**TINY_BLOCK, "arousal": True})
    with pytest.raises(ValueError):
        FlyAIConfig(**{**TINY_MODEL, "memory_gate": "nope"})


def test_learned_gate_is_parameter_light():
    """The learned-gate control replaces the n_lv x G fusion offsets by one per-group gate shared across
    groups: n_lv^2 d/G + n_lv parameters (867 vs 24 at d = 768, G = 8: under 0.1 % of a block)."""
    a = FlyAIBlock(BlockConfig(**TINY_BLOCK)); b = FlyAIBlock(BlockConfig(**{**TINY_BLOCK, "fusion": "learned"}))
    na, nb = sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters())
    d, G, n_lv = 16, 2, 3
    assert not hasattr(b, "c_level") and nb - na == (n_lv * n_lv * (d // G) + n_lv) - n_lv * G
    big = FlyAIBlock(BlockConfig(d=768, n_groups=8, n_heads=8, n_kc=128, n_pn=16, fan_in=3, d_v=8, n_rings=2, seed=0))
    assert (3 * 3 * 96 + 3 - 24) < 0.001 * sum(p.numel() for p in big.parameters())


# ---------------------------------------------------------------- categorical model
def test_arousal_bounds_and_sites():
    """gamma stays in [0.25, 4] however far the logit is pushed; 'none' gives exactly 1. Sites: the pi0
    site (modes both / pi0) changes the inferred latent; the gate site (modes both / beta) raises the
    write gate for the SAME latent; each site is inert in the modes that exclude it."""
    def make(mode):
        torch.manual_seed(0); m = FlyAI(FlyAIConfig(**{**TINY_MODEL, "arousal": mode, "K_max": 1}))
        if mode != "none":
            with torch.no_grad(): m.ar_w.fill_(50.0); m.ar_b.fill_(30.0)     # push the logit far outside the bounds
        return m.eval()
    o = torch.randint(0, 24, (2,)); new = {}
    for mode in ("both", "pi0", "beta", "none"):
        m = make(mode); st = m.init_state(2, m.emb.weight); g = m._arousal(st)
        assert (g >= 0.25 - 1e-6).all() and (g <= 4.0 + 1e-6).all()
        assert (g == 1).all() if mode == "none" else float(g.min()) > 3.99
        with torch.no_grad():
            _, new[mode], _ = m.step(st, o)
    s1 = {k: v["s1"] for k, v in new.items()}; beta = {k: v["mem"]["beta"] for k, v in new.items()}
    assert torch.equal(s1["beta"], s1["none"]) and torch.equal(s1["both"], s1["pi0"])       # the gate site never touches inference
    assert not torch.equal(s1["pi0"], s1["none"])                                            # the pi0 site does
    assert (beta["beta"] > beta["none"]).all() and (beta["both"] > beta["pi0"]).all()       # same latent, +log 4 on the gate logit


def test_adaptive_k_and_fixed_k_semantics():
    torch.manual_seed(0); x = torch.randint(0, 24, (2, 6))
    m = FlyAI(FlyAIConfig(**{**TINY_MODEL, "K_max": 3, "adaptive_K": False})); m.eval()
    with torch.no_grad(): m(x)
    assert m.last_diag["K_mean"] == 3.0
    m = FlyAI(FlyAIConfig(**{**TINY_MODEL, "K_max": 3, "adaptive_K": True, "tol": 1e9})); m.eval()
    with torch.no_grad(): m(x)
    assert m.last_diag["K_mean"] == 0.0                                      # a huge tolerance stops before any step is accepted
    m = FlyAI(FlyAIConfig(**{**TINY_MODEL, "K_max": 0})); m.eval()
    with torch.no_grad(): m(x)
    assert m.last_diag["K_mean"] == 0.0


def test_no_level2_tracker_and_precision_ratios_bounded():
    m = FlyAI(FlyAIConfig(**TINY_MODEL)); st = m.init_state(2, m.emb.weight)
    assert "pi2" not in st and set(st) == {"s1", "pi1", "pi3", "Sbar", "t", "mem", "s3"}
    with torch.no_grad(): lm_loss(m.eval(), torch.randint(0, 24, (2, 8)))
    assert m.last_diag["pi1_max"] <= GammaPrecision.PI_MAX


def test_training_mode_under_no_grad_does_not_raise():
    m = FlyAI(FlyAIConfig(**{**TINY_MODEL, "K_max": 2})); m.train()
    with torch.no_grad():
        logits, _ = m(torch.randint(0, 24, (2, 5)))
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------- decoding
def test_efe_decode_greedy_limit():
    torch.manual_seed(0); m = FlyAI(FlyAIConfig(**{**TINY_MODEL, "K_max": 1})); prefix = torch.randint(0, 24, (2, 4))
    out = efe_decode(m, prefix, 3, beta_e=0.0, top_k=8)
    m.eval(); seq = prefix.clone()
    with torch.no_grad():
        for _ in range(3):
            st = m.init_state(2, m.emb.weight)
            for t in range(seq.shape[1]): _, st, _ = m.step(st, seq[:, t])
            mu, *_ = m._priors(st); nxt = m.out(m.out_norm(mu)).argmax(-1, keepdim=True); seq = torch.cat([seq, nxt], 1)
    assert torch.equal(out, seq[:, 4:])
    m2 = FlyAI(FlyAIConfig(**{**TINY_MODEL, "K_max": 1})); m2.train(); efe_decode(m2, prefix, 1, beta_e=1.0, top_k=4)
    assert m2.training                                                       # mode restored


# ---------------------------------------------------------------- tasks
def test_hmm_posterior_predictive_matches_brute_force():
    S, V, Tn = 3, 4, 5
    rng = np.random.default_rng(3); rng2 = np.random.default_rng(3)
    T_ = rng2.dirichlet(np.full(S, 0.3), size=S); E_ = rng2.dirichlet(np.full(V, 0.3), size=S)
    x, post = HMMWindTunnel(n_states=S, vocab=V).sample(1, Tn, rng); x = x[0].numpy(); post = post[0].numpy(); pi = np.full(S, 1 / S)
    def joint(obs):
        tot = 0.0
        for path in itertools.product(range(S), repeat=len(obs)):
            p = pi[path[0]] * E_[path[0], obs[0]]
            for i in range(1, len(obs)): p *= T_[path[i - 1], path[i]] * E_[path[i], obs[i]]
            tot += p
        return tot
    for t in range(Tn):
        prev = list(x[:t]); base = joint(prev) if t > 0 else 1.0
        bf = np.array([joint(prev + [o]) / base for o in range(V)])
        assert np.abs(bf - post[t]).max() < 1e-6
    lg = torch.log(torch.from_numpy(post)[None] + 1e-12)
    assert float(predictive_kl(lg, torch.from_numpy(post)[None]).abs().max()) < 1e-5


def test_task_semantics_rebind_and_retention():
    rng = np.random.default_rng(0); x, y = nonstationary_mqar(8, 96, 8, 48, rng, n_rebind=4)
    nk = 16
    for b in range(8):
        keys, vals = x[b, 0:16:2], x[b, 1:16:2]; binding = {int(k): int(v) for k, v in zip(keys, vals)}
        rb_keys, rb_vals = x[b, 48:56:2], x[b, 49:56:2]; new = dict(binding); new.update({int(k): int(v) for k, v in zip(rb_keys, rb_vals)})
        for t in range(96):
            if y[b, t] != -100:
                assert x[b, t] < nk and (int(y[b, t]) == (binding if t < 48 else new)[int(x[b, t])])
    x, y = retention_batch(4, 64, 48, rng, n_items=4)
    for b in range(4):
        binding = {int(k): int(v) for k, v in zip(x[b, 0:8:2], x[b, 1:8:2])}
        for t in range(60, 64): assert int(y[b, t]) == binding[int(x[b, t])]


# ---------------------------------------------------------------- harness
def test_analyse_pairs_by_seed_id_and_flags_missing_controls():
    res = [dict(condition="h", lr=1e-3, seed=s, m=v) for s, v in ((0, 0.9), (1, 0.8), (2, 0.85))]
    res += [dict(condition="c", lr=1e-3, seed=s, m=v) for s, v in ((2, 0.5), (0, 0.7), (1, 0.6))]     # different order
    an = analyse(res, "m", "h", ["c", "ghost"])
    assert an["comparisons"]["c"]["seeds"] == [0, 1, 2] and abs(an["comparisons"]["c"]["effect"] - 0.25) < 1e-9
    assert an["missing_controls"] == ["ghost"]
    c = compare([0.9], [0.5]); assert not c["ci_excludes_0"] and np.isnan(c["effect_in_sd"])
