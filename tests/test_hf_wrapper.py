"""benchmarks/hf_flyai.py: the HF-transformers wrapper reproduces HybridFlyAILM, round-trips through
save_pretrained / from_pretrained, and decodes incrementally without changing the logits."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from flyai.lm import make_lm
from benchmarks.hf_flyai import FlyAIForCausalLM, FlyAIHFConfig, FlyAIState, export_checkpoint

torch.set_num_threads(1)
B, T, V = 2, 6, 100


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    lm = make_lm("tiny", "flyai", vocab=V, K_max=1)
    hf = FlyAIForCausalLM.from_flyai(lm).eval()
    return lm, hf, torch.randint(0, V, (B, T))


def test_forward_and_loss_match_flyai(tiny):
    lm, hf, x = tiny
    assert hf.model is lm and [l.kind for l in lm.layers] == ["flyai", "flyai", "flyai", "attention"]
    with torch.no_grad():
        out = hf(input_ids=x, labels=x)
        ref_logits, _ = lm(x); ref_loss = lm.loss(x)
    assert out.logits.shape == (B, T, V) and torch.equal(out.logits, ref_logits)
    assert torch.isfinite(out.loss) and torch.allclose(out.loss, ref_loss, atol=1e-5)


def test_save_pretrained_from_pretrained_round_trip(tiny, tmp_path):
    lm, hf, x = tiny
    export_checkpoint(lm, lm.cfg, None, tmp_path)                    # no tokenizer: offline
    assert (tmp_path / "config.json").exists() and (tmp_path / "model.safetensors").exists()
    cfg = transformers.AutoConfig.from_pretrained(tmp_path)
    assert isinstance(cfg, FlyAIHFConfig) and cfg.to_lm_config() == lm.cfg
    hf2 = transformers.AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert isinstance(hf2, FlyAIForCausalLM)
    sd, sd2 = hf.state_dict(), hf2.state_dict()
    assert sd.keys() == sd2.keys()
    for k in sd:                                                     # parameters and buffers (core.gain/lam/delta, E, masks)
        assert torch.equal(sd[k], sd2[k]), k
    assert hf2.model.head.weight is hf2.model.emb.weight            # tie survives the reload
    with torch.no_grad():
        assert torch.equal(hf(input_ids=x).logits, hf2(input_ids=x).logits)


def test_state_carry_over_matches_full_forward(tiny):
    _, hf, x = tiny
    with torch.no_grad():
        full = hf(input_ids=x).logits
        a = hf(input_ids=x[:, :4], use_cache=True)
        assert isinstance(a.past_key_values, FlyAIState) and a.past_key_values.seq_len == 4
        b = hf(input_ids=x[:, 4:], past_key_values=a.past_key_values, use_cache=True)
    assert b.past_key_values.seq_len == T
    assert torch.allclose(torch.cat([a.logits, b.logits], 1), full, atol=1e-5)


def test_generate_incremental_is_causal(tiny):
    _, hf, x = tiny
    with torch.no_grad():
        gen = hf.generate(x, max_new_tokens=3, do_sample=False, return_dict_in_generate=True, output_logits=True)
        seq = gen.sequences
        assert seq.shape == (B, T + 3) and torch.equal(seq[:, :T], x)
        full = hf(input_ids=seq).logits                              # plain forward over the generated sequence
        for j, step_logits in enumerate(gen.logits):                 # logits that picked token T+j came from the state
            assert torch.allclose(step_logits, full[:, T - 1 + j], atol=1e-5)
        assert torch.equal(seq[:, T:], full[:, T - 1:-1].argmax(-1))
        no_cache = hf.generate(x, max_new_tokens=3, do_sample=False, use_cache=False)
    assert torch.equal(no_cache, seq)
