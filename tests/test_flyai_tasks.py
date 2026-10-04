import numpy as np
import pytest, torch
from flyai.tasks import nonstationary_mqar, retention_batch, TopicSwitchStream, same_index_logits
from flyai.model import FlyAI, FlyAIConfig


def test_nonstationary_mqar_rebinds():
    rng = np.random.default_rng(0); x, y = nonstationary_mqar(4, 96, 8, 48, rng, n_rebind=3)
    assert x.shape == (4, 96) and (y != -100).sum() > 0 and y.max() < 32


def test_retention_batch():
    rng = np.random.default_rng(0); x, y = retention_batch(3, 64, 48, rng, n_items=4)
    assert (y[:, -4:] != -100).all() and (y[:, :-4] == -100).all()


def test_topic_switch_stream():
    try:
        s = TopicSwitchStream()
    except FileNotFoundError:
        pytest.skip("needs data/text/tinyshakespeare.txt (scripts/fetch_data.sh)")
    rng = np.random.default_rng(0); x, y, sw = s.batch("train", 2, 128, rng)
    assert x.shape == (2, 128) and sw.any() and x.max() < s.vocab


def test_same_index_logits_can_see_the_query_token():
    """FlyAI's raw logits[:, t] are emitted before x[:, t] is observed; the aligned logits must depend on it."""
    torch.manual_seed(0); torch.set_num_threads(1)
    m = FlyAI(FlyAIConfig(vocab=24, d=16, n=32, K_max=1, n_kc=64, n_pn=8, fan_in=3, d_v=4, n_heads=2, n_rings=2, n_groups=2, regulate_core=False)).eval()
    x = torch.randint(0, 24, (2, 7)); x2 = x.clone(); x2[:, 4] = (x[:, 4] + 1) % 24
    with torch.no_grad():
        raw, _ = m(x); raw2, _ = m(x2); al, _ = same_index_logits(m, x, 23); al2, _ = same_index_logits(m, x2, 23)
    assert torch.equal(raw[:, :5], raw2[:, :5])                      # raw prediction at t=4 cannot see x[:, 4]
    assert not torch.equal(al[:, 4], al2[:, 4])                      # aligned prediction at t=4 does
    assert torch.equal(al[:, :4], al2[:, :4]) and al.shape == raw.shape
