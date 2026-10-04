import numpy as np
import pytest

from flybrain.connectome import load_connectome
from flybrain.connectome.io import graph_summary
from flybrain.connectome.controls import degree_preserving_shuffle, erdos_renyi, fit_dcsbm, sample_dcsbm, graph_stats


@pytest.fixture(scope="module")
def c():
    try:
        return load_connectome()
    except FileNotFoundError:
        pytest.skip("raw connectome not fetched (scripts/fetch_data.sh)")


def test_headline_numbers(c):
    s = graph_summary(c)
    assert s["n_neurons"] == 166700 and s["n_edges"] == 10520431 and s["n_contacts"] == 101422396
    assert abs(s["frac_inhibitory"] - 0.308) < 0.002 and s["outdeg_max"] == 7749


def test_shuffle_preserves_degrees(c):
    from flybrain.modules.masked_rnn import select_core
    idx = select_core(c, 256); W = c.subgraph(idx)
    S = degree_preserving_shuffle(W, np.random.default_rng(0), attempts_per_edge=10)
    assert S.nnz == W.nnz
    assert np.array_equal(np.diff(S.indptr), np.diff(W.indptr))
    assert np.array_equal(np.bincount(S.indices, minlength=256), np.bincount(W.indices, minlength=256))
    assert S.accepted_swaps > 0.2 * W.nnz
    assert (S != W).nnz > 0


def test_er_matches_n_e(c):
    R = erdos_renyi(300, 5000, np.random.default_rng(1))
    assert R.nnz == 5000 and R.diagonal().sum() == 0


def test_dcsbm_roundtrip(c):
    from flybrain.modules.masked_rnn import select_core
    idx = select_core(c, 1024); W = c.subgraph(idx)
    fit = fit_dcsbm(W, c.cell_type[idx], c.sign[idx])
    Ws, blocks, sign = sample_dcsbm(fit, 1024, np.random.default_rng(0))
    st, so = graph_stats(W), graph_stats(Ws)
    assert abs(so["outdeg_mean"] - st["outdeg_mean"]) / st["outdeg_mean"] < 0.15
    assert abs((sign < 0).mean() - (c.sign[idx] < 0).mean()) < 0.05
