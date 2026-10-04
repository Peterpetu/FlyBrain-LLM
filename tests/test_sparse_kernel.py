"""experiments/p3_k1_sparse_kernel/sparse_chunk.py: the k-sparse chunked step reproduces the layer's own chunked and
sequential outputs for the Gated DeltaNet-gated single-store layer with the Kenyon-cell address (random and learned
expansions, product keys), on several chunk sizes, with a partial final chunk; and refuses configurations it does not cover."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "p3_k1_sparse_kernel"))
from sparse_chunk import sparse_forward  # noqa: E402
from flyprinc.hybrid import LayerConfig, StateMemoryLayer  # noqa: E402

torch.set_num_threads(1)
SMALL = dict(d_model=32, n_heads=2, d_k=8, d_v=8, d_c=4, states=(), x_only_gate=True, tie_gamma=True, n_cells=96, k_active=6, fan_in=3, pk_n1=8)


@pytest.mark.parametrize("address", ["sparse", "sparse_learned", "product"])
@pytest.mark.parametrize("chunk", [8, 32])
def test_sparse_step_matches_the_layer(address, chunk):
    torch.manual_seed(3); x = torch.randn(2, 70, 32)                                           # 70 tokens: a partial final chunk
    layer = StateMemoryLayer(LayerConfig(**SMALL, address=address, chunk=chunk)); seq = StateMemoryLayer(LayerConfig(**SMALL, address=address, chunk=0))
    seq.load_state_dict(layer.state_dict())
    with torch.no_grad():
        yd, std = layer(x); ys, sts = seq(x); yk, Sk = sparse_forward(layer, x)
    assert ((yk - yd).norm() / yd.norm()).item() < 1e-4 and ((yk - ys).norm() / ys.norm()).item() < 1e-4
    assert ((Sk - std["S"]).norm() / std["S"].norm()).item() < 1e-4


def test_refuses_unsupported_layers():
    torch.manual_seed(0); x = torch.randn(1, 16, 32)
    for bad in (dict(address="dense"), dict(x_only_gate=False), dict(n_traces=2)):
        with pytest.raises(AssertionError): sparse_forward(StateMemoryLayer(LayerConfig(**{**SMALL, "address": "sparse", "chunk": 8, **bad})), x)
