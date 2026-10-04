"""Wall-clock test of the efficiency claim (block B): time per token of one memory layer's forward pass, chunked, on the GPU,
for dense keys of the sparse arm's width, the sparse address as implemented (every row updated), the sparse address with the
exact k-sparse step of sparse_chunk.py, and dense keys of equal memory (rows = cells). Two sizes: rung 1b's and a language
model's. Eager PyTorch for all four, plus torch.compile for the two that the grids compile. Medians of repeated interleaved
measurements (CUDA events); the GPU may be shared, which is stated with the result.
Usage: python experiments/p3_k1_sparse_kernel/bench.py [--repeats 15]"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sparse_chunk import sparse_forward, sparse_chunk  # noqa: E402
from flyprinc.hybrid import LayerConfig, StateMemoryLayer, layer_cost  # noqa: E402

SIZES = {"rung 1b (d_k 16, 256 cells, 16 on; B 64, H 4, T 512, d_v 32)": dict(B=64, H=4, T=512, d_k=16, dv=32, cells=256, k=16, fan_in=4, d_model=128),
         "language model (d_k 64, 1,024 cells, 64 on; B 8, H 6, T 2048, d_v 64)": dict(B=8, H=6, T=2048, d_k=64, dv=64, cells=1024, k=64, fan_in=14, d_model=384)}


def layer_for(s, address, d_k, compile_chunk=False):
    c = LayerConfig(d_model=s["d_model"], n_heads=s["H"], d_k=d_k, d_v=s["dv"], d_c=8, states=(), x_only_gate=True, tie_gamma=True, chunk=64,
                    compile_chunk=compile_chunk, address=address, n_cells=s["cells"], k_active=s["k"], fan_in=s["fan_in"])
    return StateMemoryLayer(c).cuda().eval()


@torch.no_grad()
def timed(fn, x, repeats, warm=3):
    for _ in range(warm): fn(x)
    torch.cuda.synchronize(); out = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record(); fn(x); b.record(); torch.cuda.synchronize(); out.append(a.elapsed_time(b))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=15); ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "bench.json")); a = ap.parse_args()
    torch.manual_seed(0); results = {"gpu": torch.cuda.get_device_name(0), "note": "eager unless marked compiled; the GPU may be shared with other jobs", "sizes": {}}
    for name, s in SIZES.items():
        x = torch.randn(s["B"], s["T"], s["d_model"], device="cuda")
        arms = {"dense (equal width)": (layer_for(s, "dense", s["d_k"]), None), "sparse, as implemented": (layer_for(s, "sparse", s["d_k"]), None),
                "sparse, k-sparse step": (layer_for(s, "sparse", s["d_k"]), "kernel"), "dense (equal memory)": (layer_for(s, "dense", s["cells"]), None),
                "dense (equal width), compiled": (layer_for(s, "dense", s["d_k"], True), None), "sparse, as implemented, compiled": (layer_for(s, "sparse", s["d_k"], True), None),
                "sparse, k-sparse step, compiled": (layer_for(s, "sparse", s["d_k"]), "kernel_compiled"), "dense (equal memory), compiled": (layer_for(s, "dense", s["cells"], True), None)}
        import sparse_chunk as sc; step_c = torch.compile(sparse_chunk, dynamic=False)
        def kernel_compiled(x, l):
            saved = sc.sparse_chunk; sc.sparse_chunk = step_c
            try: return sparse_forward(l, x)
            finally: sc.sparse_chunk = saved
        fns = {k: ((lambda x, l=l: sparse_forward(l, x)) if mode == "kernel" else (lambda x, l=l: kernel_compiled(x, l)) if mode == "kernel_compiled" else (lambda x, l=l: l(x)))
               for k, (l, mode) in arms.items()}
        times = {k: [] for k in arms}
        for _ in range(a.repeats):                                                      # interleaved: every arm sees the same share of the GPU
            for k, fn in fns.items(): times[k] += timed(fn, x, 1, warm=1 if not times[k] else 0)
        tok = s["B"] * s["T"]; row = {}
        for k, (l, mode) in arms.items():
            cost = layer_cost(l.c); med = float(np.median(times[k]))
            row[k] = {"ms_per_forward": med, "us_per_token": 1000 * med / tok, "macs_per_head_token_as_implemented": cost["macs_per_head_token"],
                      "macs_per_head_token_sparse_kernel": cost["macs_active_per_head_token"], "rows": l.c.rows, "active_rows": l.c.active_rows}
        results["sizes"][name] = row
        print(f"\n{name}:"); base = row["dense (equal memory)"]["ms_per_forward"]
        for k, r in row.items(): print(f"  {k:34s} {r['ms_per_forward']:8.1f} ms  ({r['us_per_token']:.2f} us/token; {base / r['ms_per_forward']:.2f}x faster than equal-memory dense)", flush=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
