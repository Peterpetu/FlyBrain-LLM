"""Throughput of K training processes sharing the GPU (the step is launch-bound, so one process leaves the GPU idle
part of the time). Each worker trains the rung-1 v3 state arm (compiled, batch 64, T = 256, blocks of 8) and reports
its median step time after compilation; the launcher runs K = 1, 2, 3 and prints the aggregate steps per second."""
import sys, time, subprocess, argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def worker(steps, barrier, K, idx):
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT))
    from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig
    from flyprinc.lmtasks import LMTaskConfig, make_batch
    dev = "cuda"; task = LMTaskConfig(T=256, keys_per_seq=24, window=256, p_rebind=0.2, p_outlier=0.15, far_fraction=0.0); torch.manual_seed(0)
    lc = LayerConfig(d_model=128, n_heads=4, d_k=32, d_v=32, d_c=8, chunk=8, sense_block=8, compile_chunk=True, n_traces=2, states=("nconf", "zc"))
    model = HybridLM(HybridConfig(vocab=task.vocab, d_model=128, n_layers=2, n_heads=4, window=256, layer=lc, attn_every=99, pos="rope", tied=False, emb_std=1.0)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3); rng = np.random.default_rng(0); times = []
    for s in range(10 + steps):
        if s == 10:                                                       # barrier: every worker has compiled and warmed up before any timing
            (Path(barrier) / f"ready{idx}").touch()
            while len(list(Path(barrier).glob("ready*"))) < K: time.sleep(0.05)
        toks, tgt, _, _ = make_batch(task, 64, rng); x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
        torch.cuda.synchronize(); t = time.time(); logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
        if s >= 10: times.append(time.time() - t)
    print(f"MEDIAN {np.median(times):.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--worker", action="store_true"); ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--K", type=int, nargs="*", default=[1, 2, 3]); ap.add_argument("--barrier", default=None); ap.add_argument("--idx", type=int, default=0)
    args = ap.parse_args()
    if args.worker: worker(args.steps, args.barrier, args.K[0], args.idx); sys.exit()
    import tempfile
    base = None
    for K in args.K:
        bar = tempfile.mkdtemp()
        procs = [subprocess.Popen([sys.executable, __file__, "--worker", "--steps", str(args.steps), "--K", str(K), "--barrier", bar, "--idx", str(i)],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for i in range(K)]
        med = [float(re.search(r"MEDIAN ([\d.]+)", p.communicate()[0]).group(1)) for p in procs]
        agg = sum(1 / m for m in med); base = base or agg
        print(f"K = {K}: step times {', '.join(f'{m:.3f}' for m in med)} s -> aggregate {agg:.2f} steps/s ({agg / base:.2f}x one process)", flush=True)
