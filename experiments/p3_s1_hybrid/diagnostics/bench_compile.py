"""Full-size step time of the two-store state arm on the GPU: eager vs a compiled per-chunk step, and the
batch-size scaling (a launch-bound step costs about the same at B = 16 and B = 64). Also checks that the
compiled step reproduces the eager outputs and gradients."""
import sys, time, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from flyprinc.hybrid import HybridLM, HybridConfig, LayerConfig
from flyprinc.lmtasks import LMTaskConfig, make_batch

dev = "cuda"
task = LMTaskConfig()
MODES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["eager", "compile"]
BATCHES = [int(b) for b in sys.argv[2].split(",")] if len(sys.argv) > 2 else [16]
ARM = sys.argv[3] if len(sys.argv) > 3 else "state"
SENSE = int(sys.argv[4]) if len(sys.argv) > 4 else 0          # block-delayed sensing block (0 = per-token sensing with refinement)
CHUNK = int(sys.argv[5]) if len(sys.argv) > 5 else 16
states = ("nconf", "zc") if ARM == "state" else ()
print("arm", ARM, "sense_block", SENSE, "chunk", CHUNK, flush=True)


def build(compile_chunk, seed=0):
    torch.manual_seed(seed)
    lc = LayerConfig(d_model=256, n_heads=4, d_k=64, d_v=64, d_c=16, chunk=CHUNK, refine=6, n_traces=2, states=states, compile_chunk=compile_chunk, sense_block=SENSE)
    return HybridLM(HybridConfig(vocab=task.vocab, d_model=256, n_layers=4, n_heads=4, window=512, layer=lc)).to(dev)


def step(model, opt, x, y):
    logits = model(x); m = y >= 0
    loss = F.cross_entropy(logits[m], y[m]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return loss


res = {}
for B in BATCHES:
    rng = np.random.default_rng(0); toks, tgt, _, _ = make_batch(task, B, rng)
    x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
    for mode in MODES:
        model = build(mode == "compile"); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        step(model, opt, x, y); torch.cuda.synchronize(); first = time.time() - t0          # includes compilation for "compile"
        times = []
        for _ in range(3):
            torch.cuda.synchronize(); t = time.time(); step(model, opt, x, y); torch.cuda.synchronize(); times.append(time.time() - t)
        res[f"{mode}|B{B}"] = {"first_step_s": round(first, 1), "step_s": round(float(np.median(times)), 3), "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        print(mode, "B", B, res[f"{mode}|B{B}"], flush=True)
        del model, opt; torch.cuda.empty_cache()

if "compile" in MODES and "eager" in MODES:                                                   # the compiled step must reproduce the eager arithmetic
    rng = np.random.default_rng(1); toks, tgt, _, _ = make_batch(task, 2, rng)
    x, y = torch.as_tensor(toks, device=dev), torch.as_tensor(tgt, device=dev)
    outs = []
    for cc in (False, True):
        model = build(cc, seed=3); logits = model(x); m = y >= 0; loss = F.cross_entropy(logits[m], y[m]); loss.backward()
        outs.append((logits.detach(), [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]))
    dl = ((outs[0][0] - outs[1][0]).norm() / outs[0][0].norm()).item()
    dg = max(((a - b).norm() / (a.norm() + 1e-12)).item() for a, b in zip(outs[0][1], outs[1][1]) if a is not None)
    res["compile_vs_eager"] = {"logits_rel": dl, "grads_rel_max": dg}; print("compile vs eager: logits rel %.1e, grads rel max %.1e" % (dl, dg), flush=True)
print(json.dumps(res))
