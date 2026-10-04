"""EXPLORATORY language-model prototype (not pre-registered, one seed; nothing here enters a verdict). It de-risks rung 2:
does the hybrid model train on real text, what does the sparse store cost at language-model size, and does anything
differ between dense and Kenyon-cell keys in perplexity or on the long-range recall probes?

Model: flyprinc.hybrid.HybridLM, 8 layers (6 Gated DeltaNet-gated memory layers + 2 sliding-window attention layers of
`window` tokens), d_model 384, 6 heads of 64, GPT-2 vocabulary, tied embeddings. Arms: `dense` (64-dimensional keys) and
`sparse` (the Kenyon-cell address: 1,024 cells, 64 on, fan-in 14, the same 16x expansion and 1/16 coding level as rung 1b).
Data: WikiText-103 (scripts/fetch_wikitext103.py). Evaluation: validation perplexity over sliding windows and the D1-regime
probes (experiments/p3_s2_lm/probes.py). TF32 matrix multiplication is on (a speed choice of this prototype only).
Usage: python experiments/p3_s2_lm/prototype.py --arm dense --steps 3000 [--dry]"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm  # noqa: E402

ARMS = {"dense": {}, "sparse": dict(address="sparse", n_cells=1024, k_active=64, fan_in=14)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="dense", choices=list(ARMS)); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--B", type=int, default=2, help="sequences per micro-batch"); ap.add_argument("--accum", type=int, default=4, help="micro-batches per update")
    ap.add_argument("--T", type=int, default=2048); ap.add_argument("--window", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--checkpoint", action="store_true")
    ap.add_argument("--compile", action="store_true"); ap.add_argument("--dry", action="store_true", help="3 steps: step time and peak memory only")
    ap.add_argument("--eval_windows", type=int, default=None); ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "prototype_runs"))
    ap.add_argument("--save_every", type=int, default=100, help="updates between checkpoints (an interrupted run resumes from the last one)")
    ap.add_argument("--stop_after", type=int, default=None, help="exit after this many updates without finishing (simulates an interruption; for testing resume)")
    ap.add_argument("--corpus", default="wikitext103", choices=["wikitext103", "tinystories"])
    a = ap.parse_args(); dev = "cuda"; torch.manual_seed(a.seed); torch.set_float32_matmul_precision("high"); lm.set_corpus(a.corpus)
    tag = ("" if a.corpus == "wikitext103" else a.corpus + "_") + f"{a.arm}_s{a.seed}"
    out = Path(a.out) / (tag + ("_dry" if a.dry else "")); out.mkdir(parents=True, exist_ok=True)   # a dry run never writes into a real run's folder
    if (out / "metrics.json").exists() and not a.dry: raise SystemExit(f"{out} already holds a finished run")
    layer = dict(d_k=64, d_v=64, d_c=16, states=(), x_only_gate=True, tie_gamma=True, chunk=64, compile_chunk=a.compile, **ARMS[a.arm])
    (out / "config.json").write_text(json.dumps({"corpus": a.corpus, "arm": a.arm, "seed": a.seed, "T": a.T, "window": a.window, "layer": {**layer, "compile_chunk": False},
                                                 "d_model": 384, "n_layers": 8, "n_heads": 6, "attn_every": 4}, indent=1))
    model = lm.build_model(384, 8, 6, a.window, 4, layer, tied=True, emb_std=0.02, checkpoint=a.checkpoint).to(dev)
    n_params = sum(p.numel() for p in model.parameters()); n_emb = model.emb.weight.numel()
    print(f"{a.arm}: {n_params / 1e6:.1f}M parameters ({(n_params - n_emb) / 1e6:.1f}M outside the embedding)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, a.steps)))))
    ids = lm.tokens("train"); rng = np.random.default_rng(a.seed + 1); steps = 3 if a.dry else a.steps
    times = []; losses = []; start = 0; resumed = []; ck = out / "ckpt.pt"
    if ck.exists() and not a.dry:                                             # resume: weights, optimiser, schedule and the data generator's state
        st = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); rng.bit_generator.state = st["rng"]
        start, times, losses, resumed = st["step"], st["times"], st["losses"], st["resumed"] + [st["step"]]
        print(f"{a.arm}: resumed from update {start}", flush=True)
    log = open(out / "log.jsonl", "a" if start else "w"); t0 = time.time(); torch.cuda.reset_peak_memory_stats()
    tok_per_step = a.B * a.accum * a.T
    for s in range(start, steps):
        torch.cuda.synchronize(); ts = time.time(); opt.zero_grad(set_to_none=True); acc = 0.0
        for _ in range(a.accum):                                              # gradient accumulation: B x accum sequences of T tokens per update
            x, y = lm.train_batch(ids, a.B, a.T, rng); x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten()) / a.accum; loss.backward(); acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        torch.cuda.synchronize(); times.append(time.time() - ts); losses.append(acc)
        if (s + 1) % 50 == 0 or s == 0 or a.dry:
            rec = {"step": s + 1, "loss": float(np.mean(losses[-50:])), "step_s": times[-1], "tokens_per_s": tok_per_step / times[-1], "elapsed_min": (time.time() - t0) / 60,
                   "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30}
            log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        stop = (out / "STOP").exists() or (out.parent / "STOP").exists()         # graceful stop: checkpoint and exit (a hidden job has no window to close)
        if ((s + 1) % a.save_every == 0 or stop) and not a.dry and s + 1 < steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.bit_generator.state, "step": s + 1,
                        "times": times, "losses": losses, "resumed": resumed}, out / "ckpt.tmp"); os.replace(out / "ckpt.tmp", ck)   # atomic: never a half-written checkpoint
        if a.stop_after is not None and s + 1 >= a.stop_after: log.close(); raise SystemExit(f"stopped after update {s + 1} (--stop_after)")
        if stop and not a.dry: log.close(); raise SystemExit(f"STOP file found: checkpointed at update {s + 1} and stopped")
    peak = torch.cuda.max_memory_reserved() / 2**30
    res = {"corpus": a.corpus, "arm": a.arm, "seed": a.seed, "params": n_params, "params_non_embedding": n_params - n_emb, "steps": steps, "B": a.B, "accum": a.accum, "T": a.T,
           "window": a.window, "tokens_trained": steps * tok_per_step, "step_s_median": float(np.median(times[1:])) if len(times) > 1 else float("nan"),
           "tokens_per_s": tok_per_step / float(np.median(times[1:])) if len(times) > 1 else float("nan"), "train_loss_last": float(np.mean(losses[-50:])),
           "gpu_peak_reserved_gb": peak, "resumed_at_updates": resumed, "exploratory": True, "pre_registered": False}
    print(f"{a.arm}: median step {res['step_s_median']:.3f} s = {res['tokens_per_s']:.0f} tokens/s, peak {peak:.1f} GB", flush=True)
    if not a.dry:
        torch.save(model.state_dict(), out / "model.pt")                      # final weights (not committed), for scripts that sample from the model
        probes = lm.load_probes("validation") if (lm.DATA / "probes_validation.npz").exists() else None   # the recall probes exist for WikiText-103 only
        te = time.time(); res.update(lm.evaluate(model, "validation", a.T, max(1, 4096 // a.T), dev, probes=probes, max_windows=a.eval_windows))
        res["eval_min"] = (time.time() - te) / 60; res["wall_min"] = (time.time() - t0) / 60
        print(json.dumps({k: v for k, v in res.items() if k.startswith(("val_", "probe_", "eval_", "wall_"))}), flush=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1)); log.close()
    if ck.exists(): ck.unlink()                                               # the finished metrics are the record
