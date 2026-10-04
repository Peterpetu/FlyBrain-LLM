"""Rung-1b pilot on the grid's own arms and code path (exploratory; seed 0, not a grid seed): the Gated DeltaNet-gated dense
arm at P = 64 and 128 first (the P values are chosen from the dense arm alone, PREREG.md), then the sparse arm at the same
P (learnability and step time). Each run is `run.train_fn` itself, so configuration, training and evaluation are the grid's."""
import json, os, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run

if __name__ == "__main__":                                                    # the data workers re-import this file (spawn): the work runs only in the parent
    run.RUN.update(device="cuda", compile=True, data_workers=int(os.environ.get("DATA_WORKERS", "5")))
    out = Path(os.environ.get("SCRATCH", tempfile.gettempdir())) / "p3_s1b_pilot_gdn"
    jobs = [("dense", 64, ""), ("dense", 128, ""), ("sparse", 64, ""), ("sparse", 128, "")]
    if len(sys.argv) > 1: jobs = [(j.split(":") + [""])[:3] for j in sys.argv[1:]]           # arm:P or arm:P:key=value,key=value (overrides of FULL)
    for arm, P, over in jobs:
        if (out / "STOP").exists(): print("STOP file found: stopping", flush=True); break      # graceful stop between jobs (never kill a GPU process)
        P = int(P); ov = {k: int(v) for k, v in (kv.split("=") for kv in over.split(",") if kv)}
        S = dict(run.FULL, **ov, keys=P); rd = out / (f"{arm}_P{P}" + "".join(f"_{k}{v}" for k, v in ov.items())); rd.mkdir(parents=True, exist_ok=True)
        m = run.train_fn(dict(run.arms(S)[arm], _S=S, lr=1e-3, seed=0, condition=f"{arm}|P{P}"), 0, rd)
        curve = [json.loads(l) for l in open(rd / "log.jsonl")]; marks = " ".join(f"{c['step']}:{c['train_acc']:.2f}" for c in curve if c["step"] % 1000 == 0)
        loads = " ".join(f"{k[9:]}:{m[k]:.2f}" for k in m if k.startswith("acc_load_") and m[k] == m[k])
        print(f"PILOT {arm:6s} P={P:3d} T={S['tokens_per_key'] * P:4d} {over} | train {marks} | held-out {m['acc']:.3f} (by load {loads}) | "
              f"step {m['step_s_median']:.3f} s, wall {m['wall_s'] / 60:.0f} min, chunk error {m.get('chunk_error_max', float('nan')):.1e}, "
              f"peak {m.get('gpu_peak_reserved_gb', 0):.1f} GB", flush=True)
