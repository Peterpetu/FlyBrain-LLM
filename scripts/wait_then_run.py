"""Wait until a file exists (a grid's summary.json: the grid has finished), then run another job, so that a GPU does not
sit idle between two grids when nobody is at the keyboard. Optionally removes a STOP file that was pausing the second grid.
A file named STOP_CHAIN in logs/ cancels the wait. Start it detached (scripts/launch_detached.ps1).
Usage: python scripts/wait_then_run.py --wait experiments/p3_s1b_sparse/summary.json --remove experiments/p3_s1_hybrid/STOP
           -- experiments/p3_s1_hybrid/run.py --compile --device cuda --data_workers 2"""
import argparse, subprocess, sys, time
from pathlib import Path

R = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--wait", required=True); ap.add_argument("--remove", default=None); ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER); a = ap.parse_args(); cmd = [c for c in a.cmd if c != "--"]
    target = R / a.wait; print(f"waiting for {target}", flush=True)
    while not target.exists():
        if (R / "logs" / "STOP_CHAIN").exists(): raise SystemExit("STOP_CHAIN found: not starting the next job")
        time.sleep(a.poll)
    time.sleep(a.poll)                                                        # let the finished grid's process exit and release the GPU
    if a.remove and (R / a.remove).exists(): (R / a.remove).unlink(); print(f"removed {a.remove}", flush=True)
    print(f"{time.strftime('%Y-%m-%d %H:%M')}: starting {' '.join(cmd)}", flush=True)
    sys.exit(subprocess.call([sys.executable, *cmd], cwd=R))
