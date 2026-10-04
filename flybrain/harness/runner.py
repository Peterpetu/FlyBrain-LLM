"""Run grids of (condition x learning-rate x seed) with full provenance.

Each run writes to  <exp_dir>/runs/<condition>/lr<lr>/seed<seed>/  the files
config.json, env.json (git commit, torch, cpu), metrics.json and log.jsonl. Finished runs
are skipped, so grids are resumable. `train_fn(config, seed, run_dir) -> metrics dict`.

A file named STOP in the experiment directory stops the grid before its next run (the graceful way
to stop GPU work: on Windows, killing a process that is using the GPU can make the driver reset
channels and crash every other process on the GPU). Delete it to resume.

Parallel workers on one grid (`claim=True`): each worker atomically claims a run folder before
running it and skips runs another worker holds, so several processes can share a GPU without
running anything twice. A final pass with `claim=False`, after every worker has exited, re-runs
any run a crashed worker left unfinished and returns the complete result list.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

import torch

from .prereg import Prereg


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).resolve().parents[2]).strip()
    except Exception:
        return "unknown"


def run_dir_for(exp_dir: Path, condition: str, lr: float, seed: int, variant: str | None = None) -> Path:
    root = "runs" if variant is None else f"runs_{variant}"
    folder = "".join("_" if ch in '<>:"/\\|?*' else ch for ch in condition)      # a condition name may contain characters Windows forbids in folder names (e.g. "state|P48|r0.15")
    return Path(exp_dir) / root / folder / f"lr{lr:g}" / f"seed{seed}"


def _canon(cfg: dict) -> dict:
    return json.loads(json.dumps(cfg, sort_keys=True, default=str))


def _claim(rd: Path) -> bool:
    """Atomically claim a run folder for this process; False if another worker already holds it."""
    rd.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(rd / "claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"pid {os.getpid()} at {time.time():.0f}\n".encode()); os.close(fd)
    return True


def run_grid(exp_dir: Path, conditions: dict, lrs: list[float], seeds: list[int], train_fn: Callable,
             per_condition_lrs: dict | None = None, quiet: bool = False, variant: str | None = None, claim: bool = False) -> list[dict]:
    """`variant` (e.g. "replication") keeps a second grid of the same pre-registered experiment
    in `runs_<variant>/` with its own summary, so a replication never overwrites the discovery.
    `claim` makes this process one of several parallel workers (module docstring)."""
    exp_dir = Path(exp_dir); prereg = Prereg(exp_dir).freeze_or_check()
    results = []
    for name, cfg in conditions.items():
        for lr in (per_condition_lrs or {}).get(name, lrs):
            for seed in seeds:
                rd = run_dir_for(exp_dir, name, lr, seed, variant)
                mfile = rd / "metrics.json"
                full = {**cfg, "lr": lr, "seed": seed, "condition": name}
                if mfile.exists():
                    stored = json.loads((rd / "config.json").read_text()) if (rd / "config.json").exists() else None
                    if stored is not None and _canon(stored) != _canon(full):
                        raise RuntimeError(f"{rd}: a finished run with a DIFFERENT config exists (e.g. a smoke test or a renamed "
                                           f"condition). Delete it or use another --variant; it is never silently reused.")
                    m = json.loads(mfile.read_text()); results.append(m); continue
                if (exp_dir / "STOP").exists():                          # graceful stop between runs (killing a GPU process can reset the driver for every process on it)
                    if not quiet: print(f"STOP file found in {exp_dir}: stopping before {rd}", flush=True)
                    return results
                if claim:
                    if not _claim(rd):
                        continue                                          # another worker runs it; the final pass collects it
                    if mfile.exists():                                    # finished by another worker between the check above and this claim
                        (rd / "claim").unlink(missing_ok=True); results.append(json.loads(mfile.read_text())); continue
                rd.mkdir(parents=True, exist_ok=True)
                (rd / "config.json").write_text(json.dumps(full, indent=1, default=str))
                commit = git_commit()                                    # the code that runs THIS run (recorded once, at start)
                (rd / "env.json").write_text(json.dumps({"commit": commit, "prereg_sha256": prereg, "torch": torch.__version__,
                                                          "python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
                                                          "threads": torch.get_num_threads(), "device": (torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "cpu"), "started": time.time()}, indent=1))
                t0 = time.time()
                torch.manual_seed(seed)
                m = train_fn(full, seed, rd)
                m.update({"condition": name, "lr": lr, "seed": seed, "wall_s": time.time() - t0, "commit": commit})
                mfile.write_text(json.dumps(m, indent=1, default=str))
                (rd / "claim").unlink(missing_ok=True)                   # the finished metrics are the record; a stale claim of a crashed worker goes too
                if not quiet:
                    print(f"[{name} lr={lr:g} seed={seed}] " + " ".join(f"{k}={v:.4g}" for k, v in m.items() if isinstance(v, (int, float)) and k not in ("seed",)), flush=True)
                results.append(m)
    return results
