"""flybrain.harness.run_grid: resumability, provenance, and parallel workers claiming runs."""
import json

import pytest

from flybrain.harness import run_grid
from flybrain.harness.prereg import PreregError
from flybrain.harness.runner import run_dir_for


def grid(tmp_path):
    (tmp_path / "PREREG.md").write_text("test pre-registration\n")
    calls = []
    def train_fn(cfg, seed, rd):
        calls.append((cfg["condition"], seed)); return {"acc": 0.5 + seed / 100}
    return {"a": {"x": 1}, "b|r0.15": {"x": 2}}, calls, train_fn


def test_finished_runs_are_skipped_and_prereg_locked(tmp_path):
    conds, calls, fn = grid(tmp_path)
    r1 = run_grid(tmp_path, conds, [0.0], [0, 1], fn, quiet=True); assert len(r1) == 4 and len(calls) == 4
    r2 = run_grid(tmp_path, conds, [0.0], [0, 1], fn, quiet=True); assert len(r2) == 4 and len(calls) == 4       # resumed: nothing re-run
    assert (run_dir_for(tmp_path, "b|r0.15", 0.0, 1) / "metrics.json").exists()                                   # '|' mapped for Windows folders
    (tmp_path / "PREREG.md").write_text("edited after results\n")
    with pytest.raises(PreregError): run_grid(tmp_path, conds, [0.0], [0, 1], fn, quiet=True)


def test_claimed_runs_are_left_to_their_worker_and_the_final_pass_collects_them(tmp_path):
    conds, calls, fn = grid(tmp_path)
    held = run_dir_for(tmp_path, "a", 0.0, 1); held.mkdir(parents=True); (held / "claim").write_text("pid 1\n")   # another worker holds (a, seed 1)
    r = run_grid(tmp_path, conds, [0.0], [0, 1], fn, quiet=True, claim=True)
    assert ("a", 1) not in calls and len(r) == 3 and not any((run_dir_for(tmp_path, c, 0.0, s) / "claim").exists() for c in conds for s in (0, 1) if (c, s) != ("a", 1))
    r = run_grid(tmp_path, conds, [0.0], [0, 1], fn, quiet=True)                                                  # final pass: the crashed worker's run is re-run
    assert calls.count(("a", 1)) == 1 and len(r) == 4 and not (held / "claim").exists()


def test_a_stop_file_stops_the_grid_between_runs_and_resuming_completes_it(tmp_path):
    conds, calls, fn = grid(tmp_path)
    (tmp_path / "STOP").write_text(""); r = run_grid(tmp_path, conds, [0.0], [0, 1], fn, quiet=True); assert r == [] and calls == []
    (tmp_path / "STOP").unlink(); r = run_grid(tmp_path, conds, [0.0], [0, 1], fn, quiet=True); assert len(r) == 4 and len(calls) == 4


def test_a_claim_on_a_run_finished_meanwhile_is_released(tmp_path):
    conds, calls, fn = grid(tmp_path)
    run_grid(tmp_path, {"a": conds["a"]}, [0.0], [0], fn, quiet=True); n = len(calls)
    rd = run_dir_for(tmp_path, "a", 0.0, 0); from flybrain.harness.runner import _claim
    assert _claim(rd) and not _claim(rd)                                                                          # atomic: the second claim fails
    (rd / "claim").unlink(); r = run_grid(tmp_path, {"a": conds["a"]}, [0.0], [0], fn, quiet=True, claim=True)
    assert len(calls) == n and len(r) == 1 and json.loads((rd / "metrics.json").read_text())["acc"] == 0.5
