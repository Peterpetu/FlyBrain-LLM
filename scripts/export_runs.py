"""One row per finished run of an experiment grid, as a CSV next to its summary (`experiments/<exp>/runs_table.csv`, or
`runs_table_<variant>.csv`). The run folders themselves are not committed (large logs); this table keeps every per-seed
number behind a verdict in the repository. Usage: python scripts/export_runs.py p3_s1b_sparse [--variant replication]"""
import argparse, csv, json, subprocess
from pathlib import Path

R = Path(__file__).resolve().parents[1]
E = R / "experiments"
COMMIT_MAP = R / "docs" / "history" / "COMMIT_MAP.tsv"      # optional, not published: maps recorded commits into an archived copy of the history


def _exists(commit):
    return subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=R, capture_output=True).returncode == 0


def resolve(commit):
    """A run records the commit it ran at. The published repository does not contain those commits (docs/history/README.md);
    with the development history fetched into this clone they resolve directly or through COMMIT_MAP. None if unavailable."""
    if _exists(commit):
        return commit
    if COMMIT_MAP.exists():
        for line in COMMIT_MAP.read_text().splitlines()[1:]:
            old, new = line.split()
            if old.startswith(commit) and _exists(new):
                return new
    return None


def code_identity(exp, commits):
    """The runs of one grid may record different commits (other files were committed meanwhile); they are comparable only if
    the code they ran is the same. Returns the paths that differ between the first recorded commit and each later one, or
    None when the recorded commits are not available in this clone."""
    paths = ["flyprinc", "flybrain", f"experiments/{exp}/run.py", f"experiments/{exp}/PREREG.md"]
    first, diffs = commits[0], {}
    found = {c: resolve(c) for c in commits}
    if None in found.values():
        return first, None
    for c in commits[1:]:
        d = subprocess.run(["git", "diff", "--name-only", found[first], found[c], "--", *paths], cwd=R, capture_output=True, text=True)
        diffs[c] = d.stdout.split() if d.returncode == 0 else [f"git diff failed: {d.stderr.strip()}"]
    return first, diffs

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("experiment"); ap.add_argument("--variant", default=None); a = ap.parse_args()
    root = E / a.experiment / ("runs" if a.variant is None else f"runs_{a.variant}")
    rows = []
    for mf in sorted(root.glob("*/lr*/seed*/metrics.json")):
        m = json.loads(mf.read_text()); env = json.loads((mf.parent / "env.json").read_text()) if (mf.parent / "env.json").exists() else {}
        rows.append({"condition": m.get("condition"), "seed": m.get("seed"), "lr": m.get("lr"), "commit": m.get("commit", env.get("commit")),
                     "prereg_sha256": env.get("prereg_sha256"), "device": env.get("device"),
                     **{k: v for k, v in m.items() if isinstance(v, (int, float, bool)) and k not in ("seed", "lr")}})
    if not rows: raise SystemExit(f"no finished runs under {root}")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = E / a.experiment / ("runs_table.csv" if a.variant is None else f"runs_table_{a.variant}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} runs -> {out}")
    commits = sorted({r["commit"] for r in rows if r["commit"]}, key=lambda c: min(i for i, r in enumerate(rows) if r["commit"] == c))
    hashes = {r["prereg_sha256"] for r in rows}
    first, diffs = code_identity(a.experiment, commits)
    if diffs is None:
        verdict = ("code identity not re-checked: the recorded commits belong to the development history, which this clone does not "
                   "contain (docs/history/README.md records the check made before publication)")
    else:
        bad = {c: d for c, d in diffs.items() if d}
        verdict = "code identical to the first run's commit " + first[:7] + " in every run" if not bad else f"CODE DIFFERS from {first[:7]}: {bad}"
    print(f"PREREG hashes recorded: {len(hashes)} ({', '.join(h[:12] for h in hashes if h)}); commits recorded: {len(commits)}; {verdict}")
