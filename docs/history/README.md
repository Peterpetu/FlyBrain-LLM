# Publication and the development history

FlyBrainLLM is published as one commit per release, starting with release 0.3.0 (3 October 2026).
The development history behind it (15 September to 3 October 2026: the code, the pre-registrations
and every result as they were committed) is kept by the author in archived copies and is not part of
the published repository. Two kinds of record depend on that history; this file states what was
checked against it before publication, so that the published repository carries the evidence.

## Pre-registrations: when each was frozen

Every pre-registered experiment's `PREREG.md` is locked by the SHA-256 of its content
(`.prereg.sha256`, line endings normalised; `flybrain/harness/prereg.py`). The lock is written by the
first run, a later edit makes every run refuse to start, and every run records the hash it ran under
(for the GPU grids, the `prereg_sha256` column of `runs_table.csv`). The lock shows that all runs of a
grid used one pre-registration; it does not by itself show that the pre-registration was written
before the results. The development history does, and its dates are listed here (UTC, commit
dates; "locked content first committed" is the first commit whose `PREREG.md` hashes to the lock,
"lock first committed" the first commit whose `.prereg.sha256` holds that hash).

| experiment | PREREG.md SHA-256 (first 16) | locked content first committed | lock first committed | first results committed | replication results committed |
|---|---|---|---|---|---|
| `aif_h1_inference` | `b7bb1488ca20aca3` | 2026-09-15 17:51 | 2026-09-15 18:05 | 2026-09-20 10:49 |  |
| `aif_h2_precision` | `97e83add4b8d4cc3` | 2026-09-20 10:49 | 2026-09-20 10:50 | none (not run to completion) |  |
| `h1_mushroom_body` | `4c033945c00f0a5b` | 2026-09-15 13:27 | 2026-09-15 13:27 | 2026-09-15 18:05 |  |
| `h1b_capacity` | `bd922119bb0aaf34` | 2026-09-15 14:19 | 2026-09-20 11:34 | none (not run to completion) |  |
| `h2_ring_attractor` | `bd656e46306523eb` | 2026-09-15 12:58 | 2026-09-15 17:30 | 2026-09-15 17:51 |  |
| `h5_graph_mask` | `8729fc370fd94737` | 2026-09-15 12:58 | 2026-09-15 13:12 | 2026-09-15 17:30 |  |
| `p3_d1_corpus` | `d10d0cddf26d2f7d` | 2026-09-20 14:19 | 2026-09-20 14:19 | 2026-09-20 14:19 (same commit) |  |
| `p3_e0_signal` | `8a07c5317e098177` | 2026-09-20 14:19 | 2026-09-20 14:19 | 2026-09-20 14:19 (same commit) |  |
| `p3_e1_addressing` | `b7334fbc8b62ac51` | 2026-09-20 13:19 | 2026-09-20 16:21 | 2026-09-20 17:56 | 2026-09-20 18:18 |
| `p3_e2_tracking` | `6f96363906d8ecb6` | 2026-09-20 14:23 | 2026-09-20 16:21 | 2026-09-20 17:56 |  |
| `p3_e3_two_trace` | `e4a8f62e21853553` | 2026-09-20 18:01 | 2026-09-20 18:18 | 2026-09-20 21:03 |  |
| `p3_e4_two_trace_long` | `30141d4f5ba9a75f` | 2026-09-21 00:28 | 2026-09-21 00:28 | 2026-09-21 06:10 | 2026-09-21 13:19 |
| `p3_s1_hybrid` | `dfa2d0e237d9dfbf` | 2026-09-29 21:10 | 2026-09-29 21:11 | 2026-09-29 22:53 (interim) |  |
| `p3_s1b_sparse` | `7b397892b24943ea` | 2026-09-29 22:49 | 2026-09-29 22:49 | 2026-10-02 08:46 |  |

Read with three qualifications. The two diagnostics `p3_d1_corpus` and `p3_e0_signal` (no
hypothesis verdict) were committed together with their first results, so for them the history does
not show the order. `aif_h2_precision` and `h1b_capacity` were frozen and never run to completion
(REPORT.md, section 4). The first committed summary of `p3_s1_hybrid` was an interim one reading
INCOMPLETE; its computed verdict was committed on 3 October 2026. The table was generated from the
history by a script that hashes each committed `PREREG.md` exactly as the harness does; it is a
record made before publication, not something the published repository can recompute.

## Run records and code identity

Each run records the commit it ran at (`env.json` and `metrics.json` in its run directory; the
`commit` column of `runs_table.csv`). Those commits belong to the development history. Before
publication `scripts/export_runs.py` checked, for each GPU grid, that `flyprinc/`, `flybrain/`, the
grid's `run.py` and its `PREREG.md` were identical in every commit its runs recorded:

| grid | runs | PREREG hashes recorded | commits recorded | code identical in every run |
|---|---|---|---|---|
| `p3_s1b_sparse` (rung 1b) | 100 | 1 (`7b397892b249`) | 11 | yes |
| `p3_s1_hybrid` (rung 1) | 95 | 1 (`dfa2d0e237d9`) | 12 | yes |

In the published repository the script reports that this check needs the development history
instead of running it. Commit hashes quoted in documents written during the work (for example in
docs/aif/DESIGN_PANEL.md, docs/aif/VERIFICATION.md and docs/phase3/BRANCHES.md) refer to the
development history as well.
