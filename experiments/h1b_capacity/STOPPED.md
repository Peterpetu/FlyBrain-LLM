# H1b: stopped before completion (grid INCOMPLETE by the pre-registered rule)

Stopped on 2026-09-20 after 14 of 33 runs (about 5 CPU-hours at 4 threads), to release the CPU
for the phase-3 module ladder (E1, E2), which was queued behind it.

What the 14 finished runs show, from `runs/*/*/*/metrics.json` (primary metric `acc_in`,
40 pairs / T = 160; chance = 1/64 = 0.016):
- `mb` (the hypothesis arm), 3 learning rates x 3 seeds: 0.013-0.019, i.e. chance.
- `abl_dense_expanded`, 5 of 9 runs: 0.014-0.037, chance to marginally above.
- `gdn_matched` (the compute-matched control): not started.

With the hypothesis arm at chance in every finished run, no completion of the control could
turn the pre-registered verdict positive (the hypothesis requires `mb` to beat the control), so the
remaining 19 runs (about 7 CPU-hours) were judged not worth blocking the phase-3 ladder for. The
verdict is recorded as INCOMPLETE, not as a null: the control never ran. The finished runs are
kept unchanged; the grid can be resumed with `python3 experiments/h1b_capacity/run.py --threads 4`
(finished runs are skipped by their `metrics.json`).

The capacity question itself is re-asked at module level by P3-E1 (`experiments/p3_e1_addressing`),
which measures the sparse-versus-dense addressing principle directly on explicit write/query
streams at equal active rows, without a backbone that has to learn the task first.
