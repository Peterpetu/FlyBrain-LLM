# Rung-1b pilot logs (exploratory, seed 0; recorded in PREREG.md before freezing)

- `pilot_results.txt`: the result line of every pilot run. The first two lines and the next four come
  from `diagnostics/pilot_gdn.py` (the grid's own arms and code path; key width 32, then 16, where
  the line lists the overrides). The last two come from `diagnostics/pilot_capacity.py` (this
  repository's full layer at P = 32).
- `dk32_*.log.jsonl`, `dk16_*.log.jsonl`: per-100-step training logs (loss, training accuracy, GPU
  step time) of the `pilot_gdn.py` runs at key width 32 and 16.
- `dk32_sparse_P64.log.jsonl` is a partial run (about 100 steps). It was stopped when the key-width
  rule found no qualifying P at width 32, before it produced a result.
