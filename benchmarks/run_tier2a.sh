#!/usr/bin/env bash
# Run the whole Tier 2a ladder (benchmarks/PREREG_tier2.md) from a config: every arm x every seed.
#   benchmarks/run_tier2a.sh benchmarks/configs/flyai_20m_16gb.json [extra train_flyai_lm.py args...]
# Resumable: a FINISHED run (completion marker `done` present; model.pt alone is written at every eval) is skipped.
set -euo pipefail
CFG=${1:?config json}; shift || true
cd "$(dirname "$0")/.."
SEEDS=$(python -c "import json;print(' '.join(str(s) for s in json.load(open('$CFG'))['train'].get('seeds',[0,1,2])))")
ARMS=$(python -c "import json;print(' '.join(json.load(open('$CFG'))['controls']))")
SIZE=$(python -c "import json;print(json.load(open('$CFG'))['size'])")
for seed in $SEEDS; do
  ls benchmarks/runs/flyai_${SIZE}_seed${seed}_lr*/done >/dev/null 2>&1 || python benchmarks/train_flyai_lm.py --config "$CFG" --seed "$seed" --data "${DATA:-fineweb-edu}" --export "$@"
  for arm in $ARMS; do
    ls benchmarks/runs/${arm}_${SIZE}_seed${seed}_lr*/done >/dev/null 2>&1 || python benchmarks/train_flyai_lm.py --config "$CFG" --arm "$arm" --seed "$seed" --data "${DATA:-fineweb-edu}" --export "$@"
  done
done
echo "Tier 2a ladder done: benchmarks/runs/"
