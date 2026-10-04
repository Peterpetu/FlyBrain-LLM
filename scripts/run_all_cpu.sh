#!/usr/bin/env bash
# Reproduce every CPU-scale experiment in this repository, in the order they were run.
# Each script reads its PREREG.md, freezes its hash on first run, and is resumable.
# Total wall-clock on 4 cores: roughly a day. Use --threads to match your machine.
set -euo pipefail
cd "$(dirname "$0")/.."
T=${THREADS:-4}
scripts/fetch_data.sh
python scripts/verify_circuits.py > results/circuit_verification.txt
# phase 1: connectome wiring as prior (H5 calibration, H1 mushroom body, H2 ring attractor)
python experiments/h5_graph_mask/run.py --threads $T
python experiments/h1_mushroom_body/run.py --threads $T
python experiments/h2_ring_attractor/run.py --threads $T
python experiments/h1b_capacity/run.py --threads $T   # capacity regime follow-up (ran after the AIF chain)
# phase 2: active-inference model (AIF-H1..H6)
for e in aif_h1_inference aif_h2_precision aif_h3_arousal aif_h4_criticality aif_h5_efe aif_h6_windtunnel; do
  python experiments/$e/run.py --threads $T
done
python scripts/build_results.py
python scripts/make_figures.py
