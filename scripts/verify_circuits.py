#!/usr/bin/env python3
"""Re-extract the MB and CX sub-connectomes from the raw mirror files and print every
verification check against published wiring. Exit code 1 if any check fails."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flybrain.connectome import load_connectome
from flybrain.connectome.io import graph_summary
from flybrain.connectome.circuits import save_circuits

c = load_connectome()
print("== whole CNS"); print(json.dumps(graph_summary(c), indent=1))
rep = save_circuits(c); bad = 0
for name, r in rep.items():
    print(f"== {name}")
    for k, v in r["stats"].items():
        if not isinstance(v, dict): print(f"   {k}: {v}")
    for k, (ok, val) in r["verification"].items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {k}: {val}"); bad += (not ok)
    for k, v in r.get("profiles", {}).items():
        print(f"   profile {k}: {[round(x, 1) for x in v]}")
sys.exit(1 if bad else 0)
