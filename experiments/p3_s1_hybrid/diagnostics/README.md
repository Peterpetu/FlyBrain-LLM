# Rung-1 learnability diagnostics (GPU, RTX 4080 SUPER)

Scripts behind the learnability findings recorded in docs/phase3/EXPERIMENT_PLAN.md ("Learnability").
Run from the repository root with a CUDA GPU, e.g. `python experiments/p3_s1_hybrid/diagnostics/mqar_reference.py`.

| script | question |
|---|---|
| bench_compile.py | step time eager vs compiled chunk step, per-token vs block-delayed sensing (args: modes batch arm sense_block chunk) |
| block_fraction.py | fraction of key re-presentations falling in the same sensing block |
| mqar_reference.py | a standard reference transformer vs the repository's model on textbook MQAR (same batches) |
| mqar_ablate.py | which input/output setting blocked recall learning (positions, output tying, embedding scale) |
| position_test.py | length vs format, and position schemes, at 256 tokens |
| curriculum_test.py | a length curriculum from 128 to 2,048 tokens on the rung-1 task |
| learn_sweep.py | five model types on two dense recall tasks |
| pilot_dense.py | the reliability pilot for the redesigned rung 1 (memory-only models, dense task, three seeds) |
