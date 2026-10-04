# Changelog

Versions mark the state of the record, not of the code alone: a version is cut when a phase's
pre-registered grids have run and their verdicts are written into RESULTS.md and REPORT.md.
Dates are the dates of the work (from the development history), not of the data releases. The
repository is published as one commit per release, starting with 0.3.0; the development history
is kept by the author (docs/history/README.md).

## 0.3.0 (2026-10-03): phase 3, the principle ladder from first principles

- Reflection on the two null phases, axiom reset, and a research panel's design: propose a
  **state**, not a gate (docs/phase3/REFLECTION.md, AXIOMS.md, DESIGN_PANEL.md, DESIGN.md).
- Module level, CPU, pre-registered (`experiments/p3_e*`, `p3_d1_corpus`): E0 and D1
  diagnostics; **E1 SUPPORTED and replicated** (sparse random addressing 0.993 against dense keys
  0.368 at 128 stored pairs, equal active rows); E2 null; E3 null (under-trained, nine points below
  a Bayes-optimal oracle); **E4 SUPPORTED and replicated** (the history states on the two-store
  memory: +0.012 over the identical model without them, +0.007 over a state-matched single store).
- System level, one 16 GB GPU, pre-registered (`flyprinc/`, `experiments/p3_s1_hybrid`,
  `p3_s1b_sparse`): the two-store memory layer with an exact chunked form, block-delayed sensing,
  the exact reduction to Gated DeltaNet, the Kenyon-cell address, a learned expansion, the
  expansion without sparsity, and product-key addressing, all in one code path.
  **Rung 1b FALSIFIED** under its frozen rule (the sparse address higher than Gated DeltaNet at all
  fifteen seed pairs, 0.997-0.9995 against 0.79-0.81, but above two seed-spreads at only one of
  three settings; the equal-memory dense key and product keys reach the same ceiling).
  **Rung 1 FALSIFIED** on all three conjuncts (the history states in trained memory-only models:
  +0.016 and +0.006 over the identical model without them at two outlier rates, 0.3 seed-spreads;
  a single store with the same memory ahead by 0.040 and 0.094; 95 runs). Rung 2, the
  pre-registered language-model test, was not run: its rules reserved it for a supported mechanism.
- Exploratory, labelled, not pre-registered (`experiments/p3_s2_lm`, `p3_k1_sparse_kernel`,
  `p3_k2_selective_writes`): 34M-parameter hybrid language models on WikiText-103 and TinyStories
  with D1-regime recall probes and memory-ablation diagnostics; an exact k-sparse memory step with
  its wall-clock comparison; a selective-writes pilot; the memory load of natural text.
- Infrastructure: hash-locked pre-registrations that refuse to run after an edit, per-run claims
  for a shared GPU, graceful STOP files, seeded per-step batches from worker processes, a
  code-identity check across a grid's runs (`scripts/export_runs.py`), data checksums and
  manifests, a reproduction manual (docs/phase3/GPU_RUNBOOK.md), 129 unit tests.
- Publication: the first published release. docs/history/README.md records, from the development
  history, when each pre-registration was frozen relative to its results, and the code-identity
  checks of the GPU grids made before publication.
- Record: README front page, REPORT.md sections 4-7, LIMITATIONS.md (GPU section), RESULTS.md,
  and docs/phase3/EXPERIMENT_PLAN.md as the chronological log of every design change and result.

## 0.2.0 (2026-09-15): phase 2, the FlyAI active-inference model

- `flyai/`: a filter model (precision-weighted free-energy inference, Gamma precision posteriors,
  arousal, an episodic memory with a precision-gated depression rule, an attractor bank,
  expected-free-energy decoding), its specification (docs/aif/), a literature sweep, a design panel
  that found three defects before any run, a line-by-line verification report, and six
  pre-registered claims (AIF-H1 to H6).
- **AIF-H1 NOT SUPPORTED** (K = 3 inference steps against K = 0: +0.7 seed-SD, inside the spread;
  behind a wider control). AIF-H2 to H6 stopped before completion for the pivot to phase 3.
- GPU benchmark scaffold (`benchmarks/`, `PREREG_tier2.md`), executed end to end at toy size on
  CPU; never run at scale (superseded by phase 3, REPORT.md section 6).

## 0.1.0 (2026-09-15): phase 1, identified circuits from the MaleCNS connectome

- `flybrain/`: readers for the MaleCNS v1.0 release, extraction and verification of the mushroom
  body and central complex against published wiring, degree-preserving and Erdos-Renyi controls,
  a fast-weight memory module (linear attention, DeltaNet, Gated DeltaNet and mushroom-body
  variants in one code path), a ring attractor, a harness with pre-registered thresholds.
- Pre-registered grids, all **FALSIFIED**: H5 (the whole graph as a recurrent mask equals its
  degree-preserving shuffle), H1 and H1a-c (the mushroom-body memory loses to compute-matched
  Gated DeltaNet; its fly-derived parts are not necessary), H1d (the measured PN-to-KC wiring adds
  nothing over random wiring of equal fan-in), H2 (the extracted ring attractor tracks state worse
  than an unstructured layer of the same size, -9.1 seed-SD). H3 and H4 pre-registered, not run.
