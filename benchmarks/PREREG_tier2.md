# PREREG — Tier 2 (GPU): does the MB layer help a token-level LM at matched compute?

Not run in this repository. Frozen for whoever runs it.

**Hypothesis.** In a 3:1 hybrid (fast-weight : attention) decoder at 125 M parameters
trained on 2.5 B FineWeb-Edu tokens, replacing Gated-DeltaNet layers by MB layers at
matched FLOPs improves recall-heavy evaluation (NIAH, RULER, SWDE/FDA-style extraction)
without hurting perplexity or the zero-shot suite.

**Conditions.** `gdn` (baseline, tuned LR sweep of 5), `mb` (LR sweep of 3),
`mb_dense_expanded` (ablation), `attention_only` (reference). 3 seeds each.

**Metrics.** WikiText-103 ppl; LAMBADA acc; HellaSwag, PIQA, ARC-e, ARC-c, WinoGrande,
BoolQ (lm-eval-harness, zero-shot, acc_norm where defined); NIAH pass rate at 4k/16k/64k;
RULER average; BABILong qa1-qa5 at 4k-64k.

**Thresholds.** MB is supported only if it beats `gdn` on the recall suite by > 2 seed-SD
with CI excluding 0, matches perplexity within 1 %, and the gain disappears in
`mb_dense_expanded`. Anything else is reported as null. Scaling: repeat at 50 M and 350 M
before any claim about trend.

> **Phase-2 revision (before any GPU run; the phase-1 MB layer above is superseded).** The
> hypothesis arm is `flyai` (v0.2 closed-form block); the twin is `gdn` (same slots, key dim =
> head dim, the script's default); the attribution arms are `no_precision` (learned gate in
> place of the inferred precision gate) and `learned_gate_fusion`; the reference is
> `attention`. Threshold: `flyai` beats `gdn` on the recall suite by > 2 seed-SD with CI
> excluding 0, matches its validation loss within 0.02 nats, AND the recall gain is absent in
> `no_precision` or `learned_gate_fusion` (whichever ingredient it is attributed to).
> `mb_dense_expanded` and `K0` no longer exist as arms. The ingredient ladder below names the
> config arms exactly: `const_gate`, `scalar_precision`, `lyapunov_target_-0.10` /
> `lyapunov_target_0.00`, `chunk_C32` / `chunk_C128`, `per_head_codes`, `active_frac_15`,
> `no_arousal`, `gamma_pi3`, plus `surprise_gate`, `no_memory`, `no_attractor`, `no_regulation`,
> `loop_fusion_K2`, `delayed_write_C64`. LR sweeps are in each config's `train.lr_sweep`; runs
> are named with their LR so sweep runs never overwrite each other.


## Protocol additions (phase 2, before any GPU run)

* **Twins.** `flyai` (closed-form fusion, v0.2) vs `gdn` in the same 3:1 slots; at GPU
  scale the GDN twin must be the flash-linear-attention chunked kernel, not the eager
  per-token loop in this repository (which is for CPU tests only). Parameter counts are
  matched by construction (no amortised network or dense g0 in the block; set
  `--gdn_key` to d / n_heads); every table row reports measured FLOPs/token
  (torch.utils.flop_counter), tokens/s, parameter count and recurrent state size.
* **Replication rule.** Any positive three-seed verdict is re-run at the winning learning
  rate with six fresh seeds (hypothesis and strongest control) before it is reported as
  supported; discovery and replication effect sizes are reported side by side.
* **Ingredient ladder at 50M first.** Arms: closed-form vs learned-gate vs K=2 loop
  fusion; write-gate ladder {precision, surprise, learned, const}; no-attractor;
  no-memory; scalar vs grouped precision; Lyapunov target sweep; chunk size C for
  checkpointing/delayed writes; GDN and attention twins. An ingredient goes to 125M only
  if removing it costs > 2 seed-SD; FlyAI goes to 125M only if within 0.02 nats/token of
  the GDN twin at 50M.
* **Throughput gate.** If FlyAI trains at < 15 % of the GDN twin's tokens/s at 50M after
  `--compile`, the pre-registered drop order is: K-loop arms, per-token Lyapunov
  estimation (already every 8 tokens), attractor bank, then the chunk-parallel fallback
  block; each drop is recorded.


## Tier 2a — one 16 GB GPU (pre-registered reduced plan)

**Model.** `20m` (d 384, 8 layers, 6 heads; 15 M non-embedding parameters) with the 16 GB
block (`configs/flyai_20m_16gb.json`: 1,024 KCs, d_v 16, chunk 32, delayed writes),
FineWeb-Edu, GPT-2 tokenizer, sequence 512, effective batch 32, 100 M tokens per run,
bf16, `--compile`.

**Arms (each 3 seeds, same tokens).** `flyai` (hypothesis), `gdn` (matched twin; same
slots, key = head dim), `learned_gate_fusion`, `no_precision`, `surprise_gate`,
`no_regulation`, `attention`. LR: gdn {1e-3, 2e-3, 4e-3} on seed 0 then best; flyai
{2e-3, 3e-3, 4e-3} on seed 0 then best.

**Metrics.** Validation loss on 64 held-out FineWeb-Edu documents (nats/token; primary),
tokens/s, peak memory, measured FLOPs/token; after training, `benchmarks/run_lm_eval.sh`
zero-shot suite (descriptive at this size: 20M-parameter models are near chance on most
of it) and the recall suite (NIAH single/multikey at 512-2048).

**Thresholds.** FlyAI supported at this tier only if its validation loss is within 0.02
nats of `gdn` AND it beats `gdn` on NIAH multikey by > 2 seed-SD with CI excluding 0, AND
the gain is absent in `learned_gate_fusion` or `no_precision`. Anything else is a null and
is reported as such. A positive verdict is replicated with 6 fresh seeds before it is
called supported. The 50M plan (`configs/flyai_50m_16gb.json`, 300 M tokens) runs only
after the 20M ladder has a result and only for the surviving arms.
