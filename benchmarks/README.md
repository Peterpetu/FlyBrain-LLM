# Benchmarks

Two tiers. Tier 1 is what this repository actually ran (CPU) and what RESULTS.md reports.
Tier 2 is the GPU-scale evaluation a downstream claim would require; we ship the scripts
and configs and make **no claim** about their outcome.

## Tier 1 — architecture-research benchmarks (ran here)

| Benchmark | What it measures | Where |
|---|---|---|
| MQAR (Zoology; Arora et al. 2023) | in-context associative recall capacity of sub-quadratic mixers | `experiments/h1_mushroom_body` |
| Compass state tracking with 4-16x length extrapolation | cyclic-group state tracking; the S_n-word-problem family where transformers/diagonal linear RNNs fail beyond training length | `experiments/h2_ring_attractor` |
| Character-level LM, TinyShakespeare bits/char | generic sequence modelling with graph-masked recurrence | `experiments/h5_graph_mask` |

## Tier 2 — token-level LM at 50-350 M parameters + lm-eval-harness (scripted, not run here)

### Phase 2: FlyAI hybrid (`train_flyai_lm.py`, `configs/flyai_*.json`, `hf_flyai.py`)

`train_flyai_lm.py` trains a 3:1 hybrid decoder (FlyAI filter blocks with softmax
attention every 4th block) or its matched twins (`--mixer gdn` puts Gated-DeltaNet mixers
in the same slots; `--mixer attention` is all-attention). The FlyAI block uses the v0.2
closed-form precision fusion; the learned-gate and K-loop fusions are config arms. It has a local byte-level data
mode that runs anywhere (`--data local:<file>`; this is what the repository's own smoke
test uses) and a FineWeb-Edu streaming mode with the GPT-2 tokenizer for GPU runs.
`configs/flyai_{50m,125m,350m}.json` hold the pre-registered settings and the control
set (gdn, learned_gate_fusion, no_precision, no_regulation, attention and the ingredient ladder). `hf_flyai.py` wraps a trained
model as a HuggingFace `PreTrainedModel` so `lm-eval` scores it unchanged
(`lm_eval_flyai.py`).

Every run logs measured FLOPs/token from `torch.utils.flop_counter` (one forward + backward
on a real batch; counts matmul / attention / convolution FLOPs, not elementwise work, so it
is the standard comparable count across twins, and it under-counts the FlyAI block's
elementwise per-token operations — wall-clock tokens/s is logged beside it for that
reason). Multi-GPU: `torchrun --nproc_per_node=N ...` (DDP, per-rank data shards by seed;
verified with two CPU processes here).

Verified here (CPU, 1 thread, tiny 0.3 M model, seq 64, batch 8, 30 steps): all three
mixers train from loss 5.5 to 3.35 on byte-level Shakespeare with the same script.
Throughput on that CPU: flyai 420 tok/s, gdn 1,600 tok/s, attention 18,000 tok/s.

**Throughput caveat for the GPU runs (read before launching).** The FlyAI block is a
strictly sequential recurrence (closed-form fusion per token; the K-step loop exists only in the
`loop_fusion_K2` ablation arm); unlike Gated
DeltaNet it has no chunk-parallel training form (the precision gate depends on the running
error). Per layer and token it launches ~100 small kernels, so on a GPU wall-clock is
launch-bound: expect roughly 5-20x lower tokens/s than the GDN twin at the same size
unless `--compile` (fuses the per-token step with torch.compile in the default mode on any
device; `--compile_mode reduce-overhead` (CUDA graphs) is opt-in and has not been verified for
this per-token loop) and large batches are used. Practical
guidance: sequence 1024, batch as large as memory allows, `--compile`, `--bf16`; budget
2-4x the GDN wall-clock for the 125 M run. Two mitigations are implemented: per-chunk activation
checkpointing (always on in training; keeps O(T/64) fast-weight states instead of O(T)) and
the chunk-delayed write arm (`"delayed_write": true` in the block config: reads and gates
use the chunk-start memory, the chunk's updates are applied once with cumulative decay;
chunk size 1 reproduces the sequential module exactly, tested). A fully chunk-parallel
block is listed as a pre-registered fallback arm in docs/aif/DESIGN.md.

`train_lm.py` trains a hybrid decoder (3 fast-weight layers : 1 attention layer, the
Qwen3-Next / Kimi-Linear layout) with the mixer chosen by `--mixer {mb,gdn,attention}` on
FineWeb-Edu, at matched parameters and matched FLOPs. `run_lm_eval.sh` then scores the
checkpoint on the standard zero-shot suite (WikiText-103 ppl, LAMBADA, HellaSwag, PIQA,
ARC-e/c, WinoGrande, BoolQ) plus the long-context/recall suite (NIAH, RULER, BABILong).

Requirements for the full plan: one 80 GB GPU per 125 M-parameter run; for a single
16 GB GPU use the Tier 2a plan (`PREREG_tier2.md`, `configs/*_16gb.json`,
`find_batch.py`, and the runbook section "One 16 GB GPU"). Original requirement for ~6 h at 2.5 B tokens; the
`datasets`, `transformers`, `lm_eval` packages; network access to HuggingFace.

Pre-registration for Tier 2 lives in `benchmarks/PREREG_tier2.md` and must be frozen
before the first run, as for every experiment here.
