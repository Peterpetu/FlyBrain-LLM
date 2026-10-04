#!/usr/bin/env bash
# Score an exported FlyAI checkpoint (benchmarks.hf_flyai.export_checkpoint, i.e. `train_flyai_lm.py --export`
# -> <run>/hf) with lm-eval-harness on the standard zero-shot suite and the recall / long-context suite.
# `python -m benchmarks.lm_eval_flyai` is `python -m lm_eval` with the "flyai" model type registered first,
# which is what lets `--model hf` load the checkpoint. Needs the tokenizer files in $CKPT and a GPU.
set -euo pipefail
CKPT=$(cd "${1:?checkpoint dir (output of export_checkpoint)}" && pwd)
cd "$(dirname "$0")/.."
LM_EVAL="python -m benchmarks.lm_eval_flyai"
$LM_EVAL --model hf --model_args pretrained=$CKPT,trust_remote_code=True \
  --tasks wikitext,lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,boolq \
  --num_fewshot 0 --batch_size auto --output_path "$CKPT/lm_eval_zero_shot.json"
# Recall suite at the pre-registered Tier 2a lengths (PREREG_tier2.md: NIAH single / multikey at 512-2048).
# batch_size 1: the recurrent layers cannot take left-padded batches (FlyAIForCausalLM rejects them), and
# generate_until tasks pad on the left when batched. Task names follow lm-eval-harness >= 0.4.5 (RULER subtasks);
# check `lm_eval --tasks list` for the installed version.
$LM_EVAL --model hf --model_args pretrained=$CKPT,trust_remote_code=True \
  --tasks niah_single_1,niah_multikey_1,niah_multikey_2 --metadata '{"max_seq_lengths":[512,1024,2048]}' \
  --batch_size 1 --output_path "$CKPT/lm_eval_recall.json"
# 125M plan (PREREG_tier2.md, 4k-64k): add  --tasks ruler,babilong --metadata '{"max_seq_lengths":[4096,16384,65536]}'
# after exporting with max_position_embeddings >= the longest length.
