"""`python -m benchmarks.lm_eval_flyai ...` is `python -m lm_eval ...` with the FlyAI model type registered
first, so `--model hf --model_args pretrained=<export dir>,trust_remote_code=True` resolves model_type
"flyai" (see benchmarks/hf_flyai.py; the export dir comes from `export_checkpoint` / `train_flyai_lm.py --export`).

From Python: `evaluate(<export dir>, ["wikitext", "piqa"])` wraps `lm_eval.simple_evaluate` the same way."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmarks.hf_flyai  # noqa: E402,F401  (registers "flyai" with AutoConfig / AutoModelForCausalLM)


def evaluate(pretrained, tasks, num_fewshot: int = 0, batch_size="auto", **kwargs):
    import lm_eval
    return lm_eval.simple_evaluate(model="hf", model_args=f"pretrained={pretrained},trust_remote_code=True",
                                   tasks=list(tasks), num_fewshot=num_fewshot, batch_size=batch_size, **kwargs)


if __name__ == "__main__":
    runpy.run_module("lm_eval", run_name="__main__", alter_sys=True)
