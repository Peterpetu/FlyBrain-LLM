"""Text from a finished prototype run (experiments/p3_s2_lm/prototype.py): rebuilds the model from the run's config.json, loads
model.pt and samples continuations of a prompt (temperature and top-k sampling; stops at the end-of-text token). The model is
run uncompiled on the growing sequence, so every token is produced by the same forward pass that was trained.
Usage: python experiments/p3_s2_lm/sample.py --run experiments/p3_s2_lm/prototype_runs/tinystories_dense_s0 --prompt "Once upon a time" --n 3"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm  # noqa: E402


@torch.no_grad()
def generate(model, ids, max_new, temperature, top_k, eot, gen):
    for _ in range(max_new):
        logits = model(ids)[0, -1].float() / temperature
        if top_k: logits[logits < logits.topk(top_k).values[-1]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=gen)
        if int(nxt) == eot: break
        ids = torch.cat([ids, nxt[None]], 1)
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--prompt", default="Once upon a time"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max_new", type=int, default=200); ap.add_argument("--temperature", type=float, default=0.8); ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    from transformers import AutoTokenizer
    run = Path(a.run); c = json.loads((run / "config.json").read_text()); tok = AutoTokenizer.from_pretrained("gpt2")
    model = lm.build_model(c["d_model"], c["n_layers"], c["n_heads"], c["window"], c["attn_every"], c["layer"], tied=True, emb_std=0.02).to(a.device)
    model.load_state_dict(torch.load(run / "model.pt", map_location=a.device)); model.eval()
    gen = torch.Generator(device=a.device).manual_seed(a.seed); start = torch.tensor([tok(a.prompt)["input_ids"]], device=a.device)
    for i in range(a.n):
        out = generate(model, start.clone(), a.max_new, a.temperature, a.top_k, tok.eos_token_id, gen)
        print(f"--- sample {i + 1} ({c['corpus']}, {c['arm']} keys) ---\n{tok.decode(out[0].tolist())}\n", flush=True)
