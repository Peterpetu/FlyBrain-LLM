# GPU runbook — running FlyAI at 50M-350M parameters

Everything in this repository was executed on 4 CPU cores. This is the exact procedure for
the runs that would let one make a claim about FlyAI at language-model scale. Follow the
pre-registration discipline: freeze `benchmarks/PREREG_tier2.md` (or your edited copy)
before the first run and never edit it afterwards.

## 0. Environment

```bash
git clone https://github.com/Peterpetu/FlyBrain-LLM && cd FlyBrain-LLM
pip install -e ".[dev,bench]"            # torch (CUDA build), transformers, datasets, lm_eval
python -m pytest -q tests                # ~1 min on CPU; all must pass
scripts/fetch_data.sh                    # only needed for the phase-1 connectome experiments
```
One 80 GB GPU is enough for 125M at seq 1024 / batch 32; 350M wants seq 1024 / batch 16
per GPU. Multi-GPU data-parallel is wired: prefix any command with
`torchrun --nproc_per_node=N` (`--batch` is then per rank; the token budget is split across
ranks automatically).

## 0b. One 16 GB GPU (the realistic case)

Everything above assumes an 80 GB card. On a single 16 GB GPU the plan is smaller but
still a real, pre-registered comparison. What fits and what it costs:

| config | params (with tied 50k-vocab embedding) | seq x batch x accum | est. peak memory | tokens/s (est., `--compile`) | 3 seeds x 6 arms x tokens |
|---|---|---|---|---|---|
| `configs/flyai_20m_16gb.json` | 34 M (15 M non-embedding) | 512 x 16 x 2 | ~6 GB | gdn ~25k, flyai ~3-8k | 100 M tokens each: ~2 h (gdn) / ~6-10 h (flyai) per run |
| `configs/flyai_50m_16gb.json` | 52 M (26 M non-embedding) | 512 x 8 x 4 | ~10 GB | gdn ~12k, flyai ~1.5-4k | 300 M tokens each: ~7 h (gdn) / ~20-50 h (flyai) per run |

The 16 GB configs use the memory-lean block: 1,024 KCs, d_v = 16, chunk 32 with
chunk-delayed writes (one fast-weight state per chunk instead of one per token; exact for
chunk 1, tested). Throughput numbers are estimates for a 4080/4090-class card; the first
thing to do is measure them:

```bash
python benchmarks/find_batch.py --size 20m --seq 512 --mixers flyai gdn --compile --config benchmarks/configs/flyai_20m_16gb.json
python benchmarks/find_batch.py --size 50m --seq 512 --mixers flyai gdn --compile --config benchmarks/configs/flyai_50m_16gb.json
```
It prints the largest batch that fits, tokens/s and peak memory per mixer. Then run the
20M ladder first (pre-registered in `benchmarks/PREREG_tier2.md`, section "Tier 2a"):

```bash
# one command runs the whole pre-registered ladder (hypothesis + every arm x every seed), resumable:
benchmarks/run_tier2a.sh benchmarks/configs/flyai_20m_16gb.json
# or one run at a time; --config takes size, block and train defaults, --arm merges an arm's overrides:
python benchmarks/train_flyai_lm.py --config benchmarks/configs/flyai_20m_16gb.json --seed 0 --data fineweb-edu --export
python benchmarks/train_flyai_lm.py --config benchmarks/configs/flyai_20m_16gb.json --arm gdn --seed 0 --data fineweb-edu --export
python benchmarks/train_flyai_lm.py --config benchmarks/configs/flyai_20m_16gb.json --arm no_precision --seed 0 --data fineweb-edu --export
```
(The same commands with `benchmarks/configs/flyai_tiny_test.json` and
`DATA=local:data/text/tinyshakespeare.txt` reproduce the CPU smoke test used to verify the
ladder script here.) LR sweeps: pass `--lr` on seed 0 for each value, keep the best.
If FlyAI trains below 15 % of the GDN twin's tokens/s even with `--compile`, follow the
pre-registered drop order (PREREG_tier2.md) and record each drop. Do not go to 50M until
the 20M ladder has a result; do not run 125M on 16 GB.

## 1. Sanity on the GPU (10 minutes)

```bash
python benchmarks/train_flyai_lm.py --size tiny --mixer flyai --data local:data/text/tinyshakespeare.txt --steps 200 --seq 128 --batch 32 --bf16 --compile
python benchmarks/train_flyai_lm.py --size tiny --mixer gdn   --data local:data/text/tinyshakespeare.txt --steps 200 --seq 128 --batch 32 --bf16
```
Both must reach loss ~2.5 (byte-level) and print `tok_per_s`. Record the ratio of the two
throughputs: this is the compute-matching factor you will need in step 3.

## 2. The pre-registered 125M comparison (the real experiment)

Conditions (see `benchmarks/configs/flyai_125m.json`): `flyai` (hypothesis: v0.2
closed-form precision fusion), `gdn` (matched control; same slots, Gated DeltaNet — use the
flash-linear-attention kernel at GPU scale), `learned_gate_fusion` (learned gate at equal
parameters), `loop_fusion_K2` (the v0 K-step loop), `no_precision` (learned write gate),
`surprise_gate` (raw surprise), `no_regulation`, `no_attractor`, `no_memory`,
`delayed_write_C64` (efficiency arm), `attention` (reference). Run the 50M ladder first
(section 2b); an ingredient goes to 125M only if its removal costs > 2 seed-SD. Three seeds
each; any positive verdict is replicated with 6 fresh seeds. LR sweep: gdn {1e-3, 2e-3,
4e-3} at least; flyai {2e-3, 3e-3, 4e-3}. Same token budget for every condition.

```bash
for seed in 0 1 2; do
  python benchmarks/train_flyai_lm.py --size 125m --mixer flyai --data fineweb-edu --tokens 2.5e9 --seq 1024 --batch 32 --lr 3e-3 --seed $seed --bf16 --compile --export
  python benchmarks/train_flyai_lm.py --size 125m --mixer gdn   --data fineweb-edu --tokens 2.5e9 --seq 1024 --batch 32 --lr 2e-3 --seed $seed --bf16 --export
  python benchmarks/train_flyai_lm.py --size 125m --mixer flyai --data fineweb-edu --tokens 2.5e9 --seq 1024 --batch 32 --lr 3e-3 --seed $seed --bf16 --compile --export --block '{"fusion":"learned"}' --out benchmarks/runs/learned_gate_fusion
  python benchmarks/train_flyai_lm.py --size 125m --mixer flyai --data fineweb-edu --tokens 2.5e9 --seq 1024 --batch 32 --lr 3e-3 --seed $seed --bf16 --compile --export --block '{"memory_gate":"learned"}' --out benchmarks/runs/no_precision
  python benchmarks/train_flyai_lm.py --size 125m --mixer flyai --data fineweb-edu --tokens 2.5e9 --seq 1024 --batch 32 --lr 3e-3 --seed $seed --bf16 --compile --export --block '{"regulate_core":false}' --out benchmarks/runs/no_regulation
done
```

### 2b. The 50M ingredient ladder (run this before 125M)

Same commands with `--size 50m --tokens 1e9`, one arm per `controls` entry of
`benchmarks/configs/flyai_50m.json` (pass each entry's `block` dict to `--block`). Report
val loss, tokens/s and measured FLOPs/token per arm; the drop order if FlyAI trains below
15 % of the GDN twin's throughput is in `benchmarks/PREREG_tier2.md`.
Expected wall-clock (single 80 GB GPU): gdn ~6 h per run; flyai 2-4x that (sequential
inner loop; see benchmarks/README.md). If flyai is more than 4x slower, reduce `--K` to 1
first and report it.

## 3. Evaluation

```bash
benchmarks/run_lm_eval.sh benchmarks/runs/flyai_125m_seed0/hf     # zero-shot suite + recall suite
benchmarks/run_lm_eval.sh benchmarks/runs/gdn_125m_seed0/hf
```
Report, per condition: mean +/- SD over seeds of WikiText-103 ppl, LAMBADA, HellaSwag,
PIQA, ARC-e/c, WinoGrande, BoolQ, NIAH pass rates, RULER, BABILong; the paired bootstrap
CI of flyai minus gdn on each; training tokens/s and total GPU-hours (compute matching is
by tokens here, so the wall-clock ratio must be stated next to every number).

## 4. What would count as a result

Per `benchmarks/PREREG_tier2.md`: flyai beats gdn on the recall suite by > 2 seed-SD with
CI excluding 0, matches perplexity within 1 %, and the gain disappears in `learned_gate_fusion`
or `no_precision` (whichever ingredient it is attributed to). Anything else is a null and is
reported as such. Repeat at 50M and 350M before saying anything about trend.

## 5. Bayesian wind-tunnel (recommended extra validation)

Train the tiny model on synthetic HMM sequences whose exact posterior is known in closed
form and measure the KL between the model's predictive distribution and the true
posterior predictive (target: 1e-3 bits, the level small transformers reach). This tests
whether the precision / belief machinery tracks a real posterior; it is cheap and does
not need the GPU. Script scaffold: `experiments/aif_h6_windtunnel/` (to be added).
