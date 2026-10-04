# FlyBrainLLM

## Where this stands (read this first)

**What we are trying to find out.** Whether a principle from the fly's memory system makes today's
best sequence-memory layer (the gated delta rule, the core of models such as Gated DeltaNet)
better. Two earlier phases (connectome wiring as a prior; an active-inference model) came back
null. The third phase restarted from first principles with one rule learned from those nulls:
give the learned mechanism a new **state** it cannot compute for itself, never a hand-written rule.

**What we found, in order** (findings 1-5 and 8 are pre-registered runs with matched controls,
mechanically judged in [RESULTS.md](RESULTS.md); findings 6 and 7 are exploratory measurements,
labelled as such, with no verdict):

1. **Sparse addressing works, and it is a big effect.** A fixed random sparse code recalls 0.99 of
   128 stored pairs where dense keys recall 0.37, at equal active rows per token (E1; 5 seeds,
   replicated on 6 fresh seeds). This is the fly's Kenyon-cell coding principle, and it holds up.
   (Equal active rows is not equal compute: the expansion itself costs arithmetic; finding 7.)
2. **The history states alone do nothing.** Giving a learned write gate the "how often was this
   address confirmed or contradicted" states did not help on a single memory store (E2). Putting
   the same states on a two-store memory with a learned readout also did nothing at first (E3).
3. **A Bayes-optimal oracle showed there was room to improve.** Every learned variant was stuck
   about 9 points below what is theoretically achievable on the same streams, which meant they
   were under-trained, not at a ceiling.
4. **With ten times more training, the states help, a little.** On the two-store memory the states
   give about +1.2 points over the identical model without them and +0.7 over a state-matched
   single store (E4; 5 seeds). This passed its pre-registered test on fresh seeds and then a
   second replication on six more seeds. It is the first supported, replicated positive result for a
   fly-derived state in this project. It is real and it is small: about one ninth of the remaining
   gap to the oracle, and it does not move the overall accuracy at all. The gain is entirely in
   protecting confirmed memories from one-off wrong observations.

**Inside trained sequence models (GPU, October 2026).** Findings 1 and 4 were taken into trained
models, each against its strongest fair controls and with its verdict rule frozen before the runs.

5. **Sparse addressing in a trained model (rung 1b): the pre-registered verdict is FALSIFIED, and
   the controls say why it is not a fly-specific win.** Against Gated DeltaNet with 16-dimensional
   keys (the width at which a one-seed pilot showed the dense store to be the bottleneck), the
   Kenyon-cell address recalled 0.999, 0.999 and 0.997 at 32, 64 and 128 stored pairs, where
   Gated DeltaNet recalled 0.79, 0.81 and 0.79 (5 seeds per arm, 100 runs). It was higher at every
   one of fifteen seed pairs. The frozen rule asked for a gain above two seed-spreads at two of the three settings; it
   passed at one, because Gated DeltaNet's own results are erratic (one run never learned), so the
   verdict is FALSIFIED. The controls: a dense key with the same memory size reaches 1.000, product
   keys (the closest learned relative) reach 0.999-1.000, and a learned expansion does as well as
   the random one (1.000; a hair better, so the fly's random wiring is not what matters). A random
   expansion without the winner-take-all sparsity gains nothing. So a sparse address
   over a bigger store does remove the capacity limit, but any of several ways of getting a bigger
   store does the same; what the sparse forms save is cost (one sixteenth of the rows touched), and
   with 32-dimensional keys Gated DeltaNet hardly ran out of memory on this task (0.98 and 0.95
   in the one-seed pilot).
6. **In language, the memory is not what recalls far facts, so no address could help (exploratory,
   two seeds).** A 34M-parameter model trained on WikiText-103 recalls a repeated fact 0.78 of the
   time inside its attention window and 0.55 beyond it, and the figure beyond the window hardly
   moves when the memory is made to forget within 7 tokens (0.55 to 0.52). The memory carries much
   inside the window and in perplexity, and its learned decays are long enough to hold far facts:
   the model has the capacity and has not learned the retrieval. The Kenyon-cell address changes
   nothing far (0.550 against 0.551), costs nearby recall and perplexity (56 against 59), and runs
   2-2.7 times slower. On TinyStories both arms reach the same perplexity (5.84; one seed each).
7. **The address's remaining advantage, cost, is arithmetic only.** An exact sparse memory step
   does 4 times less arithmetic than the equal-memory dense store and is slower on this GPU (44
   against 21 ms per layer forward at rung-1b size, 130 against 77 at language-model size;
   compiled, medians of 15 interleaved measurements), because dense products run on tensor cores
   and gathers do not.
8. **The history states in a trained model (rung 1): FALSIFIED on every conjunct.** In trained
   memory-only models on streams where keys are re-bound and contradicted by one-off outliers (5
   seeds per arm, 95 runs, two outlier rates), the states added +0.016 and +0.006 to accuracy on the
   deciding queries over the identical model without them, inside the noise (0.3 seed-spreads), and
   a single store with the same memory size beat the two-store design they sit on by 0.040 and
   0.094. Even a plain single store with half the memory did as well as the two-store design. So
   finding 4's small module-level gain did not survive into a trained model, and this was the test
   the phase-3 design was built for.
* **Language modelling as a pre-registered test (rung 2)** was reserved for mechanisms that rungs
  1 and 1b support, by rules written before either verdict. Neither earned it. The full account,
  with what would change these conclusions, is in
  [`docs/phase3/EXPERIMENT_PLAN.md`](docs/phase3/EXPERIMENT_PLAN.md) ("C, synthesis" and "Rung 1
  result") and in [`REPORT.md`](REPORT.md).
* **In one sentence:** the fly's sparse, winner-take-all addressing is a real capacity mechanism
  but neither fly-specific nor cheaper on a GPU, the fly-derived history states gave a small gain
  in an isolated memory that vanished in a trained model, and no fly mechanism earned a
  language-model test.

The full record: [`docs/phase3/DESIGN.md`](docs/phase3/DESIGN.md) (design and odds),
[`docs/phase3/EXPERIMENT_PLAN.md`](docs/phase3/EXPERIMENT_PLAN.md) (every result in the order it
came), [`docs/phase3/AXIOMS.md`](docs/phase3/AXIOMS.md). The phase-1 and phase-2 code, verification
reports and results are kept and remain valid.

**Does the complete fruit-fly connectome contain computational principles that improve
sequence models?** A controlled research programme: extract identified circuits from
MaleCNS v1.0 (166,700 neurons, 10.5 M edges, 101 M synaptic contacts), verify them
against published wiring, re-implement their *principles* as differentiable modules, and
test them against tuned baselines at matched compute with the full control set — shuffled
graphs, random graphs, parameter-matched and compute-matched dense layers, three seeds,
pre-registered thresholds. Negative results are reported as results.

> This is not "an LLM made from a fly brain". No model here is a fly brain, nothing here
> "learns like a fly", and no claim of state of the art is made or supported. See
> [`docs/RESEARCH_DESIGN.md`](docs/RESEARCH_DESIGN.md) for why that framing is wrong and
> what the defensible question is.

## The idea in one paragraph

A connectome is structure, not computation. Wiring only means something relative to
*neuron identity*: which cell sees what. Map fly neurons onto arbitrary hidden units of a
language model and the graph degrades to its degree statistics, which is why "connectome
as recurrent mask" is a weak prior (we measure it: H5). What can transfer is the
*algorithm* an identified circuit implements. Two fly circuits have experimentally
established functions and a clear counterpart in the 2026 sequence-model frontier:

| Circuit | Established function | ML counterpart | Hypothesis |
|---|---|---|---|
| Mushroom body: ~50 PN types → ~2,000 Kenyon cells (fan-in ≈ 5), APL feedback holds ~5 % active, KC→MBON synapses *depressed* under dopamine, dopamine neurons encode prediction error via MBON feedback | one-shot associative memory with low interference (Caron 2013; Lin 2014; Hige 2015; Hattori 2017; Bennett 2021) | test-time memorisation / fast weights (Titans, Miras, DeltaNet, Gated DeltaNet) | H1: sparse-expansion keys + error-gated delta write beat dense fast weights at matched compute |
| Central complex: EPG ring, PEN shift by angular velocity, Delta7 cosine inhibition | heading integration as a circular Kalman filter with certainty in bump amplitude (Turner-Evans 2017; Hulse 2021; Kutschireiter 2023) | state tracking, length extrapolation | H2: extracted ring matrix + learned velocity extrapolates state tracking where an equal-size RNN does not |

The unifying frame is active inference used *as an objective, not a slogan*: every
fast-weight update of the form "gradient step on prediction error plus decay toward a
prior" is a variational free-energy descent (the Miras view), and the fly's mushroom body
makes specific choices inside that design space — sparse key space, precision from a
prediction-error signal, erase-type writes, global-inhibition normalisation — that ML
has not tried together. Those choices are the hypotheses; the connectome supplies their
numbers.

## FlyAI in one paragraph (phase 2)

FlyAI is a *filter*, not a feed-forward stack. At every token it (1) predicts the token
from top-down priors (a Dale-signed, E/I-balanced tanh recurrent core with a
homeostatically regulated gain near the edge of chaos, a mushroom-body episodic memory,
and a bank of continuous attractors), (2) observes the token, (3) refines its fast latent
state with up to K gradient steps on the variational free energy (adaptive: more steps
when surprised), (4) updates the *precision* of every prediction error as a Gamma
posterior and a global arousal scalar from running surprise, (5) writes the episodic
memory with a precision-gated depression rule (the dopamine rule), and (6) at generation
time chooses tokens by expected free energy (pragmatic + epistemic value), which lets it
decide when to ask rather than guess. Every ingredient is pre-registered as a falsifiable
claim (AIF-H1 .. AIF-H6) with its ablation and matched control. The implementation was
verified against the design line by line before the AIF grids ran
([`docs/aif/VERIFICATION.md`](docs/aif/VERIFICATION.md): method, property checks, every
finding and its fix). Its planned GPU path (hybrid
3:1 decoder, matched Gated-DeltaNet twin, FineWeb-Edu training, lm-eval wrapper) is kept in
`benchmarks/` and [`docs/aif/GPU_RUNBOOK.md`](docs/aif/GPU_RUNBOOK.md); it was never run, because
the phase was closed after AIF-H1 (REPORT.md, sections 5 and 6).

## What is in the repository

```
flybrain/connectome/   readers for the MaleCNS binaries, circuit extraction + verification,
                       degree-preserving shuffle, Erdos-Renyi, degree-corrected SBM
flybrain/modules/      FastWeightMemory (linear attention / DeltaNet / Gated DeltaNet / MB
                       variants in one code path), RingAttractor (connectome / canonical /
                       unstructured), MaskedDaleRNN
flybrain/tasks/        MQAR, compass state tracking, character corpora
flybrain/harness/      frozen pre-registration, resumable seed x LR grids with provenance,
                       paired bootstrap CIs, report tables
flyai/                 phase 2: FlyAI active-inference model (primitives, memory, attractor,
                       model, EFE decoding, ask-to-learn environment, tasks)
flyprinc/              phase 3: the principle modules (memory.py, the module ladder) and the
                       system-level layer and model (hybrid.py: Gated DeltaNet-style store with
                       the per-address history states and the dense / Kenyon-cell / learned /
                       product-key addresses; lmtasks.py: the recall streams)
docs/aif/, docs/phase3/  axioms, designs, literature digests, the phase-3 experiment log
                       (EXPERIMENT_PLAN.md) and the GPU runbook (GPU_RUNBOOK.md)
docs/history/          publication note: when each pre-registration was frozen (from the development
                       history, which is not published) and the code-identity checks of the GPU grids
experiments/h*/, aif_*/, p3_*/
                       one directory per pre-registered experiment: PREREG.md (hashed at the
                       first run; edits refused afterwards), run.py, SUMMARY.md, summary.json,
                       and for the GPU grids runs_table.csv (one row per run) and figures/;
                       p3_s2_lm/, p3_k1_sparse_kernel/, p3_k2_selective_writes/ hold the
                       exploratory prototypes, the sparse kernel and the pilots (not pre-registered)
data/circuits/         the verified MB and CX sub-connectomes (small .npz, committed)
data/SHA256SUMS, data/manifests/   checksums of the fetched data and the tokenized corpora
scripts/               fetch_data.sh, fetch_wikitext103.py, fetch_tinystories.py, verify_circuits.py,
                       build_results.py, export_runs.py, plot_s1.py, plot_s1b.py, launch_detached.ps1,
                       wait_then_run.py
benchmarks/            phase-2 Tier-2 GPU scripts (hybrid LM + lm-eval-harness); never run
RESULTS.md             every pre-registered number with controls, seeds, CIs and its threshold
REPORT.md              the written account, with a mandatory "Negative results" section
LIMITATIONS.md         every assumption we supplied and everything the connectome omits
CHANGELOG.md, CITATION.cff   what each version contains; how to cite this repository and its sources
```

## Verified circuits (from the data, not assumed)

`python scripts/verify_circuits.py` re-extracts both circuits and checks them against the
literature. All checks pass on both hemispheres; the ones that matter:

* 2,019 / 2,045 Kenyon cells per hemisphere; each KC receives 5.1 PN types on average
  (Caron 2013: 5-7); one GABAergic APL per side contacting > 99.9 % of KCs; 48-49 MBONs;
  158 PAM + 8 PPL1 dopaminergic neurons; MBON→DAN feedback present (300-339 edges/side).
* 46 EPG, 20 PEN_a, 22 PEN_b, 18 PEG, 42 Delta7 (identical to hemibrain counts).
  EPG→EPG recurrence is local; EPG→Delta7 input peaks 180° away and Delta7 output is local,
  so the effective EPG→Delta7→EPG inhibition is minimal at the bump and maximal opposite it;
  left- and right-bridge PEN neurons shift the bump in opposite directions. The 8-sector
  L_i = R_(9-i) glomerulus pairing was read off the Delta7 output labels, not assumed.

## Results

See [`RESULTS.md`](RESULTS.md) (tables with all controls) and [`REPORT.md`](REPORT.md)
(interpretation, negative results, the verdict table in its section 4). Phase-1 figures are in
`results/figures/`; the rung-1b figures (accuracy against memory load, learning curves) are in
`experiments/p3_s1b_sparse/figures/`, the rung-1 figures (the primary metric by arm, accuracy per
stratum, learning curves) in `experiments/p3_s1_hybrid/figures/`. Versions: [`CHANGELOG.md`](CHANGELOG.md), [`CITATION.cff`](CITATION.cff).

## Reproduce

```bash
pip install -e ".[dev,bench]"             # plus the CUDA build of torch for the GPU experiments
bash scripts/fetch_data.sh                # raw connectome (68 MB) + small corpora; not committed; checksums in data/SHA256SUMS
python scripts/verify_circuits.py         # re-extract and verify the MB / CX circuits
python -m pytest -q                       # 129 tests: loaders, shuffle invariants, modules, the chunked layer's exactness, the harness

# phase 1 (CPU)
python experiments/h5_graph_mask/run.py   # each experiment reads its PREREG.md, freezes its hash, runs the seed x LR grid, writes SUMMARY.md
python experiments/h1_mushroom_body/run.py
python experiments/h2_ring_attractor/run.py
python scripts/make_figures.py
# phase 2 (CPU): experiments/aif_h*/run.py                 phase 3, module ladder (CPU): experiments/p3_e*/run.py, p3_d1_corpus/run.py
# phase 3, GPU grids and prototypes: every command, with the environment and the run times, in docs/phase3/GPU_RUNBOOK.md
python scripts/build_results.py           # renders RESULTS.md from every summary.json
```

Every run directory records config, seed, git commit, PREREG hash and environment; the GPU grids
also export one row per run (`runs_table.csv`) and check that the code every run executed was
identical. Phases 1-2 and the phase-3 module ladder ran on 4 CPU cores, with correspondingly small
sizes; the phase-3 grids and prototypes ran on one RTX 4080 SUPER (16 GB); see LIMITATIONS.md.

## Data and credit

MaleCNS v1.0: Berg et al., *Sexual dimorphism in the complete Drosophila male central
nervous system connectome*, Cell 2026 (FlyEM / University of Cambridge / MRC LMB / Google
Research), CC-BY-4.0, via the `garyb9/fly-drone` binary mirror whose provenance we
verified against the official Janelia release (`data/circuits/ATTRIBUTION.md`).
Synapse counts are anatomical contact counts standing in for unmeasured synaptic strengths.

## License

MIT for code. Derived connectome files in `data/circuits/` are CC-BY-4.0.
