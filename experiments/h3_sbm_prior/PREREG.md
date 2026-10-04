# PREREG — H3: cell-type stochastic block model as a scalable wiring prior

**Gate.** Per the programme, H3 only earns full compute if H5 shows the graph prior
surviving scale. Independently of that gate, this experiment establishes (i) that the
dcSBM sampler reproduces the core graph's statistics and (ii) a first functional test
at two scales against Erdos-Renyi.

**Hypothesis.** A recurrent mask sampled from a degree-corrected SBM fitted to the
cell-type x cell-type graph of the 4,096-neuron core (blocks = cell types, sign fraction
per block, degree heterogeneity per block) beats an Erdos-Renyi mask with the same N and
E as a character-LM recurrent mask, at N in {512, 1024}.

**Conditions per N.** `sbm` (Dale signs from the block sign fractions), `er` (same N, E,
same sign vector), `real` for reference (top-N core; N=512, 1024 rows come from H5's
runs when present).

**Budget.** 400 steps, TinyShakespeare, seq 128, batch 32, seeds {0, 1, 2}, LRs
{1e-3, 3e-3, 1e-2} for both (both are "hypothesis-class" masks; neither is a tuned
dense baseline). Metric: validation bits/char (first 100k chars).

**Threshold.** Supported only if `sbm` beats `er` by > 2 seed-SD with CI excluding 0 at
both sizes. Otherwise the block structure carries no usable prior at this scale.
