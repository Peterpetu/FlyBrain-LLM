# Phase 3 axioms

*These replace docs/aif/AXIOMS.md for the third architecture. Each biological axiom (A) is stated as
biology and then as the engineering constraint it imposes; each methodological axiom (M) is stated
with the null that taught it. docs/phase3/REFLECTION.md records how they were reached;
docs/phase3/DESIGN.md is the design built on them.*

## Biological axioms

**A1. Evolution is the pretraining; the lifetime is the test.** The genome instantiates a competent
network; within a lifetime the fly learns by local, neuromodulator-gated plasticity and consolidates
during sleep. *Constraint:* backpropagation on a corpus plays the role of evolution and is not
unbiological at that level; what is missing from current models, and where a fly-aligned design can
add something, is the lifetime learning system: local fast plasticity with modulated rates,
compartments across timescales, offline consolidation.

**A2. Fast learning is local and three-factor.** Kenyon-cell to MBON synapses change as a product of
presynaptic activity, an eligibility term and a dopamine signal that encodes prediction error and
valence. *Constraint:* every within-context update of the memory is a local rule with a modulated
rate; the rate is a quantity inferred from prediction errors, not a constant.

**A3. Plasticity leaves a physical record of teaching.** Dopamine-driven plasticity at KC-to-MBON
synapses is depression from a non-negative ceiling, so a compartment's weight vector is a monotone
record of the teaching delivered at each address, and the fly reads that record with a dedicated
compartment (novelty). *Constraint:* a memory may carry per-address STATES that record history
(how many times confirmed, how often contradicted, how much written), read at the same address as
the content. Whether the fly uses those states to gate plasticity elsewhere is open and is the
hypothesis under test, not an axiom.

**A4. Memory is sparse, high-dimensional and compartmentalised.** Random expansion to ~2,000 cells
at ~5% coding level gives near-orthogonal addresses; compartments forget at distinct rates; labile
and consolidated memory are two stages. *Constraint:* sparse addresses are a candidate substrate
(measured: 0.90 vs 0.25 recall at four times the dense rank, docs/phase3/MEMORY_CAPACITY.md);
timescales are explicit and architectural; consolidation is a separate hypothesis.

**A5. The learning rate is inferred.** Dopamine, octopamine and internal state set how much a
surprise changes a synapse; a stable world lowers the rate, a changed world raises it.
*Constraint:* precision-weighted updates enter as one gate input (measured: matches a change-point
oracle, beats a tuned fixed rate by a point, docs/phase3/ADAPTIVE_RATE.md); never the thesis.

**A6. Compute is bounded and sparse.** Five percent of the memory is active per event.
*Constraint:* per-token memory cost O(k d) with k << n for the sparse variant; the dense variant
must stay within +15% of its twin's MACs and state.

**A7. Normalisation is transmitted, not discarded.** Antennal-lobe divisive normalisation and the
APL feedback keep the code's overlap invariant to input scale, and the normaliser itself is a signal
(Fairhall). *Constraint:* divisive normalisation before the address; the crosstalk floor of any
scalar accumulator is subtracted (rho sigma, rho = k/n); the normaliser's log-state is a gate input.

**A8. The connectome is a sample, not a blueprint.** Left and right copies of a cell type share only
52-68% of type-level edges; PN-to-KC convergence is random; phase 1 measured a whole-graph mask
indistinguishable from its shuffle. *Constraint:* no adjacency is read anywhere. What is taken from
the connectome is the motif (one shared sparse code, many readouts, at least one reporting history)
and the anatomy that grounds A3 and A7 (verified extraction: 2,019 KCs, 48 MBONs, 158 PAM + 8 PPL1
dopaminergic neurons, one APL covering 100% of KCs, docs/CIRCUITS.md).

**A9. Measure the claim.** The claim, if any, is recall under contradiction and adaptation beyond the
attention window at matched compute against test-time-learning controls; perplexity per FLOP is a
regression check, never the claim.

## Methodological axioms (each learned from a null in this repository)

**M1. Propose a state, not a gate.** *From:* H1b (learned gate beat the fly-derived error gate by
2.6 seed-SD); AIF-H1 (+0.7 seed-SD for K = 3 inference steps). A fixed rule that a learned gate with
the same inputs contains as a special case cannot win; only a per-address state the gate cannot
reconstruct can.

**M2. The control is the same code with the new inputs zeroed.** *From:* AIF-H1's "wide" control
with 1.94x the parameters and 1.32x the FLOPs, which made the comparison uninterpretable in both
directions.

**M3. Signal availability before optimisation.** *From:* the red team's finding that ATTEST's proof
did not cover its control and that ||vhat|| recovers most of the claimed signal for free; confirmed
by E0 (at beta < 1 without noise the control is at 0.92-1.00).

**M4. Regime existence before scale.** *From:* H1 at 8 key-value pairs against a store whose ceiling
was 65, where no capacity difference could appear; confirmed necessary by D1 (the regime exists in
text, as a minority: about three far persistent re-bindings per 2,048 tokens).

**M5. No middle band, and instruments are validated before use.** *From:* phase 2's "partially
supported" band; D1 v1's vacuous verdict from a defective proxy and an empty control, retracted on
its first run and kept in the record.
