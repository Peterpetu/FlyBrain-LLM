# PREREG — P3-E0: is the candidate state actually unavailable to the control?

*Module level, no training, numpy only (minutes on one core). Frozen at the first run. This is the
red team's "K-minus-1" (docs/phase3/DESIGN_PANEL.md), hardened and seeded; it runs BEFORE any gate
is trained, because a gate can only use information that is in its inputs.*

**Question.** Phase 3 proposes to give a learned write gate new per-address STATES rather than new
rules. Candidates: (a) contradiction mass `a`, a non-negative accumulator of |written residual| at
the address (ATTEST's evidence channel, d_z = 1, with the crosstalk floor rho * sigma, where sigma is
the total accumulated mass); (b) a familiarity-gated contradiction mass `zc`, the same accumulator
counting only writes whose observation disagrees (cosine <= 0.5) with what a familiar address
already stores; (c) a content-based confirmation count `nconf` (this repository's two-trace
mechanism: a vector accumulated along the observed value, reset on contradiction, read along the
stored value's own direction); (d) last-touch age (realised on sparse codes as the median age of the
address's active rows; on dense codes there is no row set, so the per-address oracle age is used,
the most charitable treatment). The control gate already sees the delta rule's own intermediates:
the error e_t and the read vhat_t = S^T k_t, hence ||e||_1, ||vhat||_1, ||vhat||_2 and
cos(v_t, vhat_t), and a MesaNet-style key covariance k^T M k with M = sum beta k k^T. Which of the
candidate states carry information about the address's teaching history that the control's
quantities do not?

**Stream (per seed).** 3 x 32 addresses in classes {NOVEL: no prior write; CLEAN: two prior
consistent writes (v, v); CONTRADICTED: two prior inconsistent writes (v, v')}, plus 96 one-shot
distractor writes, in a random interleaving that preserves each address's own event order, into a
gated-delta store S <- S + beta k (v - S^T k)^T (decay 1) with write rate beta in
{0.25, 0.5, 0.8, 1.0}. The PROBE event at every address is a write of a fresh random value: the same
current event for every class, so that only the address's history can separate the classes (the
gate's adoption-versus-outlier decision). Two contradiction modes: re-binding (v' random) and exact
cancellation (v' = -v). Key codes: SPARSE non-negative 32-of-2048 (unit norm), DENSE d = 128
L2-normalised isotropic, and DENSE anisotropic (component scale i^-0.5 and i^-1). Values: unit-norm
Gaussian, d_v = 16. Seeds 0..7. Note that at this load every row of the sparse code is shared by
about five addresses; that is the regime the per-row scalar states must survive.

**Quantities at each address's probe event, before the probe write is applied.**
Control set: ||e||_1, ||vhat||_1, ||vhat||_2, cos(v, vhat), k^T M k. Candidate states: `a` =
relu(Z . k - rho sigma) with rho = k/n for sparse codes and 1/sqrt(d) for dense (the designs' stated
initialisation), plus an oracle rho swept over 60 values in [0, 1] with the per-address floor
rho * sigma (the most charitable, label-aware treatment); `zc`; `nconf`; log age.

**Metrics.** For every scalar: pairwise AUROCs novel-vs-contradicted and clean-vs-contradicted
(CvC; the novel class is trivially separable by ||vhat||, so CvC is the informative contrast).
3-way accuracy of a multinomial logistic regression on the control set (log-magnitudes, standardised),
fit on the even seeds and scored on the odd seeds; the same regression with each candidate state
added (incremental accuracy = the information the state adds), with all states added, and on the
states alone. Dead fraction of `a` (addresses with a = 0). H_B test: ||S^T k|| after +v, -v under
the true gated-delta recurrence vs gated linear attention.

**Pre-registered kill conditions (each evaluated per key code, mode and beta; any one that fires is
reported as such):**
- K1: no single candidate state adds more than 0.02 3-way accuracy over the control logistic
  -> in that configuration the states add nothing a learned gate does not already have.
- K2: on DENSE keys the contradiction-mass CvC AUROC at the stated floor is below 0.7 -> the dense
  variant of the evidence channel is dead; only sparse non-negative addresses can carry it.
- K3: the dead fraction of `a` on dense keys exceeds 0.5 at the stated floor.
- K4: ||S^T k|| after H_B exceeds 0.1 ||v|| under the gated delta rule for a beta in [0.25, 1]
  -> the exact-cancellation "separation proposition" does not cover the control.

**Decision rule for E2.** A candidate state enters E2's `learned_gate_state` input set only if its
incremental 3-way accuracy over the control logistic is > 0.05 on sparse keys (held-out seeds),
averaged over beta and mode. Everything measured here is reported whatever it says.
