# PREREG — AIF-H3: global arousal from running surprise (precision hyper-prior)

**Hypothesis.** A single neuromodulatory scalar gamma_t = 2 * sigmoid(w log F_bar_t + c),
driven by the running free energy and multiplying all precisions, improves prediction on
character streams with topic switches (segments from Shakespeare, PTB and WikiText-2
concatenated) relative to the same model with gamma fixed at 1, and the gain is
concentrated in the tokens right after a switch.

**Task.** TopicSwitchStream (flyai/tasks.py): windows of 96 characters containing 1-3
segment boundaries; shared character vocabulary. Validation: 4 fixed seeded batches of 64
windows. Metrics: overall validation bits/char (primary); bits/char on the 16 characters
following each switch (secondary, descriptive).

**Model.** FlyAI, K = 1, n = 128, memory 512 KCs, attractor bank on. Conditions:
`arousal` (hypothesis), `no_arousal` (gamma = 1).

**Budget.** 300 AdamW steps, batch 32, LRs {1e-3, 3e-3, 1e-2}, seeds {0, 1, 2}.

**Thresholds.** Supported only if `arousal` beats `no_arousal` on overall bits/char by
> 2 seed-SD with CI excluding 0. Post-switch bits/char is reported but is not a threshold.


## v0.2 change log (before the first run)

In v0 gamma multiplied every term of F and cancelled in the preconditioned step
(nearly inert by construction; the design panel's finding). In v0.2 gamma_t is a bounded
(0.25, 4) function of the running token surprise and acts only as a relative gain: on the
sensory precision pi_0 (evidence vs prior) and on the memory write gate. Ladder:
`arousal` (both sites; hypothesis), `arousal_pi0`, `arousal_beta`, `arousal_learned`
(an input-driven learned scalar at both sites: does the surprise-driven signal beat a free
gate?), `no_arousal`. Threshold: supported only if `arousal` beats `no_arousal` AND
`arousal_learned` by > 2 seed-SD with CIs excluding 0; the single-site arms are
descriptive attribution.

## v0.2.2 change log (formal verification pass, before the first run)

Implementation fixes found by verifying the code against docs/aif/DESIGN.md (report:
docs/aif/VERIFICATION.md). None changes the hypothesis, conditions, budget, seeds, learning
rates, metrics or thresholds registered above.
* K is exactly 1 (`adaptive_K=False`), as registered.
* Precisions are bounded ratios ([1e-3, 1e3]) everywhere they are read; a diverged run enters the tables as NaN.
* The Lyapunov exponent is estimated in every condition; the gain controller is the dithered extremum-seeking design (docs/aif/VERIFICATION.md).
