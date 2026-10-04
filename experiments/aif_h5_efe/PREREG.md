# PREREG — AIF-H5: expected-free-energy decoding decides when to ask

**Hypothesis.** A decoder that scores candidate actions by expected free energy
(pragmatic value = model log-probability; epistemic value = free-energy drop that
observing the candidate would produce) asks a clarifying question when its latent state
is uncertain and answers directly when it is not, even though the training transcripts
contain no information about when asking is useful (asking is random in the data).

**Environment.** Ask-to-learn (flyai/env_ask.py): 6 key-value pairs, then a query key that
was shown with probability 0.5; the agent may emit '?' (the environment then reveals the
value) or answer immediately. Training transcripts: random asking with p = 0.5, answers
always correct. Evaluation by simulation on 512 fresh episodes per seed: accuracy, ask
rate on known vs unknown queries, and reward = accuracy - 0.3 x ask rate.

**Model.** FlyAI, K = 1, n = 128, memory 512 KCs, trained 300 steps (batch 32, LR 3e-3
fixed: this experiment is about the decoder, not the trainer), seeds {0, 1, 2}. The SAME
trained model is decoded with `greedy` (beta_e = 0) and `efe` (beta_e in {1, 3, 10}); the
best beta_e is chosen on seed 0 and applied to seeds 1, 2 (reported separately).

**Cheap competitor (required control).** An entropy-threshold policy: ask iff the entropy
of the model's own answer distribution exceeds a threshold (thresholds {0.5, 1, 1.5, 2}
nats, chosen on seed 0 like beta_e). This is the obvious non-active-inference heuristic
and the literature's standard control for epistemic decoders.

**Thresholds.** Supported only if, for the beta_e chosen on seed 0, the EFE decoder on
seeds 1-2 has (a) ask rate on unknown queries minus ask rate on known queries > 0.2,
(b) reward higher than greedy by > 2 seed-SD, and (c) reward at least as high as the
entropy-threshold policy (within seed spread). A decoder that asks more everywhere (no
known/unknown separation) does not count; matching the entropy heuristic without beating
it is reported as "epistemic value reduces to output entropy here".


## v0.2 change log (before the first run)

Model is FlyAI v0.2; the EFE machinery is unchanged (free-energy-drop proxy for
information gain on the categorical head). The entropy-threshold control stands.

## v0.2.2 change log (formal verification pass, before the first run)

Implementation fixes found by verifying the code against docs/aif/DESIGN.md (report:
docs/aif/VERIFICATION.md). None changes the hypothesis, conditions, budget, seeds, learning
rates, metrics or thresholds registered above.
* K is exactly 1 (`adaptive_K=False`), as registered.
* The entropy-threshold control now reads the entropy of the model's ANSWER distribution (the value-vocabulary slice of the prediction at the answer position, after a hypothetical '='); the first version read the value slice of the prediction for the next transcript token, which is not the answer distribution.
* The epistemic term uses the same sensory precision as the model's own inference (arousal included).
* Thresholds (a)-(c) are evaluated mechanically on the held-out seeds and printed with the verdict; a seed whose training diverges is recorded and excluded; every seed directory records config, commit, PREREG hash and torch version.
