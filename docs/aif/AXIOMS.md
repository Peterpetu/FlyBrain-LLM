# Active-inference FlyLM: axioms and first-principles derivation (draft, pre-literature)

*Written before the literature sweep returned, so that the sweep can be judged against a
position rather than replace it. Revised sections are marked.*

## 0. Why the previous architecture is "too close to ML"

The previous model took a standard decoder and swapped one mixer. Its fly content was a
frozen matrix, three numbers and a write rule. Everything that makes a brain a brain was
missing: inference is *iterative* (a state settles), it is *precision-weighted* (the same
error means different things in different contexts), it is *hierarchical with separated
timescales*, it is *homeostatic* (activity is kept in a working range), it runs near a
*critical point* (small changes can matter, but do not blow up), and it *acts* (the
output changes the input). None of those is a matrix. They are dynamical principles, and
they are exactly what active inference formalises.

## 1. Axioms

**AX1 (Generative model).** The system is a generative model of its token stream:
p(o_{1:T}, s_{1:T}) = prod_t p(o_t | s_t) p(s_t | s_{<t}), with s_t a *hierarchy* of
latent states s_t = (s^1_t, ..., s^L_t) ordered by timescale, level l predicting level
l-1 and level 0 being the tokens.

**AX2 (Perception = free-energy descent in state space).** At each token, beliefs about
s_t are updated by minimising variational free energy
F_t = sum_l pi^l * || e^l_t ||^2 / 2 - log-precision terms + KL(q(s_t) || p(s_t | s_{<t})),
where e^l_t = s^{l-1}_t - g^l(s^l_t) is the prediction error of level l about level l-1,
and pi^l its precision. This is iterative: K gradient steps on s_t per token, not one
feed-forward pass. The number of steps is itself decided by the free energy (stop when
it stops falling; spend more on surprising tokens).

**AX3 (Precision is inferred, not learned once).** pi^l_t is a belief, updated online
from the recent squared errors (a Gamma posterior: pi = (nu + n) / (nu * pi_0^{-1} +
sum e^2)). Attention, gating and "dopamine" are all the same object: precision.
Neuromodulators set the *hyper-priors* on precision (arousal = high global precision).

**AX4 (Learning = the same descent on slow variables).** Parameters change by descending
the same free energy, slowly. Fast weights (memories written within a context) are the
intermediate timescale: descent on F with respect to a memory matrix, gated by precision.
This is the mushroom-body rule. Backprop through the unrolled dynamics is an admissible
implementation of the slow descent; it is not the point.

**AX5 (Action = expected free energy).** Emitting a token is an action. The policy that
active inference prescribes minimises expected free energy: it prefers outcomes it
predicts (pragmatic value) and outcomes that resolve uncertainty about its own latent
states (epistemic value). For a language model this is a *decoding* principle with a
computable epistemic term (mutual information between the next token and the latent
state under the model's own posterior).

**AX6 (Homeostasis and criticality).** Every recurrent population keeps its activity in a
fixed working range by divisive feedback inhibition (the fly's APL and antennal-lobe
normalisation), and the recurrent gain is regulated so that the largest Lyapunov exponent
of the state dynamics sits near zero (edge of chaos): perturbations neither die nor
explode, which maximises the memory the dynamics can carry. Excitatory/inhibitory balance
with ~30 % inhibitory units (the fly's ratio) is the biological implementation.

**AX7 (Modularity by cell type).** The hierarchy is made of a small number of typed
populations with fixed connectivity *patterns* (which population talks to which, with
what sign, at what fan-in) and learned *strengths*, as in the fly's 11,770 stereotyped
types. The connectome supplies the patterns for the three circuits whose function is
known (antennal lobe -> mushroom body -> MBON/DAN loop; central-complex attractor); the
rest is supplied by AX1-AX6.

## 2. What the fly maps to, level by level

| AIF object | Fly circuit | Established function | Our module |
|---|---|---|---|
| sensory likelihood, input normalisation | antennal lobe with lateral inhibition | divisive normalisation of PN activity (Olsen & Wilson 2010) | divisive-normalisation input layer |
| fast episodic memory, precision-gated writes | PN -> KC -> MBON with APL and DANs | one-shot association; dopamine = prediction error via MBON->DAN feedback | sparse-code fast-weight memory with *inferred* precision gate |
| slow belief state with certainty | EPG/PEN/Delta7 ring | circular Kalman filter; amplitude = certainty | continuous-attractor latent state (generalised from a ring to K attractor dims) |
| innate priors over outcomes | lateral horn | fixed valence | prior term in the decoder |
| global precision / arousal | octopamine, serotonin, DAN state | gain control by internal state | neuromodulator unit driven by running surprise |
| action selection | descending neurons | winner-take-all among behaviours | expected-free-energy decoding |
| homeostasis, sparsity | APL, GABAergic feedback | keeps ~5 % KCs active | divisive normalisation in every population |
| criticality | E/I balance, 30.8 % inhibitory | (inferred) stable-yet-sensitive dynamics | Lyapunov-regulated gain, Dale-signed recurrent cores |

## 3. Falsifiable claims that follow (pre-registered later)

- AIF-H1: K-step free-energy descent on the latent state per token (adaptive K) beats
  the same network with K = 1 at matched *average* compute.
- AIF-H2: inferred (Gamma-posterior) precision gating of fast-weight writes beats a
  learned input-dependent gate on recall under non-stationarity (key-value pairs that
  change mid-context).
- AIF-H3: a global arousal signal from running surprise, setting precision hyper-priors,
  improves loss on concatenated documents with topic switches vs fixed precision.
- AIF-H4: a recurrent core regulated to Lyapunov ~ 0 with Dale-signed E/I balance
  retains information over longer contexts than the same core left unregulated or tuned
  contractive.
- AIF-H5: expected-free-energy decoding beats greedy / nucleus sampling on a task where a
  clarifying (epistemic) token is worth emitting before answering.

## 4. Non-claims

No state of the art. No "fly brain". No claim about scale beyond what is run. The
connectome contributes patterns and numbers; every dynamical assumption is ours and is
listed in LIMITATIONS.md.
