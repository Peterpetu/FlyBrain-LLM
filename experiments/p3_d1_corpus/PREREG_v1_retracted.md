# PREREG — P3-D1: does text contain re-binding beyond the attention window?

*Corpus diagnostic, no model, minutes on one core. Frozen at the first run. This is the design
panel's highest-valued item: the phase-3 memory principles only matter on streams where the same
address is written more than once, sometimes inconsistently, at distances a sliding-window
attention layer does not cover. If text does not contain that regime, the language-model rung is
aimed at nothing and must be re-scoped before any GPU time is spent.*

**Proxy.** Without a trained model, an "address" is a key-like token: a word-level token that is
capitalised or rarer than 1 in 10,000 (entities, names, definitions), and its "value" is its
continuation (the next 2 tokens). A repeat is a second occurrence of the same key within a window
W of the first; the repeat is CONSISTENT if the continuation is the same and a RE-BINDING if it
differs. The gap is the token distance between the two occurrences.

**Corpora.** WikiText-2 train (10.8 MB) and Penn Treebank train (5.1 MB), already in
`data/text/`, word-level; contiguous windows W in {512, 2048, 8192}.

**Reported.** Per corpus and window: keys per 1,000 tokens; repeats per 1,000 tokens; the fraction
of repeats that are re-bindings; re-bindings per 1,000 tokens whose gap exceeds 512 tokens (the
sliding-window size of the planned hybrid); the gap distribution of re-bindings (median, 90th
percentile); the same for a control key set (random tokens of matched frequency) to show how much
of "re-binding" is just continuation variability.

**Decision rule (pre-registered).** The language-model rung of phase 3 is planned at 2k-8k context
only if re-bindings beyond a 512-token gap occur at more than 1 per 2,048 tokens in at least one
corpus at W = 8192 AND the re-binding fraction for key-like tokens exceeds that of the frequency-
matched control by more than 5 points. Otherwise the language rung is scoped to synthetic, code and
agent streams, and that scoping is stated in the abstract of any report.

**Caveat, stated in advance.** Continuation variability is a crude proxy for contradiction; a
kNN-LM probe over a trained model would be better and is the planned follow-up once a model exists.
The numbers here bound the regime, they do not measure it exactly.
