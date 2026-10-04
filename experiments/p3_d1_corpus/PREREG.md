# PREREG — P3-D1 (v2): does text contain persistent re-binding beyond the attention window?

*Corpus diagnostic, no model, seconds on one core. Frozen at the first v2 run. This is the design
panel's highest-valued item: the phase-3 memory principles only matter on streams where the same
address is written more than once, sometimes inconsistently, at distances a sliding-window attention
layer does not cover. If text does not contain that regime, the language-model rung is aimed at
nothing and must be re-scoped before any GPU time is spent.*

**Why v2 (instrument defect in v1, recorded in `PREREG_v1_retracted.md` / `SUMMARY_v1_retracted.md`).**
v1 defined a key-like token as "capitalised or rarer than 1 in 10,000" and its value as the next two
tokens. Both halves failed their own sanity checks on the first run: 36% of all tokens were "keys"
(the frequency cut admits most of the vocabulary under Zipf's law), 92-97% of ALL repeats were
"re-bindings" for keys and for the control alike (a two-token continuation is almost never repeated
verbatim, so the proxy measured continuation variability, not contradiction), and the frequency-
matched control was empty on the lower-cased PTB (every rare token was a key, so no matched non-key
existed), which made the pre-registered "exceeds control by 5 points" clause fire vacuously. The v1
verdict is therefore void and is not used. v2 keeps the rate threshold of v1 (1 per 2,048 tokens)
and replaces the ill-defined control with a permutation null. Nothing in v2 was chosen by looking at
v2 results; v2 is frozen before it runs.

**Proxy.** Word-level tokens. Two key strata: `cap` = alphabetic tokens that begin with a capital
letter, occur at least 3 times, and whose lower-cased form is rarer than the capitalised form (so
sentence-initial common words are excluded; WikiText-2 only, PTB is lower-cased); `rare` =
alphabetic tokens with 3 to 100 occurrences (both corpora). Value = the next token. For a key with
occurrences p_1 < ... < p_m and next tokens c_1..c_m, consecutive pair (j-1, j) is a REPEAT when
p_j - p_{j-1} <= W; CONSISTENT when c_j = c_{j-1}; a RE-BINDING otherwise; a PERSISTENT RE-BINDING
when additionally c_{j+1} = c_j with p_{j+1} - p_j <= W (the new binding is used again, i.e. the
address really changed rather than merely varied). "Far" = gap p_j - p_{j-1} > 512, the sliding
window of the planned hybrid.

**Corpora.** WikiText-2 train and Penn Treebank train (already in `data/text/`), windows
W in {512, 2048, 8192}.

**Null.** Within each key type the next tokens are permuted across that key's occurrences
(positions fixed; 20 seeded permutations). Under exchangeability the multiset of continuations is
unchanged, so any excess of persistent re-bindings over the null measures temporal clustering of
bindings (a binding changes and then holds), which is the regime a mutable per-address memory serves
and a context-window copy does not. Reported: null mean, SD, z = (observed - mean) / SD, and the
observed / null ratio, for persistent re-bindings and for consistent repeats.

**Reported.** Per corpus, stratum and window: keys per 1,000 tokens; repeats per 1,000; consistent
fraction; persistent re-bindings per 1,000; far persistent re-bindings per 1,000; gap median and
90th percentile of persistent re-bindings; the null statistics above.

**Decision rule (pre-registered).** The language-model rung of phase 3 is planned at 2k-8k context
only if, in at least one corpus and stratum at W = 8192, far persistent re-bindings occur at more
than 1 per 2,048 tokens (0.488 per 1,000) AND the persistent re-binding count exceeds the permutation
null by z > 3. Otherwise the language rung is scoped to synthetic, code and agent streams, and that
scoping is stated in the abstract of any report.

**Caveat, stated in advance.** A next-token binding is a crude proxy for "the fact stored at an
address"; a kNN-LM probe over a trained model would be better and is the planned follow-up once a
model exists. The numbers here bound the regime, they do not measure it exactly.
