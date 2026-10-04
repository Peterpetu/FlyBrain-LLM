# PREREG — P3-E4: E3's nested test at the training budget that separates the arms, on fresh seeds

*Module level. Frozen at the first run. Written after E3 (null at 80 steps) and after the
EXPLORATORY long-training variant of E3 (`experiments/p3_e3_two_trace/SUMMARY_long800.md`: 800
steps, learning rate 3e-3, seeds 0-2), which is quoted here as the motivation and is NOT evidence:
on that variant `tt_learned_state` reached 0.6993 class-balanced against 0.6870 for the same
architecture with the slots zeroed (+0.0123) and 0.6924 for the state-matched single store
(+0.0070), the gain sitting on outlier keys (0.62 vs 0.51 / 0.59) at a cost on re-bound keys
(0.49 vs 0.53 / 0.51). E4 is the pre-registered confirmation of that observation on seeds it has
never seen, with the single learning rate that was best for every learned arm in E3's sweep.*

**Hypothesis K0-TT (primary), as in E3.** On the two-trace substrate a learned readout and learned
per-trace gates given the per-address states (nconf, zc, a) beat the same architecture with the
slots zeroed AND the state-matched single store, at 800 training steps.

**Everything else as E3** (`experiments/p3_e3_two_trace/PREREG.md`): task, four regimes, held-out
streams, the arms `tt_learned_state`, `tt_learned_free`, `single_free_wide` and the fixed
`two_trace` reference, the class-balanced primary metric `acc_classes`, the natural mix as
secondary. Differences: 800 Adam steps; learning rate 3e-3 only (pre-selected); seeds {5..9}; the
`learned_free` reference dropped (E3 showed it equal to the wide single store).

**Thresholds (exhaustive).** SUPPORTED iff `tt_learned_state` beats `tt_learned_free` on
`acc_classes` by > 2 seed-SD with the bootstrap CI excluding 0 AND beats `single_free_wide` by
> 2 seed-SD with CI excluding 0; otherwise FALSIFIED. The size of any effect is reported against the
oracle ceiling (0.781 class-balanced; `scripts/e2_oracle_ceiling.py`): an effect of about one point
on a nine-point gap is a small effect and is described as such. The substrate clause and the
per-class trade-off are reported whatever the verdict. Positive verdicts are replicated with
`--variant replication --seeds 10 11 12 13 14 15` before being reported as supported.

**Cost.** 3 trained arms x 5 seeds x 800 steps plus the fixed reference: about 6 CPU-hours at 4 threads.
