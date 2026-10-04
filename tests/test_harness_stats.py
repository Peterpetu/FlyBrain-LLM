"""flybrain/harness/stats.py: NaN-safe best-LR selection and the sign convention of compare(); report.py's tables."""
import re

import numpy as np
from flybrain.harness.report import analyse, markdown_table
from flybrain.harness.stats import best_lr, compare


def _res(rows):
    return [dict(condition="c", lr=lr, seed=s, m=v) for lr, s, v in rows]


def test_best_lr_never_picks_a_diverged_lr():
    r = _res([(1e-3, 0, float("nan")), (1e-3, 1, 0.5), (3e-3, 0, 0.4), (3e-3, 1, 0.45)])
    assert best_lr(r, "c", "m", higher_is_better=False)[0] == 3e-3
    assert best_lr(r, "c", "m", higher_is_better=True)[0] == 3e-3
    ok = _res([(1e-3, 0, 0.9), (3e-3, 0, 0.1)])
    assert best_lr(ok, "c", "m", True)[0] == 1e-3 and best_lr(ok, "c", "m", False)[0] == 3e-3


def test_compare_effect_sign_follows_higher_is_better():
    hyp, base = [0.9, 0.92, 0.91], [0.8, 0.82, 0.81]
    c = compare(hyp, base, higher_is_better=True); assert c["effect"] > 0 and c["ci_excludes_0"]
    c = compare(hyp, base, higher_is_better=False); assert c["effect"] < 0          # lower-is-better: hypothesis is worse
    c = compare(base, hyp, higher_is_better=False); assert c["effect"] > 0 and c["effect_in_sd"] > 2


def test_markdown_rows_have_the_header_s_columns_when_a_condition_name_contains_a_pipe():
    """Grid keys such as "state|r0.15" are condition names; unescaped, the pipe split the cell and shifted every column."""
    res = [dict(condition=c, lr=1e-3, seed=s, m=v + 0.01 * s, x=v) for c, v in (("state|r0.15", 0.8), ("free|r0.15", 0.7)) for s in range(3)]
    md = markdown_table(analyse(res, "m", "state|r0.15", ["free|r0.15"], extra_metrics=["x"]), "m", ["x"]).splitlines()
    cells = lambda line: len(re.split(r"(?<!\\)\|", line.strip())) - 2              # separators not preceded by a backslash
    assert all(cells(l) == cells(md[0]) for l in md), md
    assert md[2].startswith("| free\\|r0.15 |") or md[3].startswith("| free\\|r0.15 |")
