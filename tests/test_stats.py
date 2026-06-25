"""Statistical primitives — the layer that turns a pass-rate delta into a claim."""
from __future__ import annotations

from tjbench.stats import (
    mcnemar_exact,
    paired_delta_ci,
    pass_at_k,
    wilson_interval,
)


def test_wilson_interval_brackets_point_estimate():
    iv = wilson_interval(46, 50)  # 92%
    assert iv.low < 0.92 < iv.high
    assert 0.0 < iv.low and iv.high < 1.0


def test_wilson_perfect_score_is_not_a_point():
    iv = wilson_interval(50, 50)  # 100% observed, but CI is not [1,1]
    assert iv.low < 1.0
    assert iv.low > 0.9


def test_wilson_zero_n_is_fully_uncertain():
    iv = wilson_interval(0, 0)
    assert iv.low == 0.0 and iv.high == 1.0


def test_mcnemar_significant_when_discordance_is_lopsided():
    r = mcnemar_exact(b=10, c=2)
    assert r.p_value < 0.05
    assert r.discordant == 12


def test_mcnemar_not_significant_when_balanced():
    assert mcnemar_exact(b=3, c=2).p_value == 1.0


def test_mcnemar_no_discordance_is_p1():
    assert mcnemar_exact(b=0, c=0).p_value == 1.0


def test_small_n_cannot_reach_significance():
    # 5/5 -> 0/5 is a total wipeout, yet with only 5 paired obs McNemar
    # cannot reach p<0.05 (p = 2*0.5^5 = 0.0625). This is THE point of the
    # significance layer: small samples can't prove a regression either way.
    assert mcnemar_exact(b=5, c=0).p_value > 0.05


def test_paired_delta_ci_spans_the_point_estimate():
    ci = paired_delta_ci(b=2, c=0, n_tasks=10).as_pp()
    assert ci.low < -20.0 < ci.high   # Δ = (0-2)/10 = -20pp, inside the interval


def test_pass_at_k_estimator():
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 5, 3) == 1.0
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-9
    assert pass_at_k(5, 1, 5) == 1.0     # n-c < k → guaranteed at least one
