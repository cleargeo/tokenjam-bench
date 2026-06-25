"""Statistical primitives for turning raw pass/fail counts into defensible
claims. Pure math, zero dependencies (no scipy/numpy).

The pieces, and why each one:
  - wilson_interval: a pass rate of 46/50 is a point estimate; the Wilson
    interval is its 95% CI. Reporting "92%" without it is the exact sin we
    nailed TokenJam for (a number with no error bar).
  - mcnemar_exact: original vs candidate are scored on the SAME tasks, so the
    correct significance test for the pass-rate difference is McNemar's paired
    test on the discordant pairs — not an unpaired two-proportion test.
  - paired_delta_ci: the CI on the accuracy delta itself, using the paired
    (discordant-pair) variance so it's consistent with McNemar.
  - pass_at_k: the unbiased estimator (Chen et al., 2021) so multi-sample runs
    report pass@k honestly rather than "ran it once, it passed".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959963984540054  # standard normal quantile for a two-sided 95% interval


@dataclass(frozen=True)
class Interval:
    low: float
    high: float

    def as_pp(self) -> "Interval":
        """Express a proportion interval in percentage points."""
        return Interval(round(self.low * 100, 2), round(self.high * 100, 2))


def wilson_interval(successes: int, n: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion. n == 0 → fully uncertain."""
    if n <= 0:
        return Interval(0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return Interval(max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class McNemarResult:
    b: int                  # original PASS, candidate FAIL (downgrade broke it)
    c: int                  # original FAIL, candidate PASS (downgrade fixed it)
    statistic: float        # continuity-corrected chi-square (reference)
    p_value: float          # exact two-sided binomial p-value (primary)
    discordant: int         # b + c


def mcnemar_exact(b: int, c: int) -> McNemarResult:
    """Exact two-sided McNemar test on discordant counts.

    Under H0 (the two models are equivalent) the b discordant pairs are
    Binomial(b+c, 0.5). The exact two-sided p-value is the two-tailed binomial
    probability. b+c == 0 → no discordance → p = 1.0 (no evidence of a
    difference).
    """
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, 1.0, 0)
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p_value = min(1.0, 2.0 * tail)
    statistic = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    return McNemarResult(b, c, round(statistic, 4), round(p_value, 4), n)


def paired_delta_ci(b: int, c: int, n_tasks: int, z: float = Z_95) -> Interval:
    """95% CI on the candidate−original pass-rate difference, paired.

    Δ = (c − b)/N (concordant pairs cancel). Variance uses the paired
    (discordant-pair) form so it agrees with McNemar.
    """
    if n_tasks <= 0:
        return Interval(0.0, 0.0)
    delta = (c - b) / n_tasks
    var = ((b + c) - (c - b) ** 2 / n_tasks) / (n_tasks ** 2)
    se = math.sqrt(var) if var > 0 else 0.0
    return Interval(delta - z * se, delta + z * se)


def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    Given n_samples draws of which n_correct passed, the probability that at
    least one of k random draws passes: 1 − C(n−c, k)/C(n, k).
    """
    if k <= 0 or n_samples <= 0:
        return 0.0
    if n_samples - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n_samples - n_correct, k) / math.comb(n_samples, k)
