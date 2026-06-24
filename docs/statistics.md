# Statistics

## Overview

The `stats.py` module provides pure-math statistical primitives for evaluating proof results. No scipy or numpy — all implemented with standard library math.

## Methods

### Wilson Interval

```python
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.
    
    Returns (lower, upper) confidence bounds for the true pass rate
    given k successes out of n trials.
    """
```

Used to compute confidence intervals for original and candidate pass rates.

**Properties verified by tests:**
- CI always brackets the point estimate (k/n)
- Perfect score (k=n) still has CI < 1 (not a point)
- Zero trials (n=0) returns [0, 1] (fully uncertain)

### McNemar Exact Test

```python
def mcnemar_exact(b: int, c: int) -> tuple[float, bool]:
    """Exact two-sided McNemar test on discordant pairs.
    
    b = original passes, candidate fails
    c = original fails, candidate passes
    
    Returns (p_value, is_significant_at_0.05).
    """
```

Tests whether the difference between original and candidate pass rates is statistically significant.

**Properties verified by tests:**
- Lopsided discordance (e.g., b=5, c=0) → significant
- Balanced discordance (e.g., b=2, c=2) → not significant
- No discordance (b=0, c=0) → p=1.0
- Small n cannot reach significance (e.g., n=5 total wipeout cannot reach p<0.05)

### Paired Delta CI

```python
def paired_delta_ci(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Confidence interval on the paired pass-rate delta.
    
    delta = (candidate_pass_rate - original_pass_rate)
    """
```

Computes the confidence interval for the difference in pass rates between original and candidate.

**Properties verified by tests:**
- CI always spans the point estimate
- Symmetric around the observed delta for large n

### Pass@k Estimator

```python
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021).
    
    n = total tasks
    c = number of tasks with at least one pass in k samples
    k = samples per task
    """
```

Estimates the probability that at least one of k independent samples passes, given the observed data.

## How Statistics Are Used in Proofs

```
ProofResult
├── original_pass_rate: k_orig / n
│   └── wilson_interval(k_orig, n) → [lower, upper]
├── candidate_pass_rate: k_cand / n
│   └── wilson_interval(k_cand, n) → [lower, upper]
├── mcnemar_exact(b, c)
│   └── b = original passes, candidate fails
│   └── c = original fails, candidate passes
│   └── p_value, significant?
├── paired_delta_ci(b, c, n)
│   └── delta = candidate_rate - original_rate
│   └── [delta_lower, delta_upper]
└── verdict
    └── Based on: n, significant?, delta magnitude
```

## Verdict Logic

| Condition | Verdict | Meaning |
|-----------|---------|---------|
| n < 30 and not significant | `insufficient_evidence` | Sample too small for confidence |
| Significant, delta ≈ 0 | `preserved` | Cheaper model is just as accurate |
| Significant, delta < -5pp | `regression_detected` | Cheaper model is worse |
| Significant, delta > +5pp | `improved` | Cheaper model is better (rare) |
| Not significant, |delta| < 5pp | `likely_preserved` | Probably fine, but not proven |

## Honesty Discipline

- **Small samples are flagged**: n < 30 → `insufficient_evidence` regardless of observed rates
- **No p-hacking**: Fixed alpha = 0.05, no multiple comparison correction needed (single test)
- **CIs are reported**: Always show Wilson intervals, not just point estimates
- **Effect size matters**: A "significant" p-value with tiny delta is still flagged as `preserved`

## Related Documentation

- [Pipelines](pipelines.md) — How stats are used in assemble_proof
- [Report](api-reference.md#reportpy) — ProofResult, ProofStats
- [TokenJam's Evaluation Subsystem](https://github.com/HoomanDigital/tokenjam/blob/main/tokenjam/core/eval/) — TokenJam's own quality evaluation
