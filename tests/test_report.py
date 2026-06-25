"""Cost-validation logic: a cheaper model that inflates output tokens erodes
the per-token advantage — the report must surface it."""
from __future__ import annotations

from tjbench.report import ProofResult, ProofStats


def _stats() -> ProofStats:
    return ProofStats(
        samples_per_task=1, original_ci_pp=[80.0, 99.0], candidate_ci_pp=[78.0, 98.0],
        delta_ci_pp=[-5.0, 3.0], mcnemar_b=1, mcnemar_c=1, mcnemar_p_value=1.0,
        significant=False, alpha=0.05, verdict="no_significant_regression",
        original_pass_at_1=0.9, candidate_pass_at_1=0.88,
    )


def _result(orig_out: int, cand_out: int) -> ProofResult:
    return ProofResult(
        tokenjam_version="0.0.0", tokenjam_location=None, benchmark="samples",
        original_model="anthropic:claude-opus-4-7",
        candidate_model="anthropic:claude-haiku-4-5",
        recommended_by="test", n_tasks=20, original_pass=18, candidate_pass=17,
        original_cost_usd=1.0, candidate_cost_usd=0.4,
        original_output_tokens=orig_out, candidate_output_tokens=cand_out,
        mock=False, priced_with_defaults=False, stats=_stats(),
    )


def test_token_inflation_flag_trips_when_candidate_is_verbose():
    r = _result(orig_out=1000, cand_out=2000)   # 2x output
    assert r.output_token_inflation == 2.0
    assert r.token_inflation_flag is True


def test_no_inflation_flag_when_comparable():
    r = _result(orig_out=1000, cand_out=1050)
    assert r.token_inflation_flag is False


def test_headline_carries_ci_and_verdict():
    head = _result(1000, 1000).headline()
    assert "95% CI" in head
    assert "McNemar p=" in head
    assert "no_significant_regression" in head
