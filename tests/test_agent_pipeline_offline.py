"""Agent proof pipeline end-to-end (offline) — proves it reuses the SAME stats
machinery as the single-shot path, and that the safety gate flows through to a
measurable pass-rate regression."""
from __future__ import annotations

from agent_pipeline import run_agent_proof


def test_ok_candidate_matches_original():
    result = run_agent_proof(
        benchmark_name="sample-agent",
        original_spec="anthropic:claude-opus-4-7",
        mock=True, candidate_behavior="ok",
    )
    assert result.candidate_model == "anthropic:claude-haiku-4-5"
    assert result.n_tasks == 3
    assert result.original_pass == 3
    assert result.candidate_pass == 3
    assert result.accuracy_delta_pp == 0.0
    assert result.cost_delta_pct < 0          # haiku cheaper, priced via TokenJam
    # Same statistical block as the single-shot path.
    assert result.stats.mcnemar_p_value == 1.0
    assert len(result.stats.delta_ci_pp) == 2


def test_unsafe_candidate_regresses_via_safety_gate():
    result = run_agent_proof(
        benchmark_name="sample-agent",
        original_spec="anthropic:claude-opus-4-7",
        mock=True, candidate_behavior="unsafe",
    )
    # Every task's candidate calls a dangerous tool → fails the safety gate,
    # so the cheaper model's pass-rate collapses even though its answers are right.
    assert result.original_pass == 3
    assert result.candidate_pass == 0
    assert result.accuracy_delta_pp == -100.0
    assert result.stats.mcnemar_b == 3        # all three broken by the downgrade


def test_token_and_cost_are_summed_over_turns():
    result = run_agent_proof(
        benchmark_name="sample-agent",
        original_spec="anthropic:claude-opus-4-7",
        mock=True, candidate_behavior="ok",
    )
    # Multi-turn runs accumulate output tokens > a single completion's worth.
    assert result.original_output_tokens > 0
    assert result.original_cost_usd > 0
