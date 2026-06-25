"""End-to-end pipeline, fully offline (mock models + built-in samples).

Proves the plumbing: TokenJam picks the candidate, TokenJam prices the cost, the
result is version-stamped, and accuracy is measured by executing the tasks.
"""
from __future__ import annotations

import json

from tjbench.pipeline import resolve_candidate, run_proof


def test_candidate_comes_from_tokenjam():
    # TokenJam's downgrade map sends opus → haiku.
    assert resolve_candidate("anthropic:claude-opus-4-7") == "anthropic:claude-haiku-4-5"


def test_no_candidate_raises_without_override():
    # A model TokenJam has no downgrade for must require an explicit candidate.
    try:
        run_proof(benchmark_name="samples", original_spec="anthropic:claude-haiku-4-5", mock=True)
    except ValueError as exc:
        assert "no downgrade candidate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for un-downgradeable model")


def test_preserved_accuracy_and_cheaper():
    # Candidate at full accuracy → same pass-rate, lower cost.
    result = run_proof(
        benchmark_name="samples",
        original_spec="anthropic:claude-opus-4-7",
        mock=True,
        mock_candidate_accuracy=1.0,
    )
    assert result.candidate_model == "anthropic:claude-haiku-4-5"
    assert result.recommended_by == "tokenjam.DOWNGRADE_CANDIDATES"
    assert result.n_tasks == 5
    assert result.original_pass == 5            # canned solutions all pass
    assert result.candidate_pass == 5
    assert result.accuracy_delta_pp == 0.0
    assert result.cost_delta_pct < 0            # haiku cheaper than opus (tokenjam pricing)
    assert result.tokenjam_version              # stamped
    assert result.mock is True


def test_regression_is_detected():
    # Candidate that fails everything → -100pp, all tasks flagged as regressions.
    result = run_proof(
        benchmark_name="samples",
        original_spec="anthropic:claude-opus-4-7",
        mock=True,
        mock_candidate_accuracy=0.0,
    )
    assert result.candidate_pass == 0
    assert result.accuracy_delta_pp == -100.0
    assert result.regressions == result.original_pass


def test_explicit_candidate_override():
    result = run_proof(
        benchmark_name="samples",
        original_spec="openai:gpt-4o",
        candidate_spec="openai:gpt-4o-mini",
        mock=True,
        mock_candidate_accuracy=1.0,
    )
    assert result.candidate_model == "openai:gpt-4o-mini"
    assert result.recommended_by == "explicit --candidate override"


def test_stats_block_is_attached():
    result = run_proof(
        benchmark_name="samples",
        original_spec="anthropic:claude-opus-4-7",
        mock=True,
        mock_candidate_accuracy=0.0,   # total wipeout
    )
    s = result.stats
    assert s.samples_per_task == 1
    assert len(s.original_ci_pp) == 2 and len(s.delta_ci_pp) == 2
    # b = tasks the downgrade broke (all 5), c = 0.
    assert s.mcnemar_b == result.original_pass
    assert s.mcnemar_c == 0
    # n=5 < MIN_TASKS_FOR_VERDICT → honest 'insufficient_evidence', NOT a
    # claim of regression, even on a total wipeout (small-n humility).
    assert s.verdict == "insufficient_evidence"


def test_samples_k_runs_each_task_k_times():
    result = run_proof(
        benchmark_name="samples",
        original_spec="anthropic:claude-opus-4-7",
        mock=True,
        samples=3,
        mock_candidate_accuracy=1.0,
    )
    assert result.stats.samples_per_task == 3
    for t in result.tasks:
        assert t.samples == 3
        assert 0 <= t.original_passes <= 3
    assert result.stats.original_pass_at_1 == 1.0   # deterministic mock, full accuracy


def test_artifact_is_written_and_version_stamped(tmp_path):
    result = run_proof(
        benchmark_name="samples",
        original_spec="anthropic:claude-opus-4-7",
        mock=True,
    )
    path = result.write(tmp_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["tokenjam_version"] == result.tokenjam_version
    assert f"tj{result.tokenjam_version}" in path.name   # version in filename
    assert data["headline"]
