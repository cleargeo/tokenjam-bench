"""Judge layer — the DeepEval seam, exercised offline via the MockJudge."""
from __future__ import annotations

import pytest

from judge import JudgeCase, MockJudge, get_judge
from pipeline import run_proof


def test_mock_judge_passes_matching_output():
    j = MockJudge(threshold=0.5)
    r = j.evaluate(JudgeCase(input="q", actual_output="Paris is the capital of France",
                             expected_output="The capital of France is Paris"))
    assert r.passed and r.score > 0.5


def test_mock_judge_fails_unrelated_output():
    j = MockJudge(threshold=0.5)
    r = j.evaluate(JudgeCase(input="q", actual_output="totally unrelated text",
                             expected_output="The capital of France is Paris"))
    assert not r.passed


def test_mock_judge_uses_context_when_no_expected():
    j = MockJudge(threshold=0.3)
    r = j.evaluate(JudgeCase(input="q", actual_output="shipping takes 5-7 business days",
                             context=["Standard shipping is delivered in 5-7 business days."]))
    assert r.passed


def test_get_judge_validates_backend_and_metric():
    assert get_judge("mock").name == "mock"
    with pytest.raises(ValueError):
        get_judge("mock", metric="nonsense")
    with pytest.raises(ValueError):
        get_judge("does-not-exist")


def test_get_judge_deepeval_is_lazy_and_keygated():
    # Constructing the DeepEval judge must not require the dep; only evaluate() /
    # _build_metric() touches it. So this should succeed even without deepeval.
    j = get_judge("deepeval", metric="faithfulness")
    assert j.name == "deepeval" and j.metric == "faithfulness"


def test_judged_benchmark_runs_through_the_proof_pipeline_offline():
    # ok candidate echoes the expected answer → judge passes all.
    ok = run_proof(benchmark_name="judged", original_spec="anthropic:claude-opus-4-7",
                   mock=True, mock_candidate_accuracy=1.0)
    assert ok.n_tasks == 5
    assert ok.original_pass == 5 and ok.candidate_pass == 5
    assert ok.cost_delta_pct < 0

    # wrong candidate returns unrelated text → judge fails it.
    bad = run_proof(benchmark_name="judged", original_spec="anthropic:claude-opus-4-7",
                    mock=True, mock_candidate_accuracy=0.0)
    assert bad.original_pass == 5 and bad.candidate_pass == 0
