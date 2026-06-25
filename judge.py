"""Judge layer — semantic scoring for tasks with no executable ground truth.

Executable benchmarks give pass/fail by running tests. Many real workloads
(summaries, RAG answers, research) have no test to run — you need a *judge*.
This is the seam your boss asked for via DeepEval: a `Judge` turns a
(input, output, expected, context) case into a pass/fail + score, which feeds
the SAME Wilson/McNemar stats as everything else.

`MockJudge` is a deterministic, offline judge (token overlap) so the plumbing
and tests run with no keys. `DeepEvalJudge` (in deepeval_judge.py) is the real
one — DeepEval metrics, key-gated. NOT a quality proxy: the mock exists to
exercise the seam, real judging needs a real judge model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9]+")

# Metrics the judge layer understands (DeepEval-backed when run for real).
JUDGE_METRICS = ("correctness", "answer-relevancy", "faithfulness", "task-completion")


@dataclass
class JudgeCase:
    input: str                          # the task / question given to the model
    actual_output: str                  # the model's output to be judged
    expected_output: str | None = None  # gold answer (for correctness)
    context: list[str] | None = None    # retrieval context (for faithfulness)


@dataclass
class JudgeResult:
    metric: str
    score: float        # 0..1
    threshold: float
    passed: bool
    reason: str


@runtime_checkable
class Judge(Protocol):
    name: str
    metric: str
    threshold: float

    def evaluate(self, case: JudgeCase) -> JudgeResult: ...


def _overlap(a: str, b: str) -> float:
    """Jaccard token overlap of two strings in [0,1]."""
    sa, sb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class MockJudge:
    """Deterministic offline judge. Scores token overlap of the output against
    the reference (expected answer, else the first context chunk, else input)."""
    name = "mock"

    def __init__(self, metric: str = "correctness", threshold: float = 0.5) -> None:
        self.metric = metric
        self.threshold = threshold

    def evaluate(self, case: JudgeCase) -> JudgeResult:
        ref = case.expected_output or (case.context[0] if case.context else case.input)
        score = round(_overlap(case.actual_output, ref), 4)
        return JudgeResult(
            metric=self.metric, score=score, threshold=self.threshold,
            passed=score >= self.threshold,
            reason=f"mock token-overlap {score:.2f} vs reference (offline; not a quality proxy)",
        )


# Judge backends. 'mock' is offline; the DeepEval-backed ones name the
# OpenAI-compatible provider that supplies the judge model. 'deepeval' is a
# back-compat alias for 'openai'.
_JUDGE_PROVIDER = {"deepeval": "openai", "openai": "openai", "deepseek": "deepseek"}


def get_judge(backend: str = "mock", metric: str = "correctness",
              threshold: float = 0.5, model: str | None = None) -> Judge:
    """Resolve a judge backend. 'mock' is offline; 'openai'/'deepseek'/'deepeval'
    are DeepEval-backed and key-gated."""
    if metric not in JUDGE_METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Available: {JUDGE_METRICS}")
    if backend == "mock":
        return MockJudge(metric=metric, threshold=threshold)
    if backend in _JUDGE_PROVIDER:
        from deepeval_judge import DeepEvalJudge
        return DeepEvalJudge(metric=metric, threshold=threshold, model=model,
                             provider=_JUDGE_PROVIDER[backend])
    raise ValueError(
        f"Unknown judge backend '{backend}'. Use 'mock', 'openai', or 'deepseek'."
    )


def judge_from_env() -> Judge:
    """Build the judge from TJBENCH_JUDGE* env vars (default: offline mock).

      TJBENCH_JUDGE=mock|openai|deepseek   TJBENCH_JUDGE_METRIC=correctness
      TJBENCH_JUDGE_THRESHOLD=0.5          TJBENCH_JUDGE_MODEL=<override>
    """
    return get_judge(
        backend=os.environ.get("TJBENCH_JUDGE", "mock"),
        metric=os.environ.get("TJBENCH_JUDGE_METRIC", "correctness"),
        threshold=float(os.environ.get("TJBENCH_JUDGE_THRESHOLD", "0.5")),
        model=os.environ.get("TJBENCH_JUDGE_MODEL"),
    )
