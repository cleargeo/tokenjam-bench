"""Judged benchmark — QA / RAG / summarization tasks scored by a Judge.

These are the workloads with no executable test: the score comes from a judge
(DeepEval for real, MockJudge offline), then flows into the same proof stats as
everything else. Offline, each task embeds `# echo:` so the deterministic mock
model returns the expected answer (a weaker candidate returns a wrong string,
which the judge then fails) — exercising the judge seam with no keys.

Select the judge backend via env (default offline mock):
  TJBENCH_JUDGE=deepeval TJBENCH_JUDGE_METRIC=correctness tjbench run \
      --benchmark judged --original anthropic:claude-opus-4-7
"""
from __future__ import annotations

from benchmarks.base import ScoreResult, Task
from judge import Judge, JudgeCase, judge_from_env

# (id, question, expected answer, optional retrieval context)
_CASES = [
    ("judged/refund-policy",
     "What is the refund window?",
     "The refund window is 30 days from purchase.",
     ["Refunds are accepted within 30 days of the original purchase date."]),
    ("judged/capital",
     "What is the capital of France?",
     "The capital of France is Paris.", None),
    ("judged/retry-summary",
     "Summarize: the agent retried five times and then failed the task.",
     "The agent failed the task after five retries.", None),
    ("judged/shipping",
     "How long does standard shipping take?",
     "Standard shipping takes 5 to 7 business days.",
     ["Standard shipping is delivered in 5-7 business days; express is 2 days."]),
    ("judged/define-llm",
     "Define a large language model in one sentence.",
     "A large language model predicts the next token of text from prior tokens.",
     None),
]


def _build_tasks() -> list[Task]:
    tasks: list[Task] = []
    for tid, question, expected, context in _CASES:
        prompt = f"{question}\n# task_key: {tid}\n# echo: {expected}\n"
        tasks.append(Task(
            task_id=tid, prompt=prompt, kind="judged",
            metadata={"question": question, "expected": expected, "context": context},
        ))
    return tasks


class JudgedBenchmark:
    name = "judged"

    def __init__(self, judge: Judge | None = None) -> None:
        # Default judge from env (offline MockJudge unless TJBENCH_JUDGE=deepeval).
        self._judge = judge or judge_from_env()
        self._tasks = _build_tasks()

    def tasks(self, limit: int | None = None) -> list[Task]:
        return self._tasks if limit is None else self._tasks[:limit]

    def score(self, task: Task, completion_text: str) -> ScoreResult:
        case = JudgeCase(
            input=task.metadata.get("question", task.prompt),
            actual_output=completion_text,
            expected_output=task.metadata.get("expected"),
            context=task.metadata.get("context"),
        )
        r = self._judge.evaluate(case)
        return ScoreResult(
            r.passed,
            f"{self._judge.name}:{r.metric} score={r.score:.2f} (>= {r.threshold}) — {r.reason[:80]}",
        )
