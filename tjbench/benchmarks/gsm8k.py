"""GSM8K — grade-school math word problems (exact-match ground truth).

Loads GSM8K via `datasets` (`tokenjam-bench[datasets]`). The reference answer
ends with '#### <number>'; a completion passes iff its final number matches.
"""
from __future__ import annotations

from tjbench.benchmarks.base import ScoreResult, Task
from tjbench.benchmarks.scoring import _normalize_number, score_exact_match

_INSTRUCTION = (
    "Solve the problem step by step, then end your answer with "
    "'#### <number>' on its own line.\n\n"
)


class GSM8KBenchmark:
    name = "gsm8k"

    def __init__(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "GSM8K needs the datasets extra. Run "
                "`pip install 'tokenjam-bench[datasets]'`."
            ) from exc
        try:
            self._ds = load_dataset("openai/gsm8k", "main", split="test")
        except (ValueError, FileNotFoundError):
            self._ds = load_dataset("gsm8k", "main", split="test")

    def tasks(self, limit: int | None = None) -> list[Task]:
        rows = self._ds if limit is None else self._ds.select(range(min(limit, len(self._ds))))
        out: list[Task] = []
        for i, row in enumerate(rows):
            answer = row["answer"]
            expected = _normalize_number(answer.split("####")[-1]) if "####" in answer else ""
            out.append(Task(
                task_id=f"gsm8k/{i}",
                prompt=_INSTRUCTION + row["question"],
                kind="exact_match",
                expected=expected,
            ))
        return out

    def score(self, task: Task, completion_text: str) -> ScoreResult:
        return score_exact_match(task, completion_text)
