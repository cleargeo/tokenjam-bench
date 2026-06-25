"""HumanEval — executable code benchmark (objective pass/fail ground truth).

Loads the canonical HumanEval problems via the `datasets` library
(`tokenjam-bench[datasets]`). Each problem ships a hidden `check()` test; a
completion passes iff that check runs without raising. We ask the model to
return the full function definition, then run `{completion}` followed by the
problem's test harness in the sandbox.
"""
from __future__ import annotations

from tjbench.benchmarks.base import ScoreResult, Task
from tjbench.benchmarks.scoring import score_code

_INSTRUCTION = (
    "Complete the following Python function. Return ONLY the full function "
    "definition (you may use a ```python code block), no prose.\n\n"
)


class HumanEvalBenchmark:
    name = "humaneval"

    def __init__(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "HumanEval needs the datasets extra. Run "
                "`pip install 'tokenjam-bench[datasets]'`."
            ) from exc
        # Namespaced repo id (required by datasets >= 3); fall back to the
        # legacy bare name on older datasets versions.
        try:
            self._ds = load_dataset("openai/openai_humaneval", split="test")
        except (ValueError, FileNotFoundError):
            self._ds = load_dataset("openai_humaneval", split="test")

    def tasks(self, limit: int | None = None) -> list[Task]:
        rows = self._ds if limit is None else self._ds.select(range(min(limit, len(self._ds))))
        out: list[Task] = []
        for row in rows:
            entry_point = row["entry_point"]
            template = (
                "{completion}\n\n"
                + row["test"]
                + f"\n\ncheck({entry_point})\n"
            )
            out.append(Task(
                task_id=row["task_id"],
                prompt=_INSTRUCTION + row["prompt"],
                kind="code",
                test_program_template=template,
                metadata={"entry_point": entry_point},
            ))
        return out

    def score(self, task: Task, completion_text: str) -> ScoreResult:
        return score_code(task, completion_text)
