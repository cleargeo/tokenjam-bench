"""Built-in offline sample benchmark.

A handful of tiny code + arithmetic tasks with real test suites, so the whole
pipeline (model → score → cost → report → version stamp) runs end-to-end with
no dataset download and no network. NOT a serious accuracy benchmark — it's the
smoke/plumbing benchmark. Real proofs use `humaneval` / `gsm8k`.

The prompt embeds a `# task_key:` marker the offline MockClient reads to return
a canned solution; live models simply ignore that comment line.
"""
from __future__ import annotations

from benchmarks.base import ScoreResult, Task
from benchmarks.scoring import score_code, score_exact_match

_CODE_TASKS = [
    (
        "samples/add",
        "Write a Python function `add(a, b)` that returns the sum of a and b.",
        "{completion}\nassert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
    ),
    (
        "samples/is_even",
        "Write a Python function `is_even(n)` that returns True iff n is even.",
        "{completion}\nassert is_even(4) is True\nassert is_even(7) is False\n",
    ),
    (
        "samples/factorial",
        "Write a Python function `factorial(n)` returning n! (with factorial(0) == 1).",
        "{completion}\nassert factorial(0) == 1\nassert factorial(5) == 120\n",
    ),
]

_MATH_TASKS = [
    (
        "samples/gsm-apples",
        "Sam has 3 apples and buys 5 more. How many apples does he have? "
        "End your answer with '#### <number>'.",
        "8",
    ),
    (
        "samples/gsm-double",
        "A number is 7. What is double that number? End with '#### <number>'.",
        "14",
    ),
]


class SampleBenchmark:
    name = "samples"

    def tasks(self, limit: int | None = None) -> list[Task]:
        out: list[Task] = []
        for tid, prompt, template in _CODE_TASKS:
            out.append(Task(
                task_id=tid,
                prompt=f"# task_key: {tid}\n{prompt}",
                kind="code",
                test_program_template=template,
            ))
        for tid, prompt, expected in _MATH_TASKS:
            out.append(Task(
                task_id=tid,
                prompt=f"# task_key: {tid}\n{prompt}",
                kind="exact_match",
                expected=expected,
            ))
        return out if limit is None else out[:limit]

    def score(self, task: Task, completion_text: str) -> ScoreResult:
        if task.kind == "code":
            return score_code(task, completion_text)
        return score_exact_match(task, completion_text)
