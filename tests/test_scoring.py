"""Scorers must give objective pass/fail — this is the 'proof' substrate."""
from __future__ import annotations

from tjbench.benchmarks.base import Task
from tjbench.benchmarks.scoring import (
    extract_code,
    score_code,
    score_exact_match,
)


def test_extract_code_unwraps_fence():
    text = "Here:\n```python\ndef f():\n    return 1\n```\nDone."
    assert "def f():" in extract_code(text)
    assert "```" not in extract_code(text)


def test_score_code_pass_and_fail():
    task = Task(
        task_id="t", prompt="", kind="code",
        test_program_template="{completion}\nassert add(2, 3) == 5\n",
    )
    assert score_code(task, "def add(a, b):\n    return a + b\n").passed
    assert not score_code(task, "def add(a, b):\n    return a - b\n").passed


def test_score_code_handles_timeout():
    task = Task(
        task_id="t", prompt="", kind="code",
        test_program_template="{completion}\nspin()\n",
    )
    bad = "def spin():\n    while True:\n        pass\n"
    res = score_code(task, bad)  # uses default 10s timeout
    assert not res.passed


def test_exact_match_extracts_final_number():
    task = Task(task_id="t", prompt="", kind="exact_match", expected="8")
    assert score_exact_match(task, "blah blah\n#### 8").passed
    assert score_exact_match(task, "the answer is 8").passed       # last number
    assert not score_exact_match(task, "#### 9").passed
    assert score_exact_match(task, "#### 8.0").passed              # normalized
