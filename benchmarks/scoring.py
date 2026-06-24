"""Shared scoring helpers: code execution and exact-match.

Both produce a boolean pass/fail — the objective ground truth that makes a
result a 'proof' rather than a judgment.
"""
from __future__ import annotations

import re

from benchmarks.base import ScoreResult, Task
from exec_sandbox import run_python

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_code(text: str) -> str:
    """Pull code out of a model completion, unwrapping a ``` fence if present."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1)
    return text


def score_code(task: Task, completion_text: str, timeout_s: float = 10.0) -> ScoreResult:
    """Insert the model's code into the task's test template and execute it."""
    if not task.test_program_template:
        return ScoreResult(False, "task has no test_program_template")
    code = extract_code(completion_text)
    program = task.test_program_template.replace("{completion}", code)
    result = run_python(program, timeout_s=timeout_s)
    return ScoreResult(result.passed, result.detail)


def _final_number(text: str) -> str | None:
    """GSM8K-style answer extraction: prefer the value after '####', else the
    last number in the text."""
    if "####" in text:
        tail = text.split("####")[-1]
        m = _NUMBER_RE.search(tail)
        if m:
            return _normalize_number(m.group(0))
    matches = _NUMBER_RE.findall(text)
    return _normalize_number(matches[-1]) if matches else None


def _normalize_number(s: str) -> str:
    """Canonicalize so '8', '8.0', '08' compare equal."""
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s.strip()


def score_exact_match(task: Task, completion_text: str) -> ScoreResult:
    if task.expected is None:
        return ScoreResult(False, "task has no expected answer")
    got = _final_number(completion_text)
    want = _normalize_number(task.expected)
    if got == want:
        return ScoreResult(True, f"matched {want}")
    return ScoreResult(False, f"expected {want}, got {got}")
