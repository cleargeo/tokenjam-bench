"""Benchmark + scoring abstractions.

A `Benchmark` yields `Task`s and scores a model's completion against objective
ground truth — executable tests (code) or exact-match (numeric). `score()`
returns pass/fail, so accuracy is a *measurement*, never a judgment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class Task:
    task_id: str
    prompt: str                       # full prompt sent to the model
    kind: str                         # "code" | "exact_match"
    # code tasks:
    test_program_template: str | None = None  # {completion} placeholder
    # exact-match tasks:
    expected: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoreResult:
    passed: bool
    detail: str


@runtime_checkable
class Benchmark(Protocol):
    name: str

    def tasks(self, limit: int | None = ...) -> list[Task]: ...

    def score(self, task: Task, completion_text: str) -> ScoreResult: ...
