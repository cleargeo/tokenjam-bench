"""Agent benchmark abstractions.

Unlike an executable code benchmark (score a text completion), an agent
benchmark scores an AgentTrace: the final answer AND the tool usage. A task can
fail on a forbidden action even when the answer text is correct — that's the
whole point of validating agents rather than completions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from tjbench.agents.tools import ToolRegistry
from tjbench.agents.trace import AgentTrace
from tjbench.benchmarks.base import ScoreResult


@dataclass
class AgentTask:
    task_id: str
    prompt: str
    expected_answer: str | None = None        # substring / numeric match in final text
    expected_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    expected_order: list[str] | None = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class AgentBenchmark(Protocol):
    name: str

    def tools(self) -> ToolRegistry: ...

    def tasks(self, limit: int | None = ...) -> list[AgentTask]: ...

    def score(self, task: AgentTask, trace: AgentTrace) -> ScoreResult: ...
