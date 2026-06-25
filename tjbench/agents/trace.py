"""The observable record of an agent run.

A single-shot completion has nothing to validate beyond its text. A multi-turn
agent run produces a *trace*: which tools were called, with what arguments, in
what order, whether they errored, the per-turn token usage, and the final
answer. The trace is what makes tool-call validation, multi-turn evaluation, and
side-effect safety possible — and summing tokens across turns is what makes the
cost number honest for agents (a model that loops 8× costs 8×).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tjbench.models.base import Completion


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict
    result: str
    is_error: bool


@dataclass
class TurnRecord:
    index: int
    assistant_text: str
    tool_calls: list[ToolCallRecord]
    input_tokens: int
    output_tokens: int
    cache_tokens: int = 0


@dataclass
class AgentTrace:
    task_id: str
    turns: list[TurnRecord] = field(default_factory=list)
    final_text: str = ""
    stopped_reason: str = "final"        # "final" | "max_turns"

    def tool_sequence(self) -> list[str]:
        """Ordered tool names across the whole run."""
        return [tc.name for turn in self.turns for tc in turn.tool_calls]

    def all_tool_calls(self) -> list[ToolCallRecord]:
        return [tc for turn in self.turns for tc in turn.tool_calls]

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def total_input_tokens(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_cache_tokens(self) -> int:
        return sum(t.cache_tokens for t in self.turns)

    def as_completion(self) -> Completion:
        """Collapse the run's token usage so the existing pricing path applies."""
        return Completion(
            text=self.final_text,
            input_tokens=self.total_input_tokens,
            output_tokens=self.total_output_tokens,
            cache_tokens=self.total_cache_tokens,
        )
