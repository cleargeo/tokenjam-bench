"""Tool-calling model interface.

`complete()` (single-shot text) can't drive an agent — the model has to be able
to *request* a tool call and then continue after seeing the result. A
`ToolCallingClient.chat()` takes the running conversation plus the advertised
tools and returns an `AssistantTurn`: either final text, or one-or-more tool
calls to execute and feed back.

The message format is provider-neutral; each concrete client translates it.
  - {"role": "user", "content": str}
  - {"role": "assistant", "content": str, "tool_calls": [ToolCall...]}
  - {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class ToolCallingClient(Protocol):
    provider: str
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> AssistantTurn: ...
