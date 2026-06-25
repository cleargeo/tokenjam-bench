"""Tools an agent can call, and the registry that executes them.

A `Tool` carries a JSON-schema for its arguments (so it can be advertised to a
tool-calling model) and a `dangerous` flag. The flag is load-bearing for the
safety story from the review: a cheaper model can produce a correct-looking
final answer while taking a catastrophic action (delete vs read). Marking the
tool dangerous lets validation fail the task on the *action*, not the text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON Schema for the arguments object
    run: Callable[[dict[str, Any]], ToolResult]
    dangerous: bool = False             # has side effects worth flagging on misuse

    def spec(self) -> dict[str, Any]:
        """Provider-agnostic advertisement of this tool to a model."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def dangerous_names(self) -> set[str]:
        return {n for n, t in self._tools.items() if t.dangerous}

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(output=f"unknown tool: {name}", is_error=True)
        try:
            return tool.run(arguments or {})
        except Exception as exc:  # a tool raising is a tool error, not a crash
            return ToolResult(output=f"tool error: {exc}", is_error=True)
