"""Agent execution: the multi-turn AgentRunner, tools, trace, and validation.

This package is the keystone that turns the bench from an LLM benchmark into an
agent benchmark. It feeds the SAME proof machinery (stats + cost) as the
single-shot path — a trace yields a per-task pass/fail and a measured
multi-turn cost.
"""
from __future__ import annotations

from agents.runner import AgentRunner
from agents.tools import Tool, ToolRegistry, ToolResult
from agents.trace import AgentTrace, ToolCallRecord, TurnRecord
from agents.validation import ToolValidation, validate_tools

__all__ = [
    "AgentRunner",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "AgentTrace",
    "TurnRecord",
    "ToolCallRecord",
    "ToolValidation",
    "validate_tools",
]
