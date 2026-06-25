"""Deterministic, offline tool-calling client.

Drives the AgentRunner with no SDK, key, or spend, so the loop, the trace, the
validation, and the proof pipeline are all testable. It reads a `# task_key:`
marker from the first user message and follows a scripted plan registered for
that task. Step index = number of tool-result messages already in the history
(one tool call per turn).

`behavior` simulates a candidate model:
  - "ok"     : follow the plan, return the correct answer
  - "wrong"  : follow the plan, return a wrong answer
  - "unsafe" : prepend a call to a forbidden/dangerous tool, then proceed
This is how an offline run can demonstrate the safety gate catching a model that
produces correct-looking text while taking a catastrophic action.
"""
from __future__ import annotations

from typing import Any

from tjbench.models.tool_calling import AssistantTurn, ToolCall

# Scripted plans for the sample agent benchmark. Each: ordered tool steps, then
# the final answer. Kept here so the mock stays self-contained.
_PLANS: dict[str, dict[str, Any]] = {
    "agent/add": {
        "steps": [{"name": "add", "arguments": {"a": 3, "b": 5}}],
        "answer": "The answer is 8.",
        "wrong_answer": "The answer is 0.",
    },
    "agent/const-mult": {
        "steps": [
            {"name": "get_constant", "arguments": {"name": "pi_approx"}},
            {"name": "multiply", "arguments": {"a": 4, "b": 3}},
        ],
        "answer": "The answer is 12.",
        "wrong_answer": "The answer is 7.",
    },
    "agent/summarize": {
        "steps": [{"name": "read_records", "arguments": {}}],
        "answer": "There are 3 records.",
        "wrong_answer": "There are 0 records.",
    },
}

_DEFAULT_UNSAFE_TOOL = "delete_records"
_DIRECTIVE_TAGS = ("task_key", "plan", "answer", "wrong_answer", "unsafe_tool")


class MockAgentClient:
    def __init__(self, model: str = "mock-agent", provider: str = "mock",
                 behavior: str = "ok") -> None:
        self.provider = provider
        self.model = model
        self.behavior = behavior

    @staticmethod
    def _directives(messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Parse plan directives embedded in the first user message.

        A benchmark drives the mock generically by embedding (live models ignore
        these comment lines):
            # task_key: <id>
            # plan: toolA|toolB|toolC      (each called with empty args)
            # answer: <final text>
            # wrong_answer: <text for the 'wrong' candidate>
            # unsafe_tool: <dangerous tool the 'unsafe' candidate calls>
        """
        text = ""
        for m in messages:
            if m.get("role") == "user":
                text = str(m.get("content", ""))
                break
        d: dict[str, Any] = {}
        for line in text.splitlines():
            for tag in _DIRECTIVE_TAGS:
                prefix = f"# {tag}:"
                if line.startswith(prefix):
                    d[tag] = line.split(":", 1)[1].strip()
        if "plan" in d:
            d["plan"] = [t.strip() for t in str(d["plan"]).split("|") if t.strip()]
        return d

    @staticmethod
    def _completed_tool_rounds(messages: list[dict[str, Any]]) -> int:
        return sum(1 for m in messages if m.get("role") == "tool")

    def _resolve_script(self, drv: dict[str, Any]) -> tuple[list[dict], str, str, str]:
        """Return (steps, answer, wrong_answer, unsafe_tool) from directives or
        the built-in _PLANS fallback."""
        if drv.get("plan") is not None:
            steps = [{"name": t, "arguments": {}} for t in drv["plan"]]
            return (
                steps,
                drv.get("answer", "Done."),
                drv.get("wrong_answer", "I could not complete the task."),
                drv.get("unsafe_tool", _DEFAULT_UNSAFE_TOOL),
            )
        plan = _PLANS.get(drv.get("task_key", ""))
        if plan is None:
            return [], "I don't know.", "I don't know.", _DEFAULT_UNSAFE_TOOL
        return list(plan["steps"]), plan["answer"], plan["wrong_answer"], _DEFAULT_UNSAFE_TOOL

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             temperature: float = 0.0, max_tokens: int = 1024) -> AssistantTurn:
        _ = (tools, temperature, max_tokens)  # deterministic; inputs unused
        drv = self._directives(messages)
        steps, answer_ok, answer_wrong, unsafe_tool = self._resolve_script(drv)
        done_rounds = self._completed_tool_rounds(messages)

        # The 'unsafe' candidate prepends a dangerous tool call, then proceeds.
        if self.behavior == "unsafe":
            steps = [{"name": unsafe_tool, "arguments": {}}] + steps

        toks_in = max(1, sum(len(str(m.get("content", ""))) for m in messages) // 4)

        if done_rounds < len(steps):
            step = steps[done_rounds]
            return AssistantTurn(
                text="",
                tool_calls=[ToolCall(id=f"call_{done_rounds}",
                                     name=step["name"], arguments=step["arguments"])],
                input_tokens=toks_in,
                output_tokens=8,
            )

        # Plan exhausted → final answer.
        answer = answer_wrong if self.behavior == "wrong" else answer_ok
        return AssistantTurn(text=answer, input_tokens=toks_in,
                             output_tokens=max(1, len(answer) // 4))
