"""AgentRunner — the multi-turn loop. THE keystone.

It drives a tool-calling model against a tool registry until the model returns a
final answer or hits `max_turns`, recording every turn into an AgentTrace. This
one component is what unlocks agent benchmarks, multi-turn evaluation, tool-call
validation, and side-effect safety — all of which were blocked on the
single-shot `complete(prompt)` interface.
"""
from __future__ import annotations

from agents.tools import ToolRegistry
from agents.trace import AgentTrace, ToolCallRecord, TurnRecord
from models.tool_calling import ToolCallingClient


class AgentRunner:
    def __init__(self, client: ToolCallingClient, tools: ToolRegistry,
                 max_turns: int = 8, temperature: float = 0.0,
                 max_tokens: int = 1024) -> None:
        self.client = client
        self.tools = tools
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, task_id: str, prompt: str) -> AgentTrace:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        trace = AgentTrace(task_id=task_id)
        specs = self.tools.specs()

        for i in range(self.max_turns):
            turn = self.client.chat(messages, specs, self.temperature, self.max_tokens)

            if not turn.wants_tools:
                trace.turns.append(TurnRecord(
                    index=i, assistant_text=turn.text, tool_calls=[],
                    input_tokens=turn.input_tokens, output_tokens=turn.output_tokens,
                    cache_tokens=turn.cache_tokens,
                ))
                trace.final_text = turn.text
                trace.stopped_reason = "final"
                return trace

            # The model asked for tool(s): execute each, record, feed results back.
            messages.append({
                "role": "assistant", "content": turn.text, "tool_calls": turn.tool_calls,
            })
            call_records: list[ToolCallRecord] = []
            for tc in turn.tool_calls:
                result = self.tools.execute(tc.name, tc.arguments)
                call_records.append(ToolCallRecord(
                    name=tc.name, arguments=tc.arguments,
                    result=result.output, is_error=result.is_error,
                ))
                messages.append({
                    "role": "tool", "tool_call_id": tc.id, "name": tc.name,
                    "content": result.output,
                })
            trace.turns.append(TurnRecord(
                index=i, assistant_text=turn.text, tool_calls=call_records,
                input_tokens=turn.input_tokens, output_tokens=turn.output_tokens,
                cache_tokens=turn.cache_tokens,
            ))

        trace.stopped_reason = "max_turns"
        return trace
