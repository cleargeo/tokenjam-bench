"""AgentRunner + trace: the multi-turn loop must execute tools and record them."""
from __future__ import annotations

from agents.runner import AgentRunner
from benchmarks.sample_agent import SampleAgentBenchmark
from models.mock_agent_client import MockAgentClient


def _task(task_id):
    bench = SampleAgentBenchmark()
    task = next(t for t in bench.tasks() if t.task_id == task_id)
    return bench, task


def test_single_tool_loop_records_call_and_final_answer():
    bench, task = _task("agent/add")
    runner = AgentRunner(MockAgentClient(behavior="ok"), bench.tools())
    trace = runner.run(task.task_id, task.prompt)
    assert trace.tool_sequence() == ["add"]
    assert trace.stopped_reason == "final"
    assert "8" in trace.final_text
    # tokens accumulated across the tool turn + the final turn.
    assert trace.total_output_tokens > 0


def test_multi_step_loop_runs_tools_in_order():
    bench, task = _task("agent/const-mult")
    runner = AgentRunner(MockAgentClient(behavior="ok"), bench.tools())
    trace = runner.run(task.task_id, task.prompt)
    assert trace.tool_sequence() == ["get_constant", "multiply"]
    assert trace.num_turns == 3   # two tool turns + one final answer turn


def test_max_turns_guard_stops_runaway():
    # An agent that always asks for a tool (never finalizes) must stop at max_turns.
    bench, task = _task("agent/add")

    class _LoopForever:
        provider = "mock"
        model = "loop"

        def chat(self, messages, tools, temperature=0.0, max_tokens=1024):
            from models.tool_calling import AssistantTurn, ToolCall
            return AssistantTurn(tool_calls=[ToolCall("x", "add", {"a": 1, "b": 1})],
                                 input_tokens=1, output_tokens=1)

    trace = AgentRunner(_LoopForever(), bench.tools(), max_turns=3).run(task.task_id, task.prompt)
    assert trace.stopped_reason == "max_turns"
    assert trace.num_turns == 3
