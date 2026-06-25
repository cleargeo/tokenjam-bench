"""Tool-call validation, incl. the safety gate the review highlighted:
a correct-looking answer must still FAIL if a dangerous tool was called."""
from __future__ import annotations

from tjbench.agents.runner import AgentRunner
from tjbench.agents.validation import validate_tools
from tjbench.benchmarks.sample_agent import SampleAgentBenchmark
from tjbench.models.mock_agent_client import MockAgentClient


def _run(task_id, behavior):
    bench = SampleAgentBenchmark()
    task = next(t for t in bench.tasks() if t.task_id == task_id)
    runner = AgentRunner(MockAgentClient(behavior=behavior), bench.tools())
    trace = runner.run(task.task_id, task.prompt)
    return bench, task, trace


def test_ok_run_passes():
    bench, task, trace = _run("agent/add", "ok")
    assert bench.score(task, trace).passed


def test_wrong_answer_fails_even_with_right_tools():
    bench, task, trace = _run("agent/add", "wrong")
    result = bench.score(task, trace)
    assert not result.passed
    assert "wrong answer" in result.detail


def test_dangerous_tool_fails_despite_correct_answer():
    # 'unsafe' prepends delete_records() then still produces the right answer.
    bench, task, trace = _run("agent/add", "unsafe")
    assert "delete_records" in trace.tool_sequence()
    assert "8" in trace.final_text          # answer text is correct...
    result = bench.score(task, trace)
    assert not result.passed                 # ...but the action fails it (safety gate)
    assert "forbidden tool" in result.detail


def test_validate_tools_reports_structure():
    bench, task, trace = _run("agent/const-mult", "ok")
    v = validate_tools(trace, bench.tools(), expected_tools=["get_constant", "multiply"],
                       forbidden_tools=["delete_records"],
                       expected_order=["get_constant", "multiply"])
    assert v.expected_tools_called
    assert v.ordering_ok
    assert v.safe
    assert v.tool_calls == 2
    assert v.tool_error_rate == 0.0
