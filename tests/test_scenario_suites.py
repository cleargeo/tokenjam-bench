"""Real Scenario Library (P1) — every suite runs offline, with a working safety gate."""
from __future__ import annotations

import pytest

from tjbench.agent_pipeline import run_agent_proof
from tjbench.agents.runner import AgentRunner
from tjbench.benchmarks import AGENT_BENCHMARK_NAMES, get_agent_benchmark
from tjbench.benchmarks.scenario_suites import SCENARIO_SUITES, list_scenario_suites
from tjbench.models.mock_agent_client import MockAgentClient

SUITES = list_scenario_suites()


def test_all_suites_registered_as_agent_benchmarks():
    for name in SUITES:
        assert name in AGENT_BENCHMARK_NAMES
        assert get_agent_benchmark(name).name == name


@pytest.mark.parametrize("suite_name", SUITES)
def test_suite_has_scenarios_and_a_dangerous_tool(suite_name):
    suite = SCENARIO_SUITES[suite_name]()
    assert len(suite.tasks()) >= 3
    assert suite.tools().dangerous_names(), f"{suite_name} has no dangerous tool"


@pytest.mark.parametrize("suite_name", SUITES)
def test_ok_candidate_passes_every_scenario(suite_name):
    suite = SCENARIO_SUITES[suite_name]()
    for task in suite.tasks():
        trace = AgentRunner(MockAgentClient(behavior="ok"), suite.tools()).run(
            task.task_id, task.prompt)
        assert suite.score(task, trace).passed, f"{task.task_id} failed for ok candidate"


@pytest.mark.parametrize("suite_name", SUITES)
def test_unsafe_candidate_is_caught_by_the_safety_gate(suite_name):
    suite = SCENARIO_SUITES[suite_name]()
    for task in suite.tasks():
        trace = AgentRunner(MockAgentClient(behavior="unsafe"), suite.tools()).run(
            task.task_id, task.prompt)
        # the candidate reached for the scenario's dangerous tool
        assert task.forbidden_tools[0] in trace.tool_sequence()
        result = suite.score(task, trace)
        assert not result.passed and "unsafe action" in result.detail


def test_scenario_proof_pipeline_ok_vs_unsafe():
    ok = run_agent_proof(benchmark_name="coding-assistant",
                         original_spec="anthropic:claude-opus-4-7",
                         mock=True, candidate_behavior="ok")
    assert ok.benchmark == "coding-assistant"
    assert ok.original_pass == ok.n_tasks and ok.candidate_pass == ok.n_tasks
    assert ok.cost_delta_pct < 0

    unsafe = run_agent_proof(benchmark_name="browser-agent",
                             original_spec="anthropic:claude-opus-4-7",
                             mock=True, candidate_behavior="unsafe")
    assert unsafe.candidate_pass == 0          # every scenario fails the safety gate
    assert unsafe.accuracy_delta_pp == -100.0
