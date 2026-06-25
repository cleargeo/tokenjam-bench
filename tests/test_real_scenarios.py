"""TokenJam Real Scenarios — realistic multi-tool agent workflows, offline."""
from __future__ import annotations

from tjbench.agent_pipeline import run_agent_proof
from tjbench.agents.runner import AgentRunner
from tjbench.benchmarks.real_scenarios import RealScenariosBenchmark
from tjbench.models.mock_agent_client import MockAgentClient


def _run(task_id, behavior):
    bench = RealScenariosBenchmark()
    task = next(t for t in bench.tasks() if t.task_id == task_id)
    trace = AgentRunner(MockAgentClient(behavior=behavior), bench.tools()).run(
        task.task_id, task.prompt)
    return bench, task, trace


def test_all_four_scenarios_present():
    ids = [t.task_id for t in RealScenariosBenchmark().tasks()]
    assert ids == [
        "scenario/claude-code-bugfix",
        "scenario/rag-customer-support",
        "scenario/langgraph-research",
        "scenario/deploy-agent",
    ]


def test_ok_candidate_follows_plan_and_passes():
    bench, task, trace = _run("scenario/claude-code-bugfix", "ok")
    assert trace.tool_sequence() == ["read_file", "edit_file", "run_tests"]
    assert bench.score(task, trace).passed


def test_wrong_answer_fails_despite_right_tools():
    bench, task, trace = _run("scenario/rag-customer-support", "wrong")
    assert trace.tool_sequence() == ["search_docs", "get_article"]   # tools still right
    assert not bench.score(task, trace).passed                        # answer wrong


def test_deploy_to_prod_is_caught_by_safety_gate():
    # 'unsafe' prepends the scenario's dangerous tool (deploy_prod) then proceeds.
    bench, task, trace = _run("scenario/deploy-agent", "unsafe")
    assert "deploy_prod" in trace.tool_sequence()
    assert "staging" in trace.final_text          # answer text still looks fine...
    result = bench.score(task, trace)
    assert not result.passed                       # ...but the prod deploy fails it
    assert "unsafe action" in result.detail


def test_each_scenario_has_its_own_dangerous_tool_wired():
    bench = RealScenariosBenchmark()
    dangerous = bench.tools().dangerous_names()
    assert {"delete_repo", "issue_refund", "post_publicly", "deploy_prod"} <= dangerous


def test_proof_pipeline_ok_vs_unsafe_candidate():
    # ok candidate: matches original, cheaper. unsafe candidate: pass-rate collapses.
    ok = run_agent_proof(benchmark_name="real-scenarios",
                         original_spec="anthropic:claude-opus-4-7",
                         mock=True, candidate_behavior="ok")
    assert ok.n_tasks == 4
    assert ok.original_pass == 4 and ok.candidate_pass == 4
    assert ok.cost_delta_pct < 0

    unsafe = run_agent_proof(benchmark_name="real-scenarios",
                             original_spec="anthropic:claude-opus-4-7",
                             mock=True, candidate_behavior="unsafe")
    assert unsafe.original_pass == 4 and unsafe.candidate_pass == 0
    assert unsafe.accuracy_delta_pp == -100.0
