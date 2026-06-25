"""Agent proof pipeline.

Runs an agent benchmark on the ORIGINAL model and the CANDIDATE model (TokenJam's
recommendation), each via the multi-turn AgentRunner, scores each run on its
trace (answer correctness + tool-call validation incl. the safety gate), prices
the summed multi-turn token usage, and feeds the per-task outcomes into the SAME
assembler the single-shot path uses — so Wilson CIs, McNemar, and cost
validation apply unchanged.

This is the payoff of the keystone: agent benchmarks inherit all the statistical
rigor for free, and a candidate that takes an unsafe action fails the task even
when its answer text is correct.
"""
from __future__ import annotations

from tjbench.agents.runner import AgentRunner
from tjbench.benchmarks import get_agent_benchmark
from tjbench.cost import price_completion
from tjbench.models.anthropic_agent_client import get_tool_calling_client
from tjbench.models.registry import parse_spec
from tjbench.pipeline import assemble_proof, resolve_candidate
from tjbench.report import TaskOutcome


def _run_agent_samples(client, provider, model, benchmark, task, registry,
                       samples, max_turns, temperature, max_tokens):
    passes = 0
    out_tok = 0
    cost = 0.0
    last_detail = ""
    for _ in range(samples):
        runner = AgentRunner(client, registry, max_turns=max_turns,
                             temperature=temperature, max_tokens=max_tokens)
        trace = runner.run(task.task_id, task.prompt)
        score = benchmark.score(task, trace)
        seq = trace.tool_sequence()
        last_detail = (
            f"tools={seq} stopped={trace.stopped_reason} "
            f"turns={trace.num_turns} -> {score.detail}"
        )
        if score.passed:
            passes += 1
        out_tok += trace.total_output_tokens
        cost += price_completion(provider, model, trace.as_completion())
    return passes, out_tok, round(cost, 8), last_detail


def run_agent_proof(
    *,
    benchmark_name: str,
    original_spec: str,
    candidate_spec: str | None = None,
    limit: int | None = None,
    samples: int = 1,
    temperature: float = 0.0,
    max_turns: int = 8,
    max_tokens: int = 1024,
    mock: bool = False,
    candidate_behavior: str = "ok",
    alpha: float = 0.05,
):
    """Run an agent-benchmark proof (original vs TokenJam's candidate)."""
    if samples < 1:
        raise ValueError("samples must be >= 1")

    recommended_by = "tokenjam.DOWNGRADE_CANDIDATES"
    if candidate_spec is None:
        candidate_spec = resolve_candidate(original_spec)
        if candidate_spec is None:
            raise ValueError(
                f"TokenJam has no downgrade candidate for '{original_spec}'. "
                f"Pass --candidate explicitly to override."
            )
    else:
        recommended_by = "explicit --candidate override"

    orig_provider, orig_model = parse_spec(original_spec)
    cand_provider, cand_model = parse_spec(candidate_spec)

    # Offline: original behaves correctly, candidate's behavior is configurable
    # (ok | wrong | unsafe) to exercise the answer + safety gates.
    original = get_tool_calling_client(original_spec, mock=mock, behavior="ok")
    candidate = get_tool_calling_client(candidate_spec, mock=mock, behavior=candidate_behavior)

    benchmark = get_agent_benchmark(benchmark_name)
    registry = benchmark.tools()
    tasks = benchmark.tasks(limit=limit)

    outcomes: list[TaskOutcome] = []
    tot_o = tot_c = 0
    for task in tasks:
        o_pass, o_out, o_cost, o_detail = _run_agent_samples(
            original, orig_provider, orig_model, benchmark, task, registry,
            samples, max_turns, temperature, max_tokens)
        c_pass, c_out, c_cost, c_detail = _run_agent_samples(
            candidate, cand_provider, cand_model, benchmark, task, registry,
            samples, max_turns, temperature, max_tokens)
        tot_o += o_pass
        tot_c += c_pass
        outcomes.append(TaskOutcome(
            task_id=task.task_id, samples=samples,
            original_passes=o_pass, candidate_passes=c_pass,
            original_cost_usd=o_cost, candidate_cost_usd=c_cost,
            original_output_tokens=o_out, candidate_output_tokens=c_out,
            original_detail=o_detail, candidate_detail=c_detail,
        ))

    return assemble_proof(
        outcomes,
        benchmark_name=benchmark_name,
        original_spec=original_spec, candidate_spec=candidate_spec,
        recommended_by=recommended_by, samples=samples, mock=mock,
        orig_provider=orig_provider, orig_model=orig_model,
        cand_provider=cand_provider, cand_model=cand_model,
        sample_pass_totals=(tot_o, tot_c),
        alpha=alpha,
    )
