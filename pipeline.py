"""The proof pipeline.

For each benchmark task, run it k times on the ORIGINAL model and k times on the
CANDIDATE model (the one TokenJam recommends), score every sample against
objective ground truth, and price from MEASURED token usage. Aggregate into a
version-stamped ProofResult that carries error bars (Wilson CIs), a paired
significance test (McNemar), and a cost-inflation check.

This is the point: it doesn't ask whether outputs *look* similar — it executes
the tests, measures how often the cheaper model still passes, attaches a CI and
a p-value so the delta isn't mistaken for noise, and measures real spend.
"""
from __future__ import annotations

from benchmarks import get_benchmark
from cost import is_priced_with_defaults, price_completion
from models import Completion, get_client, parse_spec
from recommend import candidate_for
from report import (
    INSUFFICIENT_EVIDENCE,
    NO_SIGNIFICANT_REGRESSION,
    SIGNIFICANT_REGRESSION,
    ProofResult,
    ProofStats,
    TaskOutcome,
)
from stats import (
    Interval,
    mcnemar_exact,
    paired_delta_ci,
    wilson_interval,
)
from version import resolve_tokenjam_build


def _ci_pp(iv: Interval) -> list[float]:
    """Interval (proportion) → [low_pp, high_pp]."""
    pp = iv.as_pp()
    return [pp.low, pp.high]

# Below this many tasks we won't claim significance either way — too few paired
# observations for McNemar to mean anything.
MIN_TASKS_FOR_VERDICT = 10


def resolve_candidate(original_spec: str) -> str | None:
    provider, model = parse_spec(original_spec)
    cand_model = candidate_for(provider, model)
    return f"{provider}:{cand_model}" if cand_model else None


def _run_samples(client, provider, model, benchmark, task, samples, max_tokens, temperature):
    """Run one task `samples` times; return (passes, summed Completion, cost)."""
    passes = 0
    in_tok = out_tok = cache_tok = 0
    last_detail = ""
    for _ in range(samples):
        c = client.complete(task.prompt, max_tokens=max_tokens, temperature=temperature)
        in_tok += c.input_tokens
        out_tok += c.output_tokens
        cache_tok += c.cache_tokens
        score = benchmark.score(task, c.text)
        last_detail = score.detail
        if score.passed:
            passes += 1
    agg = Completion(text="", input_tokens=in_tok, output_tokens=out_tok, cache_tokens=cache_tok)
    cost = price_completion(provider, model, agg)
    return passes, out_tok, cost, last_detail


def _verdict(n_tasks: int, significant: bool, delta_pp: float) -> str:
    if n_tasks < MIN_TASKS_FOR_VERDICT:
        return INSUFFICIENT_EVIDENCE
    if significant and delta_pp < 0:
        return SIGNIFICANT_REGRESSION
    return NO_SIGNIFICANT_REGRESSION


def run_proof(
    *,
    benchmark_name: str,
    original_spec: str,
    candidate_spec: str | None = None,
    limit: int | None = None,
    samples: int = 1,
    temperature: float = 0.0,
    mock: bool = False,
    mock_candidate_accuracy: float = 0.85,
    max_tokens: int = 1024,
    alpha: float = 0.05,
) -> ProofResult:
    """Run the original-vs-candidate proof for one benchmark, with stats."""
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

    original = get_client(original_spec, mock=mock, mock_accuracy=1.0)
    candidate = get_client(candidate_spec, mock=mock, mock_accuracy=mock_candidate_accuracy)

    benchmark = get_benchmark(benchmark_name)
    tasks = benchmark.tasks(limit=limit)

    outcomes: list[TaskOutcome] = []
    orig_cost_total = cand_cost_total = 0.0
    orig_out_total = cand_out_total = 0
    total_orig_sample_passes = total_cand_sample_passes = 0
    for task in tasks:
        o_pass, o_out, o_cost, o_detail = _run_samples(
            original, orig_provider, orig_model, benchmark, task, samples, max_tokens, temperature)
        c_pass, c_out, c_cost, c_detail = _run_samples(
            candidate, cand_provider, cand_model, benchmark, task, samples, max_tokens, temperature)
        orig_cost_total += o_cost
        cand_cost_total += c_cost
        orig_out_total += o_out
        cand_out_total += c_out
        total_orig_sample_passes += o_pass
        total_cand_sample_passes += c_pass
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
        sample_pass_totals=(total_orig_sample_passes, total_cand_sample_passes),
        alpha=alpha,
    )


def assemble_proof(
    outcomes: list[TaskOutcome],
    *,
    benchmark_name: str,
    original_spec: str,
    candidate_spec: str,
    recommended_by: str,
    samples: int,
    mock: bool,
    orig_provider: str,
    orig_model: str,
    cand_provider: str,
    cand_model: str,
    sample_pass_totals: tuple[int, int],
    alpha: float = 0.05,
) -> ProofResult:
    """Turn per-task outcomes into a ProofResult with the full statistical block.

    Shared by the single-shot and agent pipelines so the Wilson/McNemar/cost
    machinery is identical for both — an agent run yields the same per-task
    pass/fail + measured cost, so it gets the same rigor for free.
    """
    build = resolve_tokenjam_build()
    n = len(outcomes)
    orig_pass = sum(1 for o in outcomes if o.original_passed)
    cand_pass = sum(1 for o in outcomes if o.candidate_passed)
    b = sum(1 for o in outcomes if o.original_passed and not o.candidate_passed)
    c = sum(1 for o in outcomes if o.candidate_passed and not o.original_passed)

    mc = mcnemar_exact(b, c)
    significant = mc.p_value < alpha
    delta_pp = round((cand_pass / n - orig_pass / n) * 100, 2) if n else 0.0
    total_orig_sample_passes, total_cand_sample_passes = sample_pass_totals

    stats = ProofStats(
        samples_per_task=samples,
        original_ci_pp=_ci_pp(wilson_interval(orig_pass, n)),
        candidate_ci_pp=_ci_pp(wilson_interval(cand_pass, n)),
        delta_ci_pp=_ci_pp(paired_delta_ci(b, c, n)),
        mcnemar_b=b, mcnemar_c=c, mcnemar_p_value=mc.p_value,
        significant=significant, alpha=alpha,
        verdict=_verdict(n, significant, delta_pp),
        original_pass_at_1=round(total_orig_sample_passes / (n * samples), 4) if n else 0.0,
        candidate_pass_at_1=round(total_cand_sample_passes / (n * samples), 4) if n else 0.0,
    )

    return ProofResult(
        tokenjam_version=build.version,
        tokenjam_location=build.location,
        benchmark=benchmark_name,
        original_model=original_spec,
        candidate_model=candidate_spec,
        recommended_by=recommended_by,
        n_tasks=n,
        original_pass=orig_pass,
        candidate_pass=cand_pass,
        original_cost_usd=round(sum(o.original_cost_usd for o in outcomes), 8),
        candidate_cost_usd=round(sum(o.candidate_cost_usd for o in outcomes), 8),
        original_output_tokens=sum(o.original_output_tokens for o in outcomes),
        candidate_output_tokens=sum(o.candidate_output_tokens for o in outcomes),
        mock=mock,
        priced_with_defaults=(
            is_priced_with_defaults(orig_provider, orig_model)
            or is_priced_with_defaults(cand_provider, cand_model)
        ),
        stats=stats,
        tasks=outcomes,
    )
