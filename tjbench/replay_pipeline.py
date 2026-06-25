"""Replay proof pipeline.

Replays real TokenJam telemetry against a cheaper CANDIDATE model and asks: does
the candidate produce answers EQUIVALENT to what the original model produced on
your actual prompts? The original output is the equivalence reference (real
sessions have no gold answer), the judge scores equivalence, and the per-turn
outcomes flow into the SAME `assemble_proof` as every other benchmark — so
Wilson CIs, McNemar, pass@k, and cost validation apply with no loss of rigor.

Two modes:
  - default: original is the historical reference (passes trivially); the
    candidate's equivalence rate gets a Wilson CI, and McNemar tests whether the
    candidate's divergences are significant.
  - `control=True`: also RE-RUN the original model and judge it against its own
    history. McNemar then compares the candidate against the original's own
    run-to-run / judge noise — the most rigorous form.

Verdict language stays evidence-based ("no statistically significant
divergence"), never "SAFE".
"""
from __future__ import annotations

from tjbench.cost import price_completion
from tjbench.judge import JudgeCase, get_judge, judge_from_env
from tjbench.models import Completion, get_client, parse_spec
from tjbench.pipeline import assemble_proof
from tjbench.recommend import candidate_for
from tjbench.replay import dominant_model, load_telemetry
from tjbench.report import TaskOutcome


def resolve_replay_candidate(original_spec: str) -> str | None:
    provider, model = parse_spec(original_spec)
    cand = candidate_for(provider, model)
    return f"{provider}:{cand}" if cand else None


def _judge_turn(judge, prompt, actual, reference):
    r = judge.evaluate(JudgeCase(input=prompt, actual_output=actual, expected_output=reference))
    return r.passed, r


def run_replay_proof(
    *,
    telemetry_path: str,
    candidate_spec: str | None = None,
    judge_backend: str | None = None,
    judge_metric: str = "correctness",
    limit: int | None = None,
    samples: int = 1,
    temperature: float = 0.0,
    control: bool = False,
    max_tokens: int = 1024,
    mock: bool = False,
    mock_candidate_accuracy: float = 0.85,
    alpha: float = 0.05,
    # dependency injection for tests (production leaves these None):
    candidate_client=None,
    original_client=None,
    judge=None,
):
    """Run a replay-validation proof over exported TokenJam telemetry."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    turns = load_telemetry(telemetry_path)
    if limit is not None:
        turns = turns[:limit]
    if not turns:
        raise ValueError(
            f"No replayable turns in '{telemetry_path}'. Telemetry needs captured "
            "prompt + completion content (TokenJam [capture] prompts/completions)."
        )

    orig_provider, orig_model = dominant_model(turns)
    original_spec = f"{orig_provider}:{orig_model}"

    recommended_by = "tokenjam.DOWNGRADE_CANDIDATES (replay)"
    if candidate_spec is None:
        candidate_spec = resolve_replay_candidate(original_spec)
        if candidate_spec is None:
            raise ValueError(
                f"TokenJam has no downgrade candidate for '{original_spec}'. "
                f"Pass --candidate explicitly."
            )
    else:
        recommended_by = "explicit --candidate (replay)"
    cand_provider, cand_model = parse_spec(candidate_spec)

    if judge is None:
        judge = get_judge(judge_backend, metric=judge_metric) if judge_backend else judge_from_env()
    if candidate_client is None:
        candidate_client = get_client(candidate_spec, mock=mock, mock_accuracy=mock_candidate_accuracy)
    if control and original_client is None:
        original_client = get_client(original_spec, mock=mock, mock_accuracy=1.0)

    outcomes: list[TaskOutcome] = []
    tot_o = tot_c = 0
    for idx, turn in enumerate(turns):
        # Candidate: run samples, judge each against the original's output.
        c_pass = c_out = 0
        c_cost = 0.0
        c_detail = ""
        for _ in range(samples):
            cc = candidate_client.complete(turn.prompt, max_tokens=max_tokens,
                                           temperature=temperature)
            passed, r = _judge_turn(judge, turn.prompt, cc.text, turn.original_output)
            c_pass += 1 if passed else 0
            c_out += cc.output_tokens
            c_cost += price_completion(cand_provider, cand_model, cc)
            c_detail = f"equiv={passed} score={r.score:.2f} vs original"

        # Original: control re-run, or the historical reference (trivially equiv).
        if control and original_client is not None:
            o_pass = o_out = 0
            o_cost = 0.0
            for _ in range(samples):
                oc = original_client.complete(turn.prompt, max_tokens=max_tokens,
                                              temperature=temperature)
                passed, _r = _judge_turn(judge, turn.prompt, oc.text, turn.original_output)
                o_pass += 1 if passed else 0
                o_out += oc.output_tokens
                o_cost += price_completion(turn.provider, turn.model, oc)
        else:
            o_pass = samples  # the historical output is the reference
            o_out = turn.output_tokens
            o_cost = price_completion(
                turn.provider, turn.model,
                Completion(text="", input_tokens=turn.input_tokens,
                           output_tokens=turn.output_tokens))

        tot_o += o_pass
        tot_c += c_pass
        outcomes.append(TaskOutcome(
            task_id=f"{turn.session_id}#{idx}", samples=samples,
            original_passes=o_pass, candidate_passes=c_pass,
            original_cost_usd=round(o_cost, 8), candidate_cost_usd=round(c_cost, 8),
            original_output_tokens=o_out, candidate_output_tokens=c_out,
            original_detail="historical reference" if not control else "control re-run",
            candidate_detail=c_detail,
        ))

    return assemble_proof(
        outcomes,
        benchmark_name="replay",
        original_spec=original_spec, candidate_spec=candidate_spec,
        recommended_by=recommended_by, samples=samples, mock=mock,
        orig_provider=orig_provider, orig_model=orig_model,
        cand_provider=cand_provider, cand_model=cand_model,
        sample_pass_totals=(tot_o, tot_c),
        alpha=alpha,
    )
