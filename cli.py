"""`tjbench` — run TokenJam savings/accuracy proofs from the command line."""
from __future__ import annotations

import json as _json

import click
from rich.console import Console
from rich.table import Table

from agent_pipeline import run_agent_proof
from benchmarks import AGENT_BENCHMARK_NAMES, BENCHMARK_NAMES
from pipeline import resolve_candidate, run_proof
from version import resolve_tokenjam_build

console = Console()


@click.group()
def cli() -> None:
    """Prove TokenJam's savings against executable-accuracy ground truth."""


@cli.command("version")
def cmd_version() -> None:
    """Show the bench + the resolved TokenJam build it will test."""
    from bench_meta import __version__

    build = resolve_tokenjam_build()
    console.print(f"tokenjam-bench {__version__}")
    console.print(f"tokenjam (under test): [bold]{build.version}[/bold]")
    console.print(f"  from: {build.location}")


@cli.command("recommend")
@click.option("--original", required=True, help="Original model, 'provider:model'.")
def cmd_recommend(original: str) -> None:
    """Show what TokenJam would downsize the given model to."""
    cand = resolve_candidate(original)
    if cand is None:
        console.print(f"TokenJam has no downgrade candidate for [bold]{original}[/bold].")
    else:
        console.print(f"{original} → [bold]{cand}[/bold]  (tokenjam.DOWNGRADE_CANDIDATES)")


@cli.command("run")
@click.option("--benchmark", type=click.Choice(BENCHMARK_NAMES), default="samples",
              show_default=True, help="Which benchmark to run.")
@click.option("--original", required=True,
              help="Original model spec, e.g. anthropic:claude-opus-4-7.")
@click.option("--candidate", default=None,
              help="Override the candidate model (default: TokenJam's recommendation).")
@click.option("--limit", type=int, default=None, help="Cap number of tasks.")
@click.option("--samples", type=int, default=1, show_default=True,
              help="Samples per task per model (k). >1 enables pass@k / variance.")
@click.option("--temperature", type=float, default=0.0, show_default=True,
              help="Sampling temperature (use >0 with --samples for variance).")
@click.option("--mock", is_flag=True,
              help="Offline run (no provider SDKs/keys/spend). Numbers illustrative.")
@click.option("--mock-candidate-accuracy", type=float, default=0.85, show_default=True,
              help="In --mock mode, simulated candidate pass fraction.")
@click.option("--max-tokens", type=int, default=1024, show_default=True)
@click.option("--out", default="results", show_default=True,
              help="Directory for the version-stamped JSON artifact.")
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
def cmd_run(benchmark: str, original: str, candidate: str | None, limit: int | None,
            samples: int, temperature: float, mock: bool, mock_candidate_accuracy: float,
            max_tokens: int, out: str, output_json: bool) -> None:
    """Run an original-vs-candidate proof and write a stamped artifact."""
    try:
        result = run_proof(
            benchmark_name=benchmark,
            original_spec=original,
            candidate_spec=candidate,
            limit=limit,
            samples=samples,
            temperature=temperature,
            mock=mock,
            mock_candidate_accuracy=mock_candidate_accuracy,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    path = result.write(out)
    _render_proof(result, path, output_json)


def _render_proof(result, path, output_json: bool) -> None:
    """Shared rendering for single-shot and agent proofs."""
    if output_json:
        payload = result.to_dict()
        payload["artifact_path"] = str(path)
        console.print_json(data=payload)
        return

    s = result.stats
    table = Table(title=result.headline(), title_style="bold")
    table.add_column("Metric")
    table.add_column("Original", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_row("Model", result.original_model, result.candidate_model)
    table.add_row(
        "Pass rate (95% CI)",
        f"{result.original_pass}/{result.n_tasks} "
        f"[{s.original_ci_pp[0]:.0f}–{s.original_ci_pp[1]:.0f}%]",
        f"{result.candidate_pass}/{result.n_tasks} "
        f"[{s.candidate_ci_pp[0]:.0f}–{s.candidate_ci_pp[1]:.0f}%]",
    )
    table.add_row("Cost (USD, measured)", f"${result.original_cost_usd:.6f}",
                  f"${result.candidate_cost_usd:.6f}")
    table.add_row("Output tokens", f"{result.original_output_tokens:,}",
                  f"{result.candidate_output_tokens:,}")
    console.print(table)

    verdict_colour = "yellow" if s.verdict != "no_significant_regression" else "green"
    console.print(
        f"Δ cost [bold]{result.cost_delta_pct:+.1f}%[/bold] (measured)   "
        f"Δ pass-rate [bold]{result.accuracy_delta_pp:+.1f}pp[/bold] "
        f"[95% CI {s.delta_ci_pp[0]:+.1f}, {s.delta_ci_pp[1]:+.1f}]"
    )
    console.print(
        f"McNemar: b={s.mcnemar_b} (broke) c={s.mcnemar_c} (fixed) "
        f"p={s.mcnemar_p_value:.3f} (α={s.alpha})   "
        f"verdict: [{verdict_colour}]{s.verdict}[/{verdict_colour}]"
    )
    console.print(f"candidate chosen by: [dim]{result.recommended_by}[/dim]")

    # Honesty footer — mirrors TokenJam's own discipline.
    notes = [
        "Accuracy = pass-rate on THIS benchmark suite; not a general "
        "'quality preserved' claim. Confidence is the CI + p-value, "
        "not a single 'safe %'.",
    ]
    if s.verdict == "insufficient_evidence":
        notes.append(
            f"Too few tasks (n={result.n_tasks}) for a significance verdict — "
            "raise --limit for a defensible result."
        )
    if result.token_inflation_flag:
        notes.append(
            f"Candidate produced {result.output_token_inflation}x the output "
            "tokens — measured savings already reflect this, but the per-token "
            "advantage is being eroded (verbosity/retries)."
        )
    if result.mock:
        notes.append(
            "MOCK run: offline, deterministic — numbers are illustrative, not "
            "from the real models. Drop --mock with API keys set for a real proof."
        )
    if result.priced_with_defaults:
        notes.append(
            "A model had no TokenJam rate; cost used TokenJam's $0.50/$2.00 "
            "default placeholder — savings figure is approximate."
        )
    for note in notes:
        console.print(f"[dim]• {note}[/dim]")
    console.print(f"\nArtifact: [dim]{path}[/dim]")


@cli.command("agent")
@click.option("--benchmark", type=click.Choice(AGENT_BENCHMARK_NAMES),
              default="sample-agent", show_default=True, help="Agent benchmark to run.")
@click.option("--original", required=True,
              help="Original model spec, e.g. anthropic:claude-opus-4-7.")
@click.option("--candidate", default=None,
              help="Override candidate (default: TokenJam's recommendation).")
@click.option("--limit", type=int, default=None, help="Cap number of tasks.")
@click.option("--samples", type=int, default=1, show_default=True,
              help="Samples per task per model (k).")
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--max-turns", type=int, default=8, show_default=True,
              help="Max agent turns before giving up on a task.")
@click.option("--max-tokens", type=int, default=1024, show_default=True)
@click.option("--mock", is_flag=True,
              help="Offline run (deterministic tool-calling mock; no keys/spend).")
@click.option("--candidate-behavior", type=click.Choice(["ok", "wrong", "unsafe"]),
              default="ok", show_default=True,
              help="In --mock mode: simulate the candidate's behavior "
                   "('unsafe' exercises the dangerous-tool safety gate).")
@click.option("--out", default="results", show_default=True)
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
def cmd_agent(benchmark: str, original: str, candidate: str | None, limit: int | None,
              samples: int, temperature: float, max_turns: int, max_tokens: int,
              mock: bool, candidate_behavior: str, out: str, output_json: bool) -> None:
    """Run a multi-turn AGENT proof (tool use + safety), with the same stats."""
    try:
        result = run_agent_proof(
            benchmark_name=benchmark, original_spec=original, candidate_spec=candidate,
            limit=limit, samples=samples, temperature=temperature, max_turns=max_turns,
            max_tokens=max_tokens, mock=mock, candidate_behavior=candidate_behavior,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    path = result.write(out)
    _render_proof(result, path, output_json)


def _summary_json(result_dict: dict) -> str:  # pragma: no cover - convenience
    return _json.dumps(result_dict, indent=2)
