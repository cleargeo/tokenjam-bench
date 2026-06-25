"""`tjbench` — run TokenJam savings/accuracy proofs from the command line."""
from __future__ import annotations

import json as _json

import click
from rich.console import Console
from rich.table import Table

from agent_pipeline import run_agent_proof
from benchmarks import AGENT_BENCHMARK_NAMES, BENCHMARK_NAMES
from matrix import build_series, load_artifacts, series_to_dict, total_regressions
from pipeline import resolve_candidate, run_proof
from report_html import load_and_render, write_html_report
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
@click.option("--html", "make_html", is_flag=True,
              help="Also write a self-contained HTML report next to the JSON.")
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
def cmd_run(benchmark: str, original: str, candidate: str | None, limit: int | None,
            samples: int, temperature: float, mock: bool, mock_candidate_accuracy: float,
            max_tokens: int, out: str, make_html: bool, output_json: bool) -> None:
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
    if make_html and not output_json:
        hp = write_html_report(result.to_dict(), out)
        console.print(f"HTML report: [dim]{hp}[/dim]")
    _record_history(result, path, out)


def _record_history(result, path, out) -> None:
    """Best-effort: index this run in the historical DB. Never breaks a run."""
    try:
        from pathlib import Path

        from history import BenchmarkHistory
        html = Path(path).with_suffix(".html")
        with BenchmarkHistory(Path(out) / "history.duckdb") as h:
            h.record(result.to_dict(), json_path=str(path),
                     html_path=str(html) if html.exists() else None)
    except Exception:
        pass


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
@click.option("--html", "make_html", is_flag=True,
              help="Also write a self-contained HTML report next to the JSON.")
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
def cmd_agent(benchmark: str, original: str, candidate: str | None, limit: int | None,
              samples: int, temperature: float, max_turns: int, max_tokens: int,
              mock: bool, candidate_behavior: str, out: str, make_html: bool,
              output_json: bool) -> None:
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
    if make_html and not output_json:
        hp = write_html_report(result.to_dict(), out)
        console.print(f"HTML report: [dim]{hp}[/dim]")
    _record_history(result, path, out)


@cli.command("report")
@click.argument("artifact", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", default=None,
              help="Output dir for the HTML (default: next to the JSON).")
@click.option("--open", "open_browser", is_flag=True, help="Open the report in a browser.")
def cmd_report(artifact: str, out: str | None, open_browser: bool) -> None:
    """Render a saved JSON proof artifact into a self-contained HTML report."""
    path = load_and_render(artifact, out)
    console.print(f"HTML report: [bold]{path}[/bold]")
    if open_browser:
        import webbrowser
        webbrowser.open(path.as_uri())


@cli.command("scenarios")
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
def cmd_scenarios(output_json: bool) -> None:
    """List the Real Scenario Library suites (run with `agent --benchmark <name>`)."""
    from benchmarks.scenario_suites import get_scenario_suite, list_scenario_suites

    suites = {}
    for name in list_scenario_suites():
        s = get_scenario_suite(name)
        suites[name] = {
            "tasks": [t.task_id for t in s.tasks()],
            "tools": s.tools().names(),
            "dangerous_tools": sorted(s.tools().dangerous_names()),
        }
    if output_json:
        console.print_json(data=suites)
        return
    for name, info in suites.items():
        console.print(f"[bold]{name}[/bold]  ({len(info['tasks'])} scenarios)")
        for tid in info["tasks"]:
            console.print(f"    • {tid}")
        console.print(f"    [dim]dangerous tools (safety gate): {info['dangerous_tools']}[/dim]")
    console.print(
        "\n[dim]Run one: tjbench agent --benchmark coding-assistant "
        "--original anthropic:claude-opus-4-7 --mock --html[/dim]"
    )


@cli.command("replay")
@click.option("--telemetry", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Exported TokenJam telemetry (.jsonl/.json) or a TokenJam .duckdb (read-only).")
@click.option("--candidate", default=None,
              help="Candidate model (default: TokenJam's downgrade for the original).")
@click.option("--judge", "judge_backend", default=None,
              type=click.Choice(["mock", "openai", "deepseek"]),
              help="Equivalence judge (default: TJBENCH_JUDGE env, else mock).")
@click.option("--judge-metric", default="correctness", show_default=True)
@click.option("--limit", type=int, default=None, help="Cap number of replayed turns.")
@click.option("--samples", type=int, default=1, show_default=True)
@click.option("--temperature", type=float, default=0.0, show_default=True)
@click.option("--control", is_flag=True,
              help="Also re-run the original model (McNemar vs its own run-to-run noise).")
@click.option("--max-tokens", type=int, default=1024, show_default=True)
@click.option("--mock", is_flag=True, help="Offline run (no provider SDKs/keys/spend).")
@click.option("--mock-candidate-accuracy", type=float, default=0.85, show_default=True)
@click.option("--out", default="results", show_default=True)
@click.option("--html", "make_html", is_flag=True,
              help="Also write a self-contained HTML report.")
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
def cmd_replay(telemetry: str, candidate: str | None, judge_backend: str | None,
               judge_metric: str, limit: int | None, samples: int, temperature: float,
               control: bool, max_tokens: int, mock: bool, mock_candidate_accuracy: float,
               out: str, make_html: bool, output_json: bool) -> None:
    """Replay real TokenJam telemetry: does the cheaper model stay equivalent?"""
    from replay_pipeline import run_replay_proof
    try:
        result = run_replay_proof(
            telemetry_path=telemetry, candidate_spec=candidate, judge_backend=judge_backend,
            judge_metric=judge_metric, limit=limit, samples=samples, temperature=temperature,
            control=control, max_tokens=max_tokens, mock=mock,
            mock_candidate_accuracy=mock_candidate_accuracy,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    path = result.write(out)
    _render_proof(result, path, output_json)
    if make_html and not output_json:
        hp = write_html_report(result.to_dict(), out)
        console.print(f"HTML report: [dim]{hp}[/dim]")
    _record_history(result, path, out)


@cli.command("serve")
@click.option("--dir", "directory", default="results", show_default=True,
              help="Directory of proof artifacts to serve.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7392, show_default=True)
@click.option("--open", "open_browser", is_flag=True, help="Open the dashboard in a browser.")
def cmd_serve(directory: str, host: str, port: int, open_browser: bool) -> None:
    """Start the live proof dashboard (offline, auto-refreshing)."""
    from dashboard import serve

    if open_browser:
        import threading
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    serve(directory=directory, host=host, port=port)


@cli.group("history")
def cmd_history() -> None:
    """Query the historical benchmark database (DuckDB)."""


def _default_db(directory: str = "results") -> str:
    return f"{directory.rstrip('/')}/history.duckdb"


@cmd_history.command("ingest")
@click.option("--dir", "directory", default="results", show_default=True,
              help="Directory of proof artifacts to index.")
@click.option("--db", default=None, help="History DB path (default: <dir>/history.duckdb).")
def cmd_history_ingest(directory: str, db: str | None) -> None:
    """Index every proof artifact in a directory into the history DB."""
    from history import BenchmarkHistory
    db = db or _default_db(directory)
    with BenchmarkHistory(db) as h:
        new, total = h.ingest_dir(directory)
    console.print(f"ingested [bold]{new}[/bold] new / {total} total runs → [dim]{db}[/dim]")


@cmd_history.command("list")
@click.option("--benchmark", default=None, help="Filter to one benchmark.")
@click.option("--limit", default=20, show_default=True)
@click.option("--db", default=_default_db(), show_default=True)
@click.option("--json", "output_json", is_flag=True)
def cmd_history_list(benchmark: str | None, limit: int, db: str, output_json: bool) -> None:
    """List recorded benchmark runs, newest first."""
    import time

    from history import BenchmarkHistory
    with BenchmarkHistory(db) as h:
        runs = h.list_runs(benchmark=benchmark, limit=limit)
    if output_json:
        console.print_json(data=runs)
        return
    if not runs:
        console.print("[dim]No runs recorded. Run a proof, or `tjbench history ingest`.[/dim]")
        return
    table = Table()
    for col in ("when", "benchmark", "tokenjam", "original → candidate", "n",
                "cand", "Δcost", "verdict"):
        table.add_column(col)
    for r in runs:
        when = time.strftime("%m-%d %H:%M", time.localtime(r.get("created_at") or 0))
        table.add_row(
            when, str(r["benchmark"]), str(r.get("tokenjam_version")),
            f"{r.get('original_model')} → {r.get('candidate_model')}",
            str(r.get("n_tasks")), str(r.get("candidate_pass")),
            f"{r.get('cost_delta_pct'):+.1f}%" if r.get("cost_delta_pct") is not None else "—",
            str(r.get("verdict")))
    console.print(table)


@cmd_history.command("versions")
@click.option("--db", default=_default_db(), show_default=True)
@click.option("--json", "output_json", is_flag=True)
def cmd_history_versions(db: str, output_json: bool) -> None:
    """List the TokenJam versions present in history (ascending)."""
    from history import BenchmarkHistory
    with BenchmarkHistory(db) as h:
        versions = h.versions()
    if output_json:
        console.print_json(data={"versions": versions})
    else:
        console.print("  ".join(versions) if versions else "[dim]no versions recorded[/dim]")


@cmd_history.command("trend")
@click.argument("benchmark")
@click.option("--original", default=None)
@click.option("--candidate", default=None)
@click.option("--db", default=_default_db(), show_default=True)
@click.option("--json", "output_json", is_flag=True)
def cmd_history_trend(benchmark: str, original: str | None, candidate: str | None,
                      db: str, output_json: bool) -> None:
    """Accuracy + cost trend for a benchmark across TokenJam versions."""
    from history import BenchmarkHistory
    with BenchmarkHistory(db) as h:
        points = h.trend(benchmark, original_model=original, candidate_model=candidate)
    if output_json:
        console.print_json(data={"benchmark": benchmark, "trend": points})
        return
    if not points:
        console.print(f"[dim]no recorded runs for '{benchmark}'.[/dim]")
        return
    table = Table(title=f"{benchmark} trend")
    for col in ("tokenjam", "cand pass-rate", "acc Δ", "Δcost", "deepeval", "verdict"):
        table.add_column(col)
    for p in points:
        de = p.get("deepeval_score")
        table.add_row(
            str(p.get("tokenjam_version")),
            f"{(p.get('candidate_pass_rate') or 0) * 100:.0f}%",
            f"{p.get('accuracy_delta_pp'):+.1f}pp" if p.get("accuracy_delta_pp") is not None else "—",
            f"{p.get('cost_delta_pct'):+.1f}%" if p.get("cost_delta_pct") is not None else "—",
            f"{de:.2f}" if de is not None else "—",
            str(p.get("verdict")))
    console.print(table)


@cmd_history.command("leaderboard")
@click.argument("benchmark")
@click.option("--db", default=_default_db(), show_default=True)
@click.option("--json", "output_json", is_flag=True)
def cmd_history_leaderboard(benchmark: str, db: str, output_json: bool) -> None:
    """Models ranked by their latest pass-rate on a benchmark."""
    from history import BenchmarkHistory
    with BenchmarkHistory(db) as h:
        rows = h.leaderboard(benchmark)
    if output_json:
        console.print_json(data={"benchmark": benchmark, "rows": rows})
        return
    if not rows:
        console.print(f"[dim]no recorded runs for '{benchmark}'.[/dim]")
        return
    table = Table(title=f"{benchmark} leaderboard")
    for c in ("rank", "model", "pass-rate", "cost", "tokenjam"):
        table.add_column(c)
    for i, r in enumerate(rows, 1):
        table.add_row(
            f"#{i}", str(r["model"]), f"{(r.get('pass_rate') or 0) * 100:.0f}%",
            f"${r['cost_usd']:.6f}" if r.get("cost_usd") is not None else "—",
            str(r.get("tokenjam_version")))
    console.print(table)


@cmd_history.command("providers")
@click.option("--db", default=_default_db(), show_default=True)
@click.option("--json", "output_json", is_flag=True)
def cmd_history_providers(db: str, output_json: bool) -> None:
    """Per-model accuracy + cost aggregated across all benchmarks."""
    from history import BenchmarkHistory
    with BenchmarkHistory(db) as h:
        rows = h.provider_matrix()
    if output_json:
        console.print_json(data={"rows": rows})
        return
    if not rows:
        console.print("[dim]no history recorded.[/dim]")
        return
    table = Table(title="provider / model matrix")
    for c in ("model", "runs", "benchmarks", "avg accuracy", "avg cost"):
        table.add_column(c)
    for r in rows:
        table.add_row(
            str(r["model"]), str(r["runs"]), str(r["benchmarks"]),
            f"{(r.get('avg_accuracy') or 0) * 100:.0f}%",
            f"${r['avg_cost_usd']:.6f}" if r.get("avg_cost_usd") is not None else "—")
    console.print(table)


@cmd_history.command("regressions")
@click.option("--limit", default=50, show_default=True)
@click.option("--db", default=_default_db(), show_default=True)
@click.option("--json", "output_json", is_flag=True)
def cmd_history_regressions(limit: int, db: str, output_json: bool) -> None:
    """Timeline of runs flagged as a regression."""
    import time

    from history import BenchmarkHistory
    with BenchmarkHistory(db) as h:
        rows = h.regressions(limit)
    if output_json:
        console.print_json(data={"rows": rows})
        return
    if not rows:
        console.print("[green]✓ no regressions recorded.[/green]")
        return
    table = Table(title="regression timeline")
    for c in ("when", "benchmark", "models", "tokenjam", "Δacc", "verdict"):
        table.add_column(c)
    for r in rows:
        table.add_row(
            time.strftime("%m-%d %H:%M", time.localtime(r.get("created_at") or 0)),
            str(r["benchmark"]),
            f"{r.get('original_model')} → {r.get('candidate_model')}",
            str(r.get("tokenjam_version")),
            f"{r['accuracy_delta_pp']:+.1f}pp" if r.get("accuracy_delta_pp") is not None else "—",
            str(r.get("verdict")))
    console.print(table)


@cli.command("matrix")
@click.option("--dir", "directory", default="results", show_default=True,
              help="Directory of version-stamped proof artifacts to compare.")
@click.option("--json", "output_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def cmd_matrix(ctx: click.Context, directory: str, output_json: bool) -> None:
    """Compare proofs across TokenJam versions and flag regressions.

    Exits non-zero when any regression (accuracy / cost / recommendation change)
    is detected — so it doubles as a CI guard against TokenJam releases.
    """
    series = build_series(load_artifacts(directory))

    if output_json:
        console.print_json(data=series_to_dict(series))
    elif not series:
        console.print(
            f"[dim]No proof artifacts in '{directory}'. Run some proofs across "
            "TokenJam versions first (run, then `make update-tokenjam`, then run again).[/dim]"
        )
    else:
        for s in series:
            table = Table(title=f"{s.benchmark} · {s.original_model}", title_style="bold")
            table.add_column("tokenjam")
            table.add_column("candidate")
            table.add_column("cand pass", justify="right")
            table.add_column("acc Δ", justify="right")
            table.add_column("cost Δ", justify="right")
            table.add_column("verdict")
            table.add_column("regressions")
            for p in s.points:
                flags = "; ".join(p.regressions)
                flag_disp = f"[yellow]{flags}[/yellow]" if flags else "[green]ok[/green]"
                table.add_row(
                    p.tokenjam_version, p.candidate_model,
                    f"{p.candidate_pass_rate*100:.0f}%",
                    f"{p.accuracy_delta_pp:+.1f}pp", f"{p.cost_delta_pct:+.1f}%",
                    p.verdict, flag_disp,
                )
            console.print(table)

    n = total_regressions(series)
    if not output_json:
        if n:
            console.print(f"\n[yellow]⚠ {n} regression(s) detected across versions.[/yellow]")
        elif series:
            console.print("\n[green]✓ no cross-version regressions.[/green]")
    ctx.exit(1 if n else 0)


def _summary_json(result_dict: dict) -> str:  # pragma: no cover - convenience
    return _json.dumps(result_dict, indent=2)
