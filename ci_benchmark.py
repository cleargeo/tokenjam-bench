"""Continuous-benchmark orchestration (P2).

Runs a fixed benchmark set for CI, writes version-stamped JSON + HTML artifacts,
and emits a Markdown summary (to GITHUB_STEP_SUMMARY when present). Designed for
`.github/workflows/benchmark.yml`, but runnable locally (`make ci-bench`).

Secret handling: live proofs run ONLY when a provider key is in the environment
(key-gated). The offline portion needs no keys and always runs. Keys are read
from the environment and never printed or persisted.
"""
from __future__ import annotations

import os
from pathlib import Path

from agent_pipeline import run_agent_proof
from pipeline import run_proof
from report_html import write_html_report
from version import resolve_tokenjam_build

# Live proofs run when any of these provider keys is present.
_LIVE_KEYS = ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


def live_enabled() -> bool:
    return any(os.environ.get(k) for k in _LIVE_KEYS)


def run_ci(out_dir: str | Path = "results", live: bool | None = None,
           live_limit: int = 10):
    """Run the CI benchmark set. Returns (list[(label, ProofResult)], summary_md)."""
    live = live_enabled() if live is None else live
    results: list[tuple[str, object]] = []

    # Offline (always): proves the proof pipeline is valid on this build/version.
    results.append((
        "samples (offline mock)",
        run_proof(benchmark_name="samples", original_spec="anthropic:claude-opus-4-7",
                  mock=True, mock_candidate_accuracy=1.0)))
    results.append((
        "coding-assistant (offline mock)",
        run_agent_proof(benchmark_name="coding-assistant",
                        original_spec="anthropic:claude-opus-4-7", mock=True,
                        candidate_behavior="ok")))

    # Live (key-gated): small real downsize proofs on the latest TokenJam.
    if live and os.environ.get("DEEPSEEK_API_KEY"):
        results.append((
            "humaneval (live deepseek)",
            run_proof(benchmark_name="humaneval",
                      original_spec="deepseek:deepseek-reasoner",
                      candidate_spec="deepseek:deepseek-chat", limit=live_limit)))
        results.append((
            "judged (live, deepseek judge)",
            run_proof(benchmark_name="judged",
                      original_spec="deepseek:deepseek-reasoner",
                      candidate_spec="deepseek:deepseek-chat", limit=5)))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for _label, r in results:
        r.write(out)
        write_html_report(r.to_dict(), out)

    # Index every run into the historical DB (best-effort; never breaks CI).
    try:
        from history import BenchmarkHistory
        with BenchmarkHistory(out / "history.duckdb") as h:
            for _label, r in results:
                h.record(r.to_dict())
    except Exception:
        pass

    return results, render_summary(results, live)


def render_summary(results, live: bool) -> str:
    build = resolve_tokenjam_build()
    mode = "LIVE + offline" if live else "offline only (no provider key set)"
    lines = [
        f"## tokenjam-bench results — tokenjam {build.version}",
        "",
        f"_mode: {mode}_",
        "",
        "| benchmark | original → candidate | n | equiv/pass | Δ cost | verdict |",
        "|---|---|--:|--:|--:|---|",
    ]
    for label, r in results:
        lines.append(
            f"| {label} | `{r.original_model}` → `{r.candidate_model}` | {r.n_tasks} | "
            f"{r.candidate_pass}/{r.n_tasks} | {r.cost_delta_pct:+.1f}% | "
            f"**{r.stats.verdict}** |")
    lines.append("")
    lines.append("_Verdicts are evidence-based (Wilson CI + McNemar); never \"safe\"._")
    return "\n".join(lines)


def main() -> None:
    results, summary = run_ci()
    print(summary)
    gh = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh:
        with open(gh, "a", encoding="utf-8") as f:
            f.write(summary + "\n")


if __name__ == "__main__":
    main()
