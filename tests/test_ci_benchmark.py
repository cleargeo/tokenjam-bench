"""Continuous-benchmark orchestration (P2) — offline path."""
from __future__ import annotations

from tjbench.ci_benchmark import live_enabled, render_summary, run_ci


def test_offline_run_produces_proofs_and_summary(tmp_path):
    results, summary = run_ci(out_dir=tmp_path, live=False)
    labels = [label for label, _ in results]
    assert "samples (offline mock)" in labels
    assert "coding-assistant (offline mock)" in labels
    # offline-only → no live rows
    assert not any("live" in label for label in labels)
    # artifacts written (json + html per proof)
    assert list(tmp_path.glob("*.json")) and list(tmp_path.glob("*.html"))


def test_summary_is_markdown_with_versions_and_verdicts(tmp_path):
    results, summary = run_ci(out_dir=tmp_path, live=False)
    assert summary.startswith("## tokenjam-bench results")
    assert "| benchmark |" in summary          # a table
    assert "verdict" in summary
    assert "never \"safe\"" in summary           # honesty line present


def test_live_gating_keys_off_env(monkeypatch):
    for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert live_enabled() is False
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-not-used-here")
    assert live_enabled() is True


def test_render_summary_handles_empty(tmp_path):
    md = render_summary([], live=False)
    assert "offline only" in md
