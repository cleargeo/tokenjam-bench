"""Live proof dashboard — scan + render (offline)."""
from __future__ import annotations

import json

from dashboard import _DASHBOARD_HTML, scan_runs


def _artifact(benchmark="humaneval", created_at=0.0, verdict="no_significant_regression"):
    return {
        "tokenjam_version": "0.5.1", "benchmark": benchmark,
        "original_model": "deepseek:deepseek-reasoner",
        "candidate_model": "deepseek:deepseek-chat", "n_tasks": 20,
        "original_pass_rate": 0.95, "candidate_pass_rate": 1.0,
        "accuracy_delta_pp": 5.0, "cost_delta_pct": -75.6, "mock": False,
        "created_at": created_at, "stats": {"verdict": verdict},
    }


def test_scan_runs_summarizes_newest_first(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_artifact("humaneval", created_at=100.0)))
    (tmp_path / "b.json").write_text(json.dumps(_artifact("judged", created_at=200.0)))
    (tmp_path / "junk.json").write_text("{not valid")        # ignored
    runs = scan_runs(tmp_path)
    assert [r["benchmark"] for r in runs] == ["judged", "humaneval"]  # newest first
    top = runs[0]
    assert top["file"] == "b.json"
    assert top["candidate_pass_rate"] == 100.0
    assert top["cost_delta_pct"] == -75.6
    assert top["verdict"] == "no_significant_regression"


def test_scan_empty_dir_is_safe(tmp_path):
    assert scan_runs(tmp_path) == []


def test_dashboard_html_is_offline():
    html = _DASHBOARD_HTML
    assert "<!doctype html>" in html.lower()
    # No render-time external resources (offline-first, like TokenJam Lens).
    for bad in ("http://", "https://", "src=", "cdn", "googleapis"):
        assert bad not in html
    # It polls the local API for realtime updates.
    assert "/api/runs" in html and "setInterval" in html
