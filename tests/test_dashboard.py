"""Live proof dashboard — scan + render (offline)."""
from __future__ import annotations

import json

from tjbench.dashboard import _DASHBOARD_HTML, scan_runs


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


def test_scan_runs_hides_mock_and_demo_from_production(tmp_path):
    # Production dashboards must show measured runs only — never mock (--mock/dev)
    # or demo (seeded fixture) artifacts.
    real = _artifact("gsm8k", created_at=300.0)
    mock = {**_artifact("humaneval", created_at=200.0), "mock": True}
    demo = {**_artifact("judged", created_at=100.0), "demo": True}
    (tmp_path / "real.json").write_text(json.dumps(real))
    (tmp_path / "mock.json").write_text(json.dumps(mock))
    (tmp_path / "demo.json").write_text(json.dumps(demo))

    prod = scan_runs(tmp_path)                       # default = production
    assert [r["benchmark"] for r in prod] == ["gsm8k"]

    everything = scan_runs(tmp_path, include_dev=True)   # dev opt-in
    assert {r["benchmark"] for r in everything} == {"gsm8k", "humaneval", "judged"}


def test_history_summary_empty_and_populated(tmp_path):
    from tjbench.dashboard import history_summary
    from tjbench.history import BenchmarkHistory

    db = tmp_path / "history.duckdb"
    assert history_summary(db)["available"] is False        # no DB yet
    with BenchmarkHistory(db) as h:
        h.record({"benchmark": "humaneval", "tokenjam_version": "0.5.1",
                  "created_at": 1.0, "original_model": "a:opus", "candidate_model": "a:haiku",
                  "candidate_pass_rate": 1.0, "stats": {"verdict": "no_significant_regression"}})
    s = history_summary(db)
    assert s["available"] and s["count"] == 1
    assert s["versions"] == ["0.5.1"]
    assert s["configs"][0]["benchmark"] == "humaneval"


def test_dashboard_html_is_offline():
    html = _DASHBOARD_HTML
    assert "<!doctype html>" in html.lower()
    # No render-time external resources (offline-first, like TokenJam Lens).
    for bad in ("http://", "https://", "src=", "cdn", "googleapis"):
        assert bad not in html
    # It reads from the local API.
    assert "/api/runs" in html
    # Static evidence dashboard: no live-poll clock that would imply realtime
    # data it doesn't have.
    assert "setInterval" not in html
    # ...and draws an inline-SVG accuracy/cost trend (no charting lib).
    assert "drawChart" in html and "chartbox" in html and "polyline" in html
