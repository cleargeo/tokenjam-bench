"""Cross-version regression matrix — the 'TokenJam changes daily' guard."""
from __future__ import annotations

import json

from tjbench.matrix import (
    _version_key,
    build_series,
    load_artifacts,
    series_to_dict,
    total_regressions,
)


def _artifact(version, *, candidate="anthropic:claude-haiku-4-5", cand_rate=0.90,
              cost_delta=-84.0, created_at=0.0, benchmark="humaneval",
              original="anthropic:claude-opus-4-7"):
    return {
        "tokenjam_version": version, "benchmark": benchmark, "original_model": original,
        "candidate_model": candidate, "n_tasks": 50, "created_at": created_at,
        "candidate_pass_rate": cand_rate, "accuracy_delta_pp": -1.0,
        "cost_delta_pct": cost_delta, "stats": {"verdict": "no_significant_regression"},
    }


def test_version_sorting_is_numeric_not_lexical():
    assert _version_key("0.5.2") < _version_key("0.5.10")   # 2 < 10, not "10" < "2"
    assert _version_key("0.5.1") < _version_key("0.6.0")


def test_clean_series_has_no_regressions():
    arts = [_artifact("0.5.1", cand_rate=0.90), _artifact("0.5.2", cand_rate=0.90)]
    series = build_series(arts)
    assert len(series) == 1
    assert total_regressions(series) == 0
    assert series[0].points[1].regressions == []


def test_accuracy_regression_detected():
    arts = [_artifact("0.5.1", cand_rate=0.90), _artifact("0.5.2", cand_rate=0.85)]  # -5pp
    series = build_series(arts)
    regs = series[0].points[1].regressions
    assert any("accuracy" in r for r in regs)
    assert total_regressions(series) == 1


def test_cost_regression_detected_when_savings_shrink():
    # savings shrink from -84% to -70% (cost delta +14pp) → regression
    arts = [_artifact("0.5.1", cost_delta=-84.0), _artifact("0.5.2", cost_delta=-70.0)]
    series = build_series(arts)
    assert any("savings shrank" in r for r in series[0].points[1].regressions)


def test_recommendation_change_detected():
    arts = [
        _artifact("0.5.1", candidate="anthropic:claude-haiku-4-5"),
        _artifact("0.5.2", candidate="anthropic:claude-haiku-4-6"),
    ]
    series = build_series(arts)
    pt = series[0].points[1]
    assert pt.candidate_changed
    assert any("recommendation changed" in r for r in pt.regressions)


def test_latest_artifact_wins_per_version():
    # two artifacts for 0.5.1; the later created_at should be the one used
    arts = [
        _artifact("0.5.1", cand_rate=0.50, created_at=100.0),
        _artifact("0.5.1", cand_rate=0.95, created_at=200.0),
    ]
    series = build_series(arts)
    assert series[0].points[0].candidate_pass_rate == 0.95


def test_separate_benchmarks_are_separate_series():
    arts = [_artifact("0.5.1", benchmark="humaneval"), _artifact("0.5.1", benchmark="gsm8k")]
    series = build_series(arts)
    assert len(series) == 2


def test_load_and_serialize_roundtrip(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_artifact("0.5.1")))
    (tmp_path / "b.json").write_text(json.dumps(_artifact("0.5.2", cand_rate=0.80)))
    (tmp_path / "junk.json").write_text("{not valid")          # ignored gracefully
    series = build_series(load_artifacts(tmp_path))
    payload = series_to_dict(series)
    assert payload["regressions_found"] >= 1
    assert payload["series"][0]["points"][1]["regressions"]
