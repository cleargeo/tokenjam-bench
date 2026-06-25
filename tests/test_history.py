"""Historical benchmark database (P3) — schema, idempotent record, queries."""
from __future__ import annotations

import json

from tjbench.history import BenchmarkHistory, run_id_for


def _proof(benchmark="humaneval", version="0.5.1", created_at=100.0,
           candidate_rate=1.0, cost_delta=-75.6, verdict="no_significant_regression"):
    return {
        "benchmark": benchmark, "tokenjam_version": version, "created_at": created_at,
        "original_model": "deepseek:deepseek-reasoner",
        "candidate_model": "deepseek:deepseek-chat",
        "recommended_by": "explicit --candidate", "n_tasks": 20,
        "original_pass": 19, "candidate_pass": int(candidate_rate * 20),
        "original_pass_rate": 0.95, "candidate_pass_rate": candidate_rate,
        "accuracy_delta_pp": 5.0, "cost_delta_pct": cost_delta,
        "original_cost_usd": 0.015, "candidate_cost_usd": 0.0037,
        "mock": False, "priced_with_defaults": False, "output_token_inflation": 0.9,
        "stats": {
            "candidate_ci_pp": [84.0, 100.0], "delta_ci_pp": [-4.5, 14.6],
            "mcnemar_b": 0, "mcnemar_c": 1, "mcnemar_p_value": 1.0,
            "significant": False, "verdict": verdict, "samples_per_task": 1,
        },
    }


def test_migrations_run_and_record_is_idempotent(tmp_path):
    with BenchmarkHistory(tmp_path / "h.duckdb") as h:
        rid1 = h.record(_proof())
        rid2 = h.record(_proof())            # same identity → upsert, not duplicate
        assert rid1 == rid2 == run_id_for(_proof())
        assert h.count() == 1
        row = h.list_runs()[0]
        assert row["verdict"] == "no_significant_regression"
        assert row["wilson_low"] == 84.0 and row["wilson_high"] == 100.0
        assert row["mcnemar_c"] == 1


def test_deepeval_score_set_for_judge_benchmarks(tmp_path):
    with BenchmarkHistory(tmp_path / "h.duckdb") as h:
        h.record(_proof(benchmark="judged", candidate_rate=0.8))
        h.record(_proof(benchmark="humaneval", candidate_rate=1.0))
        by_b = {r["benchmark"]: r for r in h.list_runs()}
        assert by_b["judged"]["deepeval_score"] == 0.8     # judge pass-rate
        assert by_b["humaneval"]["deepeval_score"] is None  # executable → no judge score


def test_ingest_dir_counts_new_then_idempotent(tmp_path):
    rd = tmp_path / "results"
    rd.mkdir()
    (rd / "a.json").write_text(json.dumps(_proof(version="0.5.1", created_at=1)))
    (rd / "b.json").write_text(json.dumps(_proof(version="0.5.2", created_at=2)))
    (rd / "junk.json").write_text("{nope")
    with BenchmarkHistory(tmp_path / "h.duckdb") as h:
        new, total = h.ingest_dir(rd)
        assert (new, total) == (2, 2)
        new2, total2 = h.ingest_dir(rd)       # nothing new on re-ingest
        assert new2 == 0 and total2 == 2
        assert h.count() == 2


def test_versions_sorted_numerically_and_trend(tmp_path):
    with BenchmarkHistory(tmp_path / "h.duckdb") as h:
        h.record(_proof(version="0.5.10", created_at=3, candidate_rate=0.9))
        h.record(_proof(version="0.5.2", created_at=2, candidate_rate=1.0))
        h.record(_proof(version="0.5.1", created_at=1, candidate_rate=0.95))
        assert h.versions() == ["0.5.1", "0.5.2", "0.5.10"]   # 2 < 10, not lexical
        tr = h.trend("humaneval")
        assert [p["tokenjam_version"] for p in tr] == ["0.5.1", "0.5.2", "0.5.10"]  # by time
        assert tr[0]["candidate_pass_rate"] == 0.95


def test_configs(tmp_path):
    with BenchmarkHistory(tmp_path / "h.duckdb") as h:
        h.record(_proof(benchmark="humaneval", created_at=1))
        h.record(_proof(benchmark="judged", created_at=2))
        cfgs = {c["benchmark"] for c in h.configs()}
        assert cfgs == {"humaneval", "judged"}


def _pair(benchmark, version, ca, orig, origr, cand, candr, verdict="no_significant_regression"):
    return {"benchmark": benchmark, "tokenjam_version": version, "created_at": ca,
            "original_model": orig, "candidate_model": cand,
            "original_pass_rate": origr, "candidate_pass_rate": candr,
            "accuracy_delta_pp": (candr - origr) * 100, "cost_delta_pct": -80.0,
            "original_cost_usd": 0.02, "candidate_cost_usd": 0.004,
            "stats": {"verdict": verdict}}


def test_leaderboard_ranks_models_by_latest_pass_rate(tmp_path):
    with BenchmarkHistory(tmp_path / "h.duckdb") as h:
        h.record(_pair("humaneval", "0.5.1", 1, "a:opus", 0.95, "a:haiku", 0.90))
        h.record(_pair("humaneval", "0.5.2", 2, "a:opus", 0.95, "a:haiku", 0.85))  # latest haiku
        h.record(_pair("humaneval", "0.5.1", 3, "o:gpt5", 0.96, "o:mini", 0.93))
        lb = h.leaderboard("humaneval")
        ranked = [(r["model"], round(r["pass_rate"], 2)) for r in lb]
        assert ranked[0] == ("o:gpt5", 0.96)             # top
        assert ("a:haiku", 0.85) in ranked                # latest haiku (not 0.90)


def test_provider_matrix_aggregates_per_model(tmp_path):
    with BenchmarkHistory(tmp_path / "h.duckdb") as h:
        h.record(_pair("humaneval", "0.5.1", 1, "a:opus", 0.95, "a:haiku", 0.90))
        h.record(_pair("judged", "0.5.1", 2, "a:opus", 0.95, "a:haiku", 0.80))
        by_model = {r["model"]: r for r in h.provider_matrix()}
        assert by_model["a:opus"]["runs"] == 2
        assert by_model["a:haiku"]["benchmarks"] == 2
        assert abs(by_model["a:haiku"]["avg_accuracy"] - 0.85) < 1e-9


def test_version_summary_counts_regressions_correctly(tmp_path):
    with BenchmarkHistory(tmp_path / "h.duckdb") as h:
        h.record(_pair("humaneval", "0.5.1", 1, "a:opus", 0.95, "a:haiku", 0.90))
        h.record(_pair("humaneval", "0.5.2", 2, "a:opus", 0.95, "a:haiku", 0.70,
                       verdict="significant_regression"))
        by_v = {r["version"]: r for r in h.version_summary()}
        assert by_v["0.5.1"]["regressions"] == 0
        assert by_v["0.5.2"]["regressions"] == 1
        # 'no_significant_regression' must NOT count as a regression (substring trap)
        assert by_v["0.5.1"]["runs"] == 1


def test_regressions_timeline_only_flags_regressions(tmp_path):
    with BenchmarkHistory(tmp_path / "h.duckdb") as h:
        h.record(_pair("humaneval", "0.5.1", 1, "a:opus", 0.95, "a:haiku", 0.90))
        h.record(_pair("humaneval", "0.5.2", 2, "a:opus", 0.95, "a:haiku", 0.6,
                       verdict="significant_regression"))
        regs = h.regressions()
        assert len(regs) == 1 and regs[0]["verdict"] == "significant_regression"


def test_read_only_mode_can_query(tmp_path):
    db = tmp_path / "h.duckdb"
    with BenchmarkHistory(db) as h:                       # create + write
        h.record(_proof())
    with BenchmarkHistory(db, read_only=True) as ro:      # read-only open
        assert ro.count() == 1
        assert ro.versions() == ["0.5.1"]
