"""Historical benchmark database (P3).

Every proof artifact is an isolated file; this is the queryable index over all of
them. One DuckDB table, `benchmark_runs`, holds a row per ProofResult — version,
models, accuracy, Wilson CI, McNemar, cost, verdict, DeepEval score, and the
artifact paths — so trends, leaderboards, regression history, and the analytics
dashboard have a single source of truth.

DuckDB comes in via the `tokenjam` dependency (we never add it ourselves). The
schema uses an append-only MIGRATIONS list, mirroring TokenJam's own migration
pattern. `record()` is idempotent (deterministic run_id), so re-ingesting the
same results is safe.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import duckdb

DEFAULT_DB = "results/history.duckdb"

# Append-only migrations. Never edit a shipped one; add a new (version, sql).
MIGRATIONS: list[tuple[int, str]] = [
    (1, """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            run_id              TEXT PRIMARY KEY,
            created_at          DOUBLE NOT NULL,
            benchmark           TEXT NOT NULL,
            original_model      TEXT,
            candidate_model     TEXT,
            recommended_by      TEXT,
            tokenjam_version    TEXT,
            n_tasks             INTEGER,
            original_pass       INTEGER,
            candidate_pass      INTEGER,
            original_pass_rate  DOUBLE,
            candidate_pass_rate DOUBLE,
            accuracy_delta_pp   DOUBLE,
            cost_delta_pct      DOUBLE,
            original_cost_usd   DOUBLE,
            candidate_cost_usd  DOUBLE,
            wilson_low          DOUBLE,
            wilson_high         DOUBLE,
            delta_ci_low        DOUBLE,
            delta_ci_high       DOUBLE,
            mcnemar_b           INTEGER,
            mcnemar_c           INTEGER,
            mcnemar_p           DOUBLE,
            significant         BOOLEAN,
            verdict             TEXT,
            samples_per_task    INTEGER,
            mock                BOOLEAN,
            priced_with_defaults BOOLEAN,
            output_token_inflation DOUBLE,
            deepeval_score      DOUBLE,
            json_path           TEXT,
            html_path           TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_runs_benchmark ON benchmark_runs(benchmark);
        CREATE INDEX IF NOT EXISTS idx_runs_version   ON benchmark_runs(tokenjam_version);
    """),
]

# Benchmarks where the candidate pass-rate IS a judge/equivalence score.
_JUDGE_BENCHMARKS = {"judged", "replay"}


def run_id_for(d: dict) -> str:
    key = "|".join(str(d.get(k, "")) for k in (
        "benchmark", "original_model", "candidate_model", "tokenjam_version", "created_at"))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _version_key(v: str) -> tuple:
    parts: list[Any] = []
    for chunk in str(v).split("."):
        num = "".join(ch for ch in chunk if ch.isdigit())
        parts.append((0, int(num)) if num and num == chunk else (1, chunk))
    return tuple(parts)


def _run_migrations(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version INTEGER PRIMARY KEY, applied_at TIMESTAMP)")
    applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
    for version, sql in MIGRATIONS:
        if version not in applied:
            for stmt in sql.split(";"):
                if stmt.strip():
                    conn.execute(stmt)
            conn.execute("INSERT INTO schema_migrations VALUES (?, now())", [version])


_COLUMNS = [
    "run_id", "created_at", "benchmark", "original_model", "candidate_model",
    "recommended_by", "tokenjam_version", "n_tasks", "original_pass", "candidate_pass",
    "original_pass_rate", "candidate_pass_rate", "accuracy_delta_pp", "cost_delta_pct",
    "original_cost_usd", "candidate_cost_usd", "wilson_low", "wilson_high",
    "delta_ci_low", "delta_ci_high", "mcnemar_b", "mcnemar_c", "mcnemar_p",
    "significant", "verdict", "samples_per_task", "mock", "priced_with_defaults",
    "output_token_inflation", "deepeval_score", "json_path", "html_path",
]


class BenchmarkHistory:
    """Persistent DuckDB store of every benchmark run + the query layer."""

    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(p))
        _run_migrations(self._conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BenchmarkHistory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write --

    def record(self, d: dict, *, json_path: str | None = None,
               html_path: str | None = None) -> str:
        """Idempotently upsert one ProofResult dict; returns its run_id."""
        s = d.get("stats", {}) or {}
        ci = s.get("candidate_ci_pp", [None, None])
        dci = s.get("delta_ci_pp", [None, None])
        bench = d.get("benchmark")
        deepeval = d.get("candidate_pass_rate") if bench in _JUDGE_BENCHMARKS else None
        row = {
            "run_id": run_id_for(d),
            "created_at": float(d.get("created_at", 0.0)),
            "benchmark": bench,
            "original_model": d.get("original_model"),
            "candidate_model": d.get("candidate_model"),
            "recommended_by": d.get("recommended_by"),
            "tokenjam_version": d.get("tokenjam_version"),
            "n_tasks": d.get("n_tasks"),
            "original_pass": d.get("original_pass"),
            "candidate_pass": d.get("candidate_pass"),
            "original_pass_rate": d.get("original_pass_rate"),
            "candidate_pass_rate": d.get("candidate_pass_rate"),
            "accuracy_delta_pp": d.get("accuracy_delta_pp"),
            "cost_delta_pct": d.get("cost_delta_pct"),
            "original_cost_usd": d.get("original_cost_usd"),
            "candidate_cost_usd": d.get("candidate_cost_usd"),
            "wilson_low": ci[0], "wilson_high": ci[1],
            "delta_ci_low": dci[0], "delta_ci_high": dci[1],
            "mcnemar_b": s.get("mcnemar_b"), "mcnemar_c": s.get("mcnemar_c"),
            "mcnemar_p": s.get("mcnemar_p_value"), "significant": s.get("significant"),
            "verdict": s.get("verdict"), "samples_per_task": s.get("samples_per_task"),
            "mock": d.get("mock"), "priced_with_defaults": d.get("priced_with_defaults"),
            "output_token_inflation": d.get("output_token_inflation"),
            "deepeval_score": deepeval, "json_path": json_path, "html_path": html_path,
        }
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO benchmark_runs ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders})",
            [row[c] for c in _COLUMNS],
        )
        return row["run_id"]

    def ingest_dir(self, directory: str | Path) -> tuple[int, int]:
        """Record every *.json proof artifact in a dir. Returns (new, total)."""
        new = total = 0
        for jp in sorted(Path(directory).glob("*.json")):
            try:
                d = json.loads(jp.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if "tokenjam_version" not in d or "benchmark" not in d:
                continue
            total += 1
            rid = run_id_for(d)
            exists = self._conn.execute(
                "SELECT 1 FROM benchmark_runs WHERE run_id = ?", [rid]).fetchone()
            html = jp.with_suffix(".html")
            self.record(d, json_path=str(jp),
                        html_path=str(html) if html.exists() else None)
            if not exists:
                new += 1
        return new, total

    # -- read (the query layer) --

    def _rows(self, sql: str, params: list | None = None) -> list[dict]:
        cur = self._conn.execute(sql, params or [])
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def list_runs(self, benchmark: str | None = None, limit: int = 100) -> list[dict]:
        where = "WHERE benchmark = ?" if benchmark else ""
        params = ([benchmark] if benchmark else []) + [limit]
        return self._rows(
            f"SELECT * FROM benchmark_runs {where} ORDER BY created_at DESC LIMIT ?", params)

    def versions(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT tokenjam_version FROM benchmark_runs "
            "WHERE tokenjam_version IS NOT NULL").fetchall()
        return sorted((r[0] for r in rows), key=_version_key)

    def configs(self) -> list[dict]:
        return self._rows(
            "SELECT benchmark, original_model, candidate_model, COUNT(*) AS runs "
            "FROM benchmark_runs GROUP BY benchmark, original_model, candidate_model "
            "ORDER BY runs DESC")

    def trend(self, benchmark: str, original_model: str | None = None,
              candidate_model: str | None = None) -> list[dict]:
        clauses = ["benchmark = ?"]
        params: list[Any] = [benchmark]
        if original_model:
            clauses.append("original_model = ?")
            params.append(original_model)
        if candidate_model:
            clauses.append("candidate_model = ?")
            params.append(candidate_model)
        return self._rows(
            "SELECT created_at, tokenjam_version, candidate_pass_rate, accuracy_delta_pp, "
            "cost_delta_pct, deepeval_score, verdict FROM benchmark_runs "
            f"WHERE {' AND '.join(clauses)} ORDER BY created_at", params)

    def count(self) -> int:
        r = self._conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()
        return int(r[0]) if r else 0
