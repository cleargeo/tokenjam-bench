"""Bench & Evaluation dashboard — the bench's answer to TokenJam Lens.

`tjb serve` starts a local, offline, auto-refreshing dashboard over the
version-stamped proof artifacts in a results directory. It is NOT an
observability tool: every page answers one question — *can I trust TokenJam's
recommendations?* — with executable benchmarks and statistical validation.

Strict honesty: it shows only what was measured. No fabricated, extrapolated, or
placeholder values; no blanket reassurance about accuracy; no single confidence
scalar. The honest statement of confidence is the Wilson CI plus the McNemar
p-value and one of three hedged verdicts (no_significant_regression /
significant_regression / insufficient_evidence).

Information architecture: a small core of real pages (Overview, Benchmarks,
Leaderboards, Scenario Library, Regression Center, Reports, Settings). Pages that
need data the bench does not have yet (DeepEval judge sub-scores, multi-version
Trends / Version Comparison, Replay) are hidden from the nav until a real run
populates them — they are never shown as permanent empty pages, and never faked.

Offline-first (like TokenJam Lens): one self-contained page, inline CSS/JS, no
external HTTP, stdlib http.server only — no new dependencies, no charting lib.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tjbench.bench_meta import __version__ as BENCH_VERSION
from tjbench.report_html import render_html_from_dict


def _pair(seq, i):
    """Safe index into a [low, high] list that may be missing/short."""
    try:
        return seq[i]
    except (TypeError, IndexError, KeyError):
        return None


def scan_runs(directory: str | Path, include_dev: bool = False) -> list[dict[str, Any]]:
    """Summarize every proof artifact in `directory`, newest first.

    Carries the statistical block that already lives in each artifact (Wilson
    CIs, McNemar counts, measured costs) so the UI can render evidence-rich
    cards without any new query path or backend change.

    Production-only by default: mock (`--mock`/dev), demo (seeded-fixture), and
    archived (docs/evidence/archive/) artifacts are skipped so every dashboard
    number traces to a real measured run. Pass `include_dev=True` to include
    mock/demo runs (local development); archived artifacts stay non-headline
    either way.
    """
    runs: list[dict[str, Any]] = []
    for p in sorted(Path(directory).glob("*.json")):
        # Archived legacy artifacts are kept as a historical record but are never
        # headline/dashboard data — treat them as non-surfaced, like mock/demo.
        if any(part == "archive" for part in p.parts):
            continue
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "tokenjam_version" not in d or "benchmark" not in d:
            continue
        # Production dashboards show real measured runs only — never mock
        # (dev/--mock) or demo (seeded fixture) artifacts. include_dev=True opts in.
        if not include_dev and (d.get("mock") or d.get("demo")):
            continue
        s = d.get("stats", {}) or {}
        cand_ci = s.get("candidate_ci_pp") or []
        delta_ci = s.get("delta_ci_pp") or []
        row = {
            "file": p.name,
            "benchmark": d.get("benchmark", "?"),
            "original_model": d.get("original_model", "?"),
            "candidate_model": d.get("candidate_model", "?"),
            "recommended_by": d.get("recommended_by", ""),
            "tokenjam_version": d.get("tokenjam_version", "?"),
            "n_tasks": d.get("n_tasks", 0),
            "original_pass": d.get("original_pass"),
            "candidate_pass": d.get("candidate_pass"),
            "original_pass_rate": round(d.get("original_pass_rate", 0.0) * 100, 1),
            "candidate_pass_rate": round(d.get("candidate_pass_rate", 0.0) * 100, 1),
            "accuracy_delta_pp": d.get("accuracy_delta_pp", 0.0),
            "cost_delta_pct": d.get("cost_delta_pct", 0.0),
            "original_cost_usd": d.get("original_cost_usd"),
            "candidate_cost_usd": d.get("candidate_cost_usd"),
            "original_output_tokens": d.get("original_output_tokens"),
            "candidate_output_tokens": d.get("candidate_output_tokens"),
            "output_token_inflation": d.get("output_token_inflation"),
            "regressions": d.get("regressions"),
            "priced_with_defaults": d.get("priced_with_defaults", False),
            "samples_per_task": s.get("samples_per_task"),
            "wilson_low": _pair(cand_ci, 0),
            "wilson_high": _pair(cand_ci, 1),
            "delta_low": _pair(delta_ci, 0),
            "delta_high": _pair(delta_ci, 1),
            "mcnemar_b": s.get("mcnemar_b"),
            "mcnemar_c": s.get("mcnemar_c"),
            "mcnemar_p": s.get("mcnemar_p_value"),
            "significant": s.get("significant"),
            "verdict": s.get("verdict", "?"),
            "mock": d.get("mock", False),
            "created_at": d.get("created_at", 0.0),
        }
        # LLM-judge sub-scores (correctness / faithfulness / …), only if the
        # artifact actually carries them. The DeepEval page stays hidden until a
        # real judged run produces these — they are never synthesized.
        if isinstance(d.get("judge"), dict) and d["judge"]:
            row["judge"] = d["judge"]
        runs.append(row)
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs


def scenario_catalog() -> list[dict[str, Any]]:
    """Read-only catalog of registered scenario suites (name + task count).

    Lets the Scenario Library page list every suite — even ones with no runs
    yet — without guessing counts client-side. Purely structural; no stats.
    """
    out: list[dict[str, Any]] = []
    try:
        from tjbench.benchmarks.scenario_suites import SCENARIO_SUITES
    except Exception:
        return out
    for name, factory in SCENARIO_SUITES.items():
        try:
            suite = factory()
            n_tasks = len(suite.tasks())
            try:
                n_tools = len(suite.tools())
            except Exception:
                n_tools = None
            out.append({"name": name, "n_tasks": n_tasks, "n_tools": n_tools})
        except Exception:
            out.append({"name": name, "n_tasks": None, "n_tools": None})
    return out


def _hist_ro(db_path, fn, default):
    """Run a read-only history query; tolerate a missing/locked DB."""
    try:
        from tjbench.history import BenchmarkHistory
        with BenchmarkHistory(db_path, read_only=True) as h:
            return fn(h)
    except Exception:
        return default


def history_summary(db_path: str | Path) -> dict:
    """Read-only summary of the historical DB (count, versions, configs)."""
    import duckdb

    from tjbench.history import _version_key
    p = Path(db_path)
    empty = {"available": False, "count": 0, "versions": [], "configs": []}
    if not p.exists():
        return empty
    try:
        conn = duckdb.connect(str(p), read_only=True)
        try:
            count = conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
            vers = [r[0] for r in conn.execute(
                "SELECT DISTINCT tokenjam_version FROM benchmark_runs "
                "WHERE tokenjam_version IS NOT NULL").fetchall()]
            cfgs = [dict(zip(["benchmark", "original_model", "candidate_model", "runs"], r))
                    for r in conn.execute(
                        "SELECT benchmark, original_model, candidate_model, COUNT(*) "
                        "FROM benchmark_runs GROUP BY 1,2,3 ORDER BY 4 DESC").fetchall()]
        finally:
            conn.close()
        return {"available": True, "count": int(count),
                "versions": sorted(vers, key=_version_key), "configs": cfgs}
    except Exception:
        return empty


def serve(directory: str | Path = "results", host: str = "127.0.0.1",
          port: int = 7392) -> None:
    """Start the dashboard server (blocking until Ctrl-C)."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    # Index existing artifacts into the historical DB (best-effort).
    try:
        from tjbench.history import BenchmarkHistory
        with BenchmarkHistory(root / "history.duckdb") as _h:
            _h.ingest_dir(root)
    except Exception:
        pass

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            return

        def _send(self, body: bytes, ctype: str, status: int = 200,
                  extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(_DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/api/runs":
                self._send(json.dumps(scan_runs(root)).encode(), "application/json")
            elif path == "/api/scenarios":
                self._send(json.dumps({"rows": scenario_catalog()}).encode(),
                           "application/json")
            elif path == "/api/history":
                self._send(json.dumps(history_summary(root / "history.duckdb")).encode(),
                           "application/json")
            elif path == "/api/info":
                self._send(json.dumps({"directory": str(root),
                                       "version": BENCH_VERSION}).encode(),
                           "application/json")
            elif path.startswith("/raw/"):
                self._serve_raw(path)
            elif path.startswith("/api/"):
                self._send(json.dumps(self._analytics(path)).encode(), "application/json")
            elif path.startswith("/report/"):
                # Path-traversal safe: only a basename of an existing .json here.
                name = Path(path[len("/report/"):]).name
                target = root / name
                if name.endswith(".json") and target.is_file():
                    try:
                        d = json.loads(target.read_text())
                        self._send(render_html_from_dict(d).encode(),
                                   "text/html; charset=utf-8")
                    except (json.JSONDecodeError, OSError):
                        self._send(b"bad artifact", "text/plain", 500)
                else:
                    self._send(b"not found", "text/plain", 404)
            else:
                self._send(b"not found", "text/plain", 404)

        def do_DELETE(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path.startswith("/api/report/"):
                name = Path(path[len("/api/report/"):]).name
                target = root / name
                if name.endswith(".json") and target.is_file():
                    try:
                        target.unlink()
                        self._send(json.dumps({"deleted": name}).encode(),
                                   "application/json")
                    except OSError:
                        self._send(b"delete failed", "text/plain", 500)
                else:
                    self._send(b"not found", "text/plain", 404)
            else:
                self._send(b"not found", "text/plain", 404)

        def _serve_raw(self, path: str) -> None:
            from urllib.parse import parse_qs, urlparse
            name = Path(path[len("/raw/"):]).name
            target = root / name
            if not (name.endswith(".json") and target.is_file()):
                self._send(b"not found", "text/plain", 404)
                return
            try:
                body = target.read_bytes()
            except OSError:
                self._send(b"bad artifact", "text/plain", 500)
                return
            extra = {}
            if parse_qs(urlparse(self.path).query).get("download"):
                extra["Content-Disposition"] = f'attachment; filename="{name}"'
            self._send(body, "application/json", 200, extra)

        def _analytics(self, path: str) -> dict:
            from urllib.parse import parse_qs, urlparse
            db = root / "history.duckdb"
            q = parse_qs(urlparse(self.path).query)

            def one(k, default=None):
                v = q.get(k)
                return v[0] if v else default

            # History-backed analytics. These power the data-starved pages
            # (Trends / Version Comparison) that only surface once >=2 TokenJam
            # versions exist, plus the Regression Center and Settings.
            if path == "/api/version-summary":
                return {"rows": _hist_ro(db, lambda h: h.version_summary(), [])}
            if path == "/api/regressions":
                return {"rows": _hist_ro(db, lambda h: h.regressions(50), [])}
            if path == "/api/configs":
                return {"rows": _hist_ro(db, lambda h: h.configs(), [])}
            if path == "/api/trend":
                bm = one("benchmark", "")
                return {"benchmark": bm, "rows": _hist_ro(
                    db, lambda h: h.trend(bm, one("original"), one("candidate")), [])}
            return {}

    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print(f"tokenjam-bench dashboard → {url}  (serving {root}/ · Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()


_DASHBOARD_HTML = r"""<!doctype html><html lang=en data-theme=dark><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>TokenJam Bench</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjUwIDMwIDE1NiAxNTgiIGZpbGw9Im5vbmUiPjxzdHlsZT46cm9vdHtjb2xvcjojMTExfUBtZWRpYSAocHJlZmVycy1jb2xvci1zY2hlbWU6ZGFyayl7OnJvb3R7Y29sb3I6I2U4ZThlOH19PC9zdHlsZT4gPGcgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iNyIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj4gPHJlY3QgeD0iNzQiIHk9IjQ0IiB3aWR0aD0iOTIiIGhlaWdodD0iMjAiIHJ4PSI2IiByeT0iNiIvPiA8cGF0aCBkPSJNIDEwNCA3MiBMIDc0IDcyIEwgNzQgMTUyIEwgMTA0IDE1MiIvPiA8cGF0aCBkPSJNIDEzNiA3MiBMIDE2NiA3MiBMIDE2NiAxNTIgTCAxMzYgMTUyIi8+IDwvZz4gPGcgZmlsbD0iY3VycmVudENvbG9yIiBzdHJva2U9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+IDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDg1LjY4LDEwNCkiIGQ9Ik0yMS4xOSwtMTQuMzIgTDE2Ljc1LC0wLjAwIEwxMy42MywtMC4wMCBMMTAuNzQsLTEwLjU4IEw3Ljg1LC0wLjAwIEw0LjczLC0wLjAwIEwwLjI2LC0xNC4zMiBMMy4yNywtMTQuMzIgTDYuMjYsLTIuODEgTDkuMzEsLTE0LjMyIEwxMi40MCwtMTQuMzIgTDE1LjMyLC0yLjg2IEwxOC4yOCwtMTQuMzIgTDIxLjE5LC0xNC4zMiBaIi8+IDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEwNi44MywxMDcpIiBkPSJNMC41NCwtNy41MCBMMC41NCwtOC43NSBMMy4wNCwtOC43NSBMMy4wNCwtMC4wMCBMMS42NiwtMC4wMCBMMS42NiwtNy41MCBMMC41NCwtNy41MCBaIE00Ljc0LC03LjUwIEw0Ljc0LC04Ljc1IEw3LjI0LC04Ljc1IEw3LjI0LC0wLjAwIEw1Ljg2LC0wLjAwIEw1Ljg2LC03LjUwIEw0Ljc0LC03LjUwIFoiLz4gPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTIzLjk3LDEwNCkiIGQ9Ik0yMS4xOSwtMTQuMzIgTDE2Ljc1LC0wLjAwIEwxMy42MywtMC4wMCBMMTAuNzQsLTEwLjU4IEw3Ljg1LC0wLjAwIEw0LjczLC0wLjAwIEwwLjI2LC0xNC4zMiBMMy4yNywtMTQuMzIgTDYuMjYsLTIuODEgTDkuMzEsLTE0LjMyIEwxMi40MCwtMTQuMzIgTDE1LjMyLC0yLjg2IEwxOC4yOCwtMTQuMzIgTDIxLjE5LC0xNC4zMiBaIi8+IDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE0NS4xMiwxMDcpIiBkPSJNMC41NCwtNy41MCBMMC41NCwtOC43NSBMMy4wNCwtOC43NSBMMy4wNCwtMC4wMCBMMS42NiwtMC4wMCBMMS42NiwtNy41MCBMMC41NCwtNy41MCBaIE01LjY4LC0xLjg2IFE2LjgzLC0yLjg2IDcuNDksLTMuNTAgUTguMTYsLTQuMTQgOC42MCwtNC44NCBROS4wNSwtNS41MyA5LjA1LC02LjIzIFE5LjA1LC02Ljk1IDguNzEsLTcuMzYgUTguMzYsLTcuNzYgNy42MywtNy43NiBRNi45MiwtNy43NiA2LjUzLC03LjMxIFE2LjE0LC02Ljg2IDYuMTIsLTYuMTEgTDQuODAsLTYuMTEgUTQuODQsLTcuNDggNS42MiwtOC4yMCBRNi40MSwtOC45MyA3LjYyLC04LjkzIFE4LjkzLC04LjkzIDkuNjcsLTguMjEgUTEwLjQwLC03LjQ5IDEwLjQwLC02LjI5IFExMC40MCwtNS40MiA5Ljk3LC00LjYzIFE5LjUzLC0zLjgzIDguOTIsLTMuMjAgUTguMzIsLTIuNTcgNy4zOCwtMS43NCBMNi44NCwtMS4yNiBMMTAuNjQsLTEuMjYgTDEwLjY0LC0wLjEyIEw0LjgxLC0wLjEyIEw0LjgxLC0xLjEyIEw1LjY4LC0xLjg2IFoiLz4gPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODQuMzUsMTI4KSIgZD0iTTIxLjE5LC0xNC4zMiBMMTYuNzUsLTAuMDAgTDEzLjYzLC0wLjAwIEwxMC43NCwtMTAuNTggTDcuODUsLTAuMDAgTDQuNzMsLTAuMDAgTDAuMjYsLTE0LjMyIEwzLjI3LC0xNC4zMiBMNi4yNiwtMi44MSBMOS4zMSwtMTQuMzIgTDEyLjQwLC0xNC4zMiBMMTUuMzIsLTIuODYgTDE4LjI4LC0xNC4zMiBMMjEuMTksLTE0LjMyIFoiLz4gPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTA1LjQ0LDEzMSkiIGQ9Ik0xLjQ4LC0xLjg2IFEyLjYzLC0yLjg2IDMuMjksLTMuNTAgUTMuOTYsLTQuMTQgNC40MCwtNC44NCBRNC44NSwtNS41MyA0Ljg1LC02LjIzIFE0Ljg1LC02Ljk1IDQuNTEsLTcuMzYgUTQuMTYsLTcuNzYgMy40MywtNy43NiBRMi43MiwtNy43NiAyLjMzLC03LjMxIFExLjk0LC02Ljg2IDEuOTIsLTYuMTEgTDAuNjAsLTYuMTEgUTAuNjQsLTcuNDggMS40MiwtOC4yMCBRMi4yMSwtOC45MyAzLjQyLC04LjkzIFE0LjczLC04LjkzIDUuNDcsLTguMjEgUTYuMjAsLTcuNDkgNi4yMCwtNi4yOSBRNi4yMCwtNS40MiA1Ljc3LC00LjYzIFE1LjMzLC0zLjgzIDQuNzIsLTMuMjAgUTQuMTIsLTIuNTcgMy4xOCwtMS43NCBMMi42NCwtMS4yNiBMNi40NCwtMS4yNiBMNi40NCwtMC4xMiBMMC42MSwtMC4xMiBMMC42MSwtMS4xMiBMMS40OCwtMS44NiBaIE03LjQ2LC03LjUwIEw3LjQ2LC04Ljc1IEw5Ljk2LC04Ljc1IEw5Ljk2LC0wLjAwIEw4LjU4LC0wLjAwIEw4LjU4LC03LjUwIEw3LjQ2LC03LjUwIFoiLz4gPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTIyLjY0LDEyOCkiIGQ9Ik0yMS4xOSwtMTQuMzIgTDE2Ljc1LC0wLjAwIEwxMy42MywtMC4wMCBMMTAuNzQsLTEwLjU4IEw3Ljg1LC0wLjAwIEw0LjczLC0wLjAwIEwwLjI2LC0xNC4zMiBMMy4yNywtMTQuMzIgTDYuMjYsLTIuODEgTDkuMzEsLTE0LjMyIEwxMi40MCwtMTQuMzIgTDE1LjMyLC0yLjg2IEwxOC4yOCwtMTQuMzIgTDIxLjE5LC0xNC4zMiBaIi8+IDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE0My43MywxMzEpIiBkPSJNMS40OCwtMS44NiBRMi42MywtMi44NiAzLjI5LC0zLjUwIFEzLjk2LC00LjE0IDQuNDAsLTQuODQgUTQuODUsLTUuNTMgNC44NSwtNi4yMyBRNC44NSwtNi45NSA0LjUxLC03LjM2IFE0LjE2LC03Ljc2IDMuNDMsLTcuNzYgUTIuNzIsLTcuNzYgMi4zMywtNy4zMSBRMS45NCwtNi44NiAxLjkyLC02LjExIEwwLjYwLC02LjExIFEwLjY0LC03LjQ4IDEuNDIsLTguMjAgUTIuMjEsLTguOTMgMy40MiwtOC45MyBRNC43MywtOC45MyA1LjQ3LC04LjIxIFE2LjIwLC03LjQ5IDYuMjAsLTYuMjkgUTYuMjAsLTUuNDIgNS43NywtNC42MyBRNS4zMywtMy44MyA0LjcyLC0zLjIwIFE0LjEyLC0yLjU3IDMuMTgsLTEuNzQgTDIuNjQsLTEuMjYgTDYuNDQsLTEuMjYgTDYuNDQsLTAuMTIgTDAuNjEsLTAuMTIgTDAuNjEsLTEuMTIgTDEuNDgsLTEuODYgWiBNOC40MCwtMS44NiBROS41NSwtMi44NiAxMC4yMiwtMy41MCBRMTAuODgsLTQuMTQgMTEuMzMsLTQuODQgUTExLjc3LC01LjUzIDExLjc3LC02LjIzIFExMS43NywtNi45NSAxMS40MywtNy4zNiBRMTEuMDksLTcuNzYgMTAuMzYsLTcuNzYgUTkuNjUsLTcuNzYgOS4yNiwtNy4zMSBROC44NywtNi44NiA4Ljg0LC02LjExIEw3LjUyLC02LjExIFE3LjU2LC03LjQ4IDguMzUsLTguMjAgUTkuMTMsLTguOTMgMTAuMzQsLTguOTMgUTExLjY1LC04LjkzIDEyLjM5LC04LjIxIFExMy4xMywtNy40OSAxMy4xMywtNi4yOSBRMTMuMTMsLTUuNDIgMTIuNjksLTQuNjMgUTEyLjI1LC0zLjgzIDExLjY1LC0zLjIwIFExMS4wNCwtMi41NyAxMC4xMCwtMS43NCBMOS41NiwtMS4yNiBMMTMuMzcsLTEuMjYgTDEzLjM3LC0wLjEyIEw3LjU0LC0wLjEyIEw3LjU0LC0xLjEyIEw4LjQwLC0xLjg2IFoiLz4gPC9nPiA8Y2lyY2xlIGN4PSIxNzIiIGN5PSIxNjAiIHI9IjIzIiBmaWxsPSIjMTZhMzRhIi8+IDxwYXRoIGQ9Ik0xNjEgMTYxIEwxNjkgMTY5IEwxODUgMTQ5IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iNi41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4gPC9zdmc+">
<style>
:root{
 --bg:#000; --surface:#0a0a0a; --surface2:#111; --border:#1f1f1f; --border2:#2a2a2a;
 --text:#ededed; --dim:#a1a1a1; --dim2:#6b6b6b; --accent:#ededed; --warn:#f5a623;
 --ok:#0ce490; --bad:#ff5c5c;
 --c1:#3d8eff; --c2:#0ce490; --c3:#f5a623; --c4:#b36bff; --c5:#ff6b9d; --c6:#2dd4bf;
 --c7:#ff5c5c; --c8:#a3e635; --c9:#818cf8; --c10:#fb923c; --c11:#e879f9; --c12:#38bdf8;
 --radius:10px;
 --font:"Geist",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
 --mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
[data-theme=light]{
 --bg:#fff; --surface:#fafafa; --surface2:#f3f3f3; --border:#eaeaea; --border2:#dcdcdc;
 --text:#000; --dim:#666; --dim2:#999; --accent:#000; --warn:#9a6700;
 --ok:#0a7d54; --bad:#cf2330;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 var(--font);
 -webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.mono{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums}
.muted{color:var(--dim)}
.app{display:flex;min-height:100vh}
/* sidebar */
.side{width:236px;flex:0 0 236px;position:sticky;top:0;height:100vh;display:flex;
 flex-direction:column;background:var(--bg);border-right:1px solid var(--border);padding:18px 12px}
.brand{display:flex;align-items:center;gap:10px;padding:4px 8px 18px;font-weight:600;font-size:14px}
.brand .glyph{width:30px;height:30px;flex:0 0 auto;color:var(--text)}
.brand .glyph svg{width:100%;height:100%;display:block}
.brand small{display:block;color:var(--dim);font-weight:450;font-size:11px;letter-spacing:.02em}
.nav{display:flex;flex-direction:column;gap:1px;overflow:auto}
.navsec{font-size:10.5px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
 color:var(--dim2);padding:16px 10px 6px}
.nav a{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;color:var(--dim);
 font-size:13.5px;font-weight:450;border-left:2px solid transparent;cursor:pointer;
 transition:background .12s,color .12s}
.nav a .ic{width:17px;height:17px;flex:0 0 auto;display:grid;place-items:center;color:var(--dim2)}
.nav a .ic svg{width:17px;height:17px}
.nav a:hover{background:var(--surface);color:var(--text)}
.nav a:hover .ic{color:var(--text)}
.nav a.active{background:var(--surface);color:var(--text);font-weight:550;border-left-color:var(--accent)}
.nav a.active .ic{color:var(--text)}
.side-foot{margin-top:auto;padding-top:12px;border-top:1px solid var(--border);
 display:flex;flex-direction:column;gap:8px}
.foot-row{display:flex;align-items:center;justify-content:space-between;color:var(--dim2);font-size:11px;padding:0 8px}
.tbtn{cursor:pointer;border:1px solid var(--border2);background:var(--surface);color:var(--dim);
 border-radius:7px;padding:5px 9px;font-size:11.5px;transition:.12s;display:inline-flex;align-items:center;gap:6px}
.tbtn:hover{color:var(--text);border-color:var(--accent)}
.tbtn svg{width:13px;height:13px}
/* main */
.main{flex:1;min-width:0;display:flex;flex-direction:column}
.top{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:12px;
 padding:16px 26px;background:var(--bg);border-bottom:1px solid var(--border)}
.top h1{font-size:18px;margin:0;font-weight:600;letter-spacing:-.01em}
.chip{font-size:11.5px;color:var(--dim);background:var(--surface);border:1px solid var(--border);
 border-radius:999px;padding:3px 10px;font-weight:500}
.spacer{flex:1}
.ctrls{display:flex;align-items:center;gap:8px}
.view{padding:22px 26px 64px}
.lead{color:var(--dim);margin:-2px 0 18px;font-size:13.5px;max-width:780px}
.sect{font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim2);
 margin:26px 0 12px;font-weight:600}
.sect:first-child{margin-top:0}
.sect .sub{text-transform:none;letter-spacing:0;font-weight:400;color:var(--dim2)}
/* controls */
select,button.btn,input.in{background:var(--surface);color:var(--text);border:1px solid var(--border2);
 border-radius:8px;padding:7px 11px;font-size:13px;font-family:inherit}
button.btn{cursor:pointer;transition:.12s}
button.btn:hover{border-color:var(--accent)}
select:focus,input.in:focus,button:focus{outline:none;border-color:var(--accent)}
/* grids + cards */
.grid{display:grid;gap:13px}
.g2{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(3,1fr)}
.g4{grid-template-columns:repeat(4,1fr)}
.auto{grid-template-columns:repeat(auto-fill,minmax(230px,1fr))}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:15px 17px}
.stat .lbl{color:var(--dim);font-size:12px;font-weight:450}
.stat .num{font-size:24px;font-weight:600;letter-spacing:-.02em;margin-top:6px;line-height:1.1}
.stat .sub{color:var(--dim2);font-size:11.5px;margin-top:3px}
/* banners */
.banner{display:flex;align-items:center;gap:12px;border-radius:var(--radius);padding:13px 16px;
 border:1px solid var(--border);background:var(--surface);font-size:13.5px}
.banner .bi{width:26px;height:26px;flex:0 0 auto;display:grid;place-items:center}
.banner .bi svg{width:18px;height:18px}
.banner.ok .bi{color:var(--ok)}.banner.bad .bi{color:var(--bad)}.banner.info .bi{color:var(--dim)}
.banner.ok{border-color:color-mix(in srgb,var(--ok) 32%,var(--border))}
.banner.bad{border-color:color-mix(in srgb,var(--bad) 36%,var(--border))}
.banner b{font-weight:600}.banner .bsub{color:var(--dim);font-size:12.5px}
/* badges */
.badge{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:500;
 padding:3px 9px;border-radius:999px;border:1px solid var(--border2);color:var(--dim);white-space:nowrap}
.badge::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.b-ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 30%,transparent)}
.b-bad{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 30%,transparent)}
.b-mut{color:var(--dim)}
.tag{font-size:10.5px;color:var(--dim);border:1px solid var(--border2);border-radius:6px;
 padding:1px 6px;text-transform:uppercase;letter-spacing:.03em}
.delta{font-weight:550;font-variant-numeric:tabular-nums}.ok{color:var(--ok)}.bad{color:var(--bad)}
/* tables */
.tblbar{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.tblbar .in{flex:0 0 auto;width:240px;max-width:60%}
.tblcount{color:var(--dim2);font-size:12px;margin-left:auto}
.tblscroll{overflow:auto;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface)}
table.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl thead th{position:sticky;top:0;background:var(--surface2);color:var(--dim);text-align:left;
 font-weight:550;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em;
 padding:10px 14px;border-bottom:1px solid var(--border);white-space:nowrap;z-index:1}
.tbl th.srt{cursor:pointer;user-select:none}.tbl th.srt:hover{color:var(--text)}
.tbl th.on{color:var(--text)}
.tbl td{padding:10px 14px;border-bottom:1px solid var(--border);vertical-align:middle}
.tbl tbody tr:hover{background:var(--surface2)}
.tbl tbody tr:last-child td{border-bottom:none}
.empty{color:var(--dim);text-align:center;padding:30px}
.acts{display:flex;gap:6px;flex-wrap:wrap}
a.lnk,button.lnk{color:var(--text);border:1px solid var(--border2);
 border-radius:7px;padding:3px 9px;font-size:12px;background:transparent;cursor:pointer;
 font-family:inherit;transition:.12s}
a.lnk:hover,button.lnk:hover{border-color:var(--accent)}
button.lnk.danger{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 30%,transparent)}
button.lnk.danger:hover{border-color:var(--bad)}
/* charts */
.chart{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:15px 17px}
.chart h3{margin:0 0 3px;font-size:13.5px;font-weight:550}
.chart .ch-sub{color:var(--dim);font-size:12px;margin:0 0 12px}
.legend{display:flex;gap:16px;font-size:12px;color:var(--dim);margin-top:8px;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:middle}
/* horizontal bars */
.hbars{display:flex;flex-direction:column;gap:9px}
.hbar{display:grid;grid-template-columns:150px 1fr 64px;align-items:center;gap:12px;font-size:12.5px}
.hbar-l{color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hbar-track{height:8px;border-radius:6px;background:var(--surface2);overflow:hidden}
.hbar-track > i{display:block;height:100%;border-radius:6px}
.hbar-v{text-align:right;color:var(--text);font-variant-numeric:tabular-nums}
/* mini kv rows */
.kv{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:13px}
.kv .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
.kv .v{font-size:17px;font-weight:600;margin-top:2px}
.kv .v.sm{font-size:13px;font-weight:500}
.cardrow{display:flex;flex-wrap:wrap;align-items:center;gap:8px}
.cardttl{font-weight:600;font-size:15px;white-space:nowrap}
.cardmeta{color:var(--dim);font-size:12px;margin-top:3px}
.brk{border-top:1px solid var(--border);margin-top:12px;padding-top:11px;display:flex;gap:16px;flex-wrap:wrap;font-size:11.5px;color:var(--dim)}
.brk b{color:var(--text)}
/* settings */
.set-row{display:flex;align-items:center;justify-content:space-between;gap:16px;
 padding:14px 0;border-bottom:1px solid var(--border)}
.set-row:last-child{border-bottom:none}
.set-row .k{font-weight:500}.set-row .d{color:var(--dim);font-size:12.5px;margin-top:2px}
/* warn note */
.note{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--warn);margin-top:10px}
.note svg{width:14px;height:14px;flex:0 0 auto}
@media(max-width:1080px){.g4{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(2,1fr)}}
@media(max-width:760px){.side{display:none}.g2{grid-template-columns:1fr}.hbar{grid-template-columns:110px 1fr 54px}}
</style></head><body>
<div class=app>
 <aside class=side>
  <div class=brand><span class=glyph><svg viewBox="50 30 156 158" fill="none"> <g stroke="currentColor" stroke-width="7" stroke-linecap="round" stroke-linejoin="round"> <rect x="74" y="44" width="92" height="20" rx="6" ry="6"/> <path d="M 104 72 L 74 72 L 74 152 L 104 152"/> <path d="M 136 72 L 166 72 L 166 152 L 136 152"/> </g> <g fill="currentColor" stroke="none" fill-rule="evenodd"> <path transform="translate(85.68,104)" d="M21.19,-14.32 L16.75,-0.00 L13.63,-0.00 L10.74,-10.58 L7.85,-0.00 L4.73,-0.00 L0.26,-14.32 L3.27,-14.32 L6.26,-2.81 L9.31,-14.32 L12.40,-14.32 L15.32,-2.86 L18.28,-14.32 L21.19,-14.32 Z"/> <path transform="translate(106.83,107)" d="M0.54,-7.50 L0.54,-8.75 L3.04,-8.75 L3.04,-0.00 L1.66,-0.00 L1.66,-7.50 L0.54,-7.50 Z M4.74,-7.50 L4.74,-8.75 L7.24,-8.75 L7.24,-0.00 L5.86,-0.00 L5.86,-7.50 L4.74,-7.50 Z"/> <path transform="translate(123.97,104)" d="M21.19,-14.32 L16.75,-0.00 L13.63,-0.00 L10.74,-10.58 L7.85,-0.00 L4.73,-0.00 L0.26,-14.32 L3.27,-14.32 L6.26,-2.81 L9.31,-14.32 L12.40,-14.32 L15.32,-2.86 L18.28,-14.32 L21.19,-14.32 Z"/> <path transform="translate(145.12,107)" d="M0.54,-7.50 L0.54,-8.75 L3.04,-8.75 L3.04,-0.00 L1.66,-0.00 L1.66,-7.50 L0.54,-7.50 Z M5.68,-1.86 Q6.83,-2.86 7.49,-3.50 Q8.16,-4.14 8.60,-4.84 Q9.05,-5.53 9.05,-6.23 Q9.05,-6.95 8.71,-7.36 Q8.36,-7.76 7.63,-7.76 Q6.92,-7.76 6.53,-7.31 Q6.14,-6.86 6.12,-6.11 L4.80,-6.11 Q4.84,-7.48 5.62,-8.20 Q6.41,-8.93 7.62,-8.93 Q8.93,-8.93 9.67,-8.21 Q10.40,-7.49 10.40,-6.29 Q10.40,-5.42 9.97,-4.63 Q9.53,-3.83 8.92,-3.20 Q8.32,-2.57 7.38,-1.74 L6.84,-1.26 L10.64,-1.26 L10.64,-0.12 L4.81,-0.12 L4.81,-1.12 L5.68,-1.86 Z"/> <path transform="translate(84.35,128)" d="M21.19,-14.32 L16.75,-0.00 L13.63,-0.00 L10.74,-10.58 L7.85,-0.00 L4.73,-0.00 L0.26,-14.32 L3.27,-14.32 L6.26,-2.81 L9.31,-14.32 L12.40,-14.32 L15.32,-2.86 L18.28,-14.32 L21.19,-14.32 Z"/> <path transform="translate(105.44,131)" d="M1.48,-1.86 Q2.63,-2.86 3.29,-3.50 Q3.96,-4.14 4.40,-4.84 Q4.85,-5.53 4.85,-6.23 Q4.85,-6.95 4.51,-7.36 Q4.16,-7.76 3.43,-7.76 Q2.72,-7.76 2.33,-7.31 Q1.94,-6.86 1.92,-6.11 L0.60,-6.11 Q0.64,-7.48 1.42,-8.20 Q2.21,-8.93 3.42,-8.93 Q4.73,-8.93 5.47,-8.21 Q6.20,-7.49 6.20,-6.29 Q6.20,-5.42 5.77,-4.63 Q5.33,-3.83 4.72,-3.20 Q4.12,-2.57 3.18,-1.74 L2.64,-1.26 L6.44,-1.26 L6.44,-0.12 L0.61,-0.12 L0.61,-1.12 L1.48,-1.86 Z M7.46,-7.50 L7.46,-8.75 L9.96,-8.75 L9.96,-0.00 L8.58,-0.00 L8.58,-7.50 L7.46,-7.50 Z"/> <path transform="translate(122.64,128)" d="M21.19,-14.32 L16.75,-0.00 L13.63,-0.00 L10.74,-10.58 L7.85,-0.00 L4.73,-0.00 L0.26,-14.32 L3.27,-14.32 L6.26,-2.81 L9.31,-14.32 L12.40,-14.32 L15.32,-2.86 L18.28,-14.32 L21.19,-14.32 Z"/> <path transform="translate(143.73,131)" d="M1.48,-1.86 Q2.63,-2.86 3.29,-3.50 Q3.96,-4.14 4.40,-4.84 Q4.85,-5.53 4.85,-6.23 Q4.85,-6.95 4.51,-7.36 Q4.16,-7.76 3.43,-7.76 Q2.72,-7.76 2.33,-7.31 Q1.94,-6.86 1.92,-6.11 L0.60,-6.11 Q0.64,-7.48 1.42,-8.20 Q2.21,-8.93 3.42,-8.93 Q4.73,-8.93 5.47,-8.21 Q6.20,-7.49 6.20,-6.29 Q6.20,-5.42 5.77,-4.63 Q5.33,-3.83 4.72,-3.20 Q4.12,-2.57 3.18,-1.74 L2.64,-1.26 L6.44,-1.26 L6.44,-0.12 L0.61,-0.12 L0.61,-1.12 L1.48,-1.86 Z M8.40,-1.86 Q9.55,-2.86 10.22,-3.50 Q10.88,-4.14 11.33,-4.84 Q11.77,-5.53 11.77,-6.23 Q11.77,-6.95 11.43,-7.36 Q11.09,-7.76 10.36,-7.76 Q9.65,-7.76 9.26,-7.31 Q8.87,-6.86 8.84,-6.11 L7.52,-6.11 Q7.56,-7.48 8.35,-8.20 Q9.13,-8.93 10.34,-8.93 Q11.65,-8.93 12.39,-8.21 Q13.13,-7.49 13.13,-6.29 Q13.13,-5.42 12.69,-4.63 Q12.25,-3.83 11.65,-3.20 Q11.04,-2.57 10.10,-1.74 L9.56,-1.26 L13.37,-1.26 L13.37,-0.12 L7.54,-0.12 L7.54,-1.12 L8.40,-1.86 Z"/> </g> <circle cx="172" cy="160" r="28" style="fill:var(--bg)"/> <circle cx="172" cy="160" r="23" fill="#16a34a"/> <path d="M161 161 L169 169 L185 149" fill="none" stroke="#ffffff" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round"/> </svg></span><div>TokenJam Bench<small>Benchmark &amp; Evaluate</small></div></div>
  <nav class=nav id=nav></nav>
  <div class=side-foot>
   <div class=foot-row><span id=ver>tjb &middot;&middot;&middot;</span>
    <span class=tbtn id=themeBtn><svg viewBox="0 0 24 24" fill=none stroke=currentColor stroke-width=1.8 stroke-linecap=round stroke-linejoin=round><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>Theme</span></div>
  </div>
 </aside>
 <main class=main>
  <header class=top>
   <h1 id=title>Overview</h1>
   <span class=chip id=ctxchip></span>
   <div class=spacer></div>
   <div class=ctrls id=ctrls></div>
  </header>
  <section class=view id=view><div class=empty>loading&hellip;</div></section>
 </main>
</div>
<script>
"use strict";
// ---- nav definition (filtered by data availability at boot) ----------------
const NAV_DEF=[
 ["Evidence",[["overview","Overview"],["benchmarks","Benchmarks"],["leaderboards","Leaderboards"],
   ["scenarios","Scenario Library"],["regressions","Regression Center"]]],
 ["Analysis",[["deepeval","DeepEval"],["trends","Trends"],["versions","Version Comparison"],["replay","Replay"]]],
 ["Workspace",[["reports","Reports"],["settings","Settings"]]]];
const LABEL=Object.fromEntries(NAV_DEF.flatMap(g=>g[1]));
const _IC='<svg viewBox="0 0 24 24" fill=none stroke=currentColor stroke-width=1.8 stroke-linecap=round stroke-linejoin=round>';
const ICONS={
 overview:_IC+'<rect x=3 y=3 width=7 height=9 rx=1 /><rect x=14 y=3 width=7 height=5 rx=1 /><rect x=14 y=12 width=7 height=9 rx=1 /><rect x=3 y=16 width=7 height=5 rx=1 /></svg>',
 benchmarks:_IC+'<line x1=6 y1=20 x2=6 y2=14 /><line x1=12 y1=20 x2=12 y2=4 /><line x1=18 y1=20 x2=18 y2=9 /></svg>',
 leaderboards:_IC+'<line x1=4 y1=20 x2=20 y2=20 /><rect x=5 y=12 width=4 height=8 /><rect x=10 y=7 width=4 height=13 /><rect x=15 y=14 width=4 height=6 /></svg>',
 scenarios:_IC+'<rect x=4 y=8 width=16 height=12 rx=2 /><path d="M12 8V5"/><circle cx=9 cy=14 r=1 /><circle cx=15 cy=14 r=1 /></svg>',
 regressions:_IC+'<path d="M22 17 13.5 8.5l-5 5L2 7"/><path d="M16 17h6v-6"/></svg>',
 deepeval:_IC+'<circle cx=12 cy=12 r=9 /><path d="m9 12 2 2 4-4"/></svg>',
 trends:_IC+'<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
 versions:_IC+'<line x1=6 y1=3 x2=6 y2=15 /><circle cx=18 cy=6 r=3 /><circle cx=6 cy=18 r=3 /><path d="M18 9a9 9 0 0 1-9 9"/></svg>',
 replay:_IC+'<path d="M21 12a9 9 0 1 1-3-6.7L21 8"/><path d="M21 3v5h-5"/></svg>',
 reports:_IC+'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8"/><path d="M16 17H8"/></svg>',
 settings:_IC+'<circle cx=12 cy=12 r=3 /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'};
const BIc='<svg viewBox="0 0 24 24" fill=none stroke=currentColor stroke-width=2.2 stroke-linecap=round stroke-linejoin=round>';
const BI={ok:BIc+'<path d="M20 6 9 17l-5-5"/></svg>',
 bad:BIc+'<path d="M10.3 3.3 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.3a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
 info:_IC+'<circle cx=12 cy=12 r=9 /><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>'};
// ---- verdict semantics (the three honest hedged verdicts only) -------------
const GOOD=new Set(["no_significant_regression"]);
const BAD=new Set(["significant_regression"]);
function vclass(v){return GOOD.has(v)?"b-ok":BAD.has(v)?"b-bad":"b-mut";}
function badge(v){return '<span class="badge '+vclass(v)+'">'+esc(String(v||"?").replace(/_/g," "))+'</span>';}
// ---- prefs -----------------------------------------------------------------
const PREF={get(k,d){try{const v=localStorage.getItem("tjb."+k);return v==null?d:v;}catch(e){return d;}},
 set(k,v){try{localStorage.setItem("tjb."+k,v);}catch(e){}}};
// ---- helpers ---------------------------------------------------------------
const M=()=>document.getElementById("view");
function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function code(s){return '<span class=mono>'+esc(s)+'</span>';}
function fmtTime(ts){if(!ts)return"—";return new Date(ts*1000).toLocaleString([],{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"});}
function ago(ts){if(!ts)return"—";const s=Date.now()/1000-ts;if(s<60)return"just now";
 if(s<3600)return Math.floor(s/60)+"m ago";if(s<86400)return Math.floor(s/3600)+"h ago";return Math.floor(s/86400)+"d ago";}
async function getJSON(u){try{const r=await fetch(u);if(!r.ok)return null;return await r.json();}catch(e){return null;}}
function provOf(m){return String(m||"").split(":")[0]||"?";}
function modelOf(m){const p=String(m||"").split(":");return p.length>1?p.slice(1).join(":"):p[0];}
const PROVLBL={anthropic:"Anthropic",openai:"OpenAI",deepseek:"DeepSeek",google:"Gemini",gemini:"Gemini",groq:"Groq",openrouter:"OpenRouter"};
function provLabel(p){return PROVLBL[p]||p;}
function pct(x){return x==null?"—":(Math.round(x*10)/10)+"%";}
function pp(x){return x==null?"—":(x>=0?"+":"")+(Math.round(x*10)/10)+"pp";}
function usd(x){if(x==null)return"—";const a=Math.abs(x);
 if(a>=1000)return"$"+Math.round(x).toLocaleString();
 if(a>=1)return"$"+x.toFixed(2);
 return"$"+x.toFixed(a<0.01?5:4);}
function fmtTok(n){if(n==null)return"—";n=Math.round(n);
 if(n>=1e6)return(n/1e6).toFixed(n>=1e7?0:1)+"M";if(n>=1e3)return(n/1e3).toFixed(n>=1e5?0:1)+"k";return String(n);}
function avg(xs){const v=xs.filter(x=>x!=null&&!isNaN(x));return v.length?v.reduce((a,b)=>a+b,0)/v.length:null;}
// cost_delta_pct: negative = candidate cheaper (good). Measured, never extrapolated.
function costCell(x){if(x==null)return'<span class=muted>—</span>';
 const cls=x<0?"ok":(x>0?"bad":"");return '<span class="delta '+cls+'">'+(x>0?"+":"")+(Math.round(x*10)/10)+'%</span>';}
function accDelta(x){if(x==null)return"—";const cls=x>0?"ok":(x<0?"bad":"");
 return '<span class="delta '+cls+'">'+pp(x)+'</span>';}
// statistical-confidence cell: McNemar p + delta CI (no single scalar)
function conf(r){
 if(r.mcnemar_p==null&&r.delta_low==null)return '<span class=muted>—</span>';
 const p=r.mcnemar_p==null?"":"p="+Number(r.mcnemar_p).toFixed(3);
 const ci=(r.delta_low==null||r.delta_high==null)?"":'<span class=muted>CI ['+pp(r.delta_low)+", "+pp(r.delta_high)+"]</span>";
 return '<div class=mono style="font-size:12px">'+p+'</div>'+(ci?'<div style="font-size:11px">'+ci+'</div>':"");}
function statCard(num,lbl,sub){
 return '<div class="card stat"><div class=lbl>'+esc(lbl)+'</div><div class=num>'+num+'</div>'+
  (sub?'<div class=sub>'+sub+'</div>':"")+'</div>';}
function defl(r){return r.priced_with_defaults?' <span class=tag title="cost used TokenJam default placeholder rates">default rates</span>':"";}
// ---- inline-SVG charts (no library; monochrome chrome, color for data) ------
// NOTE: CSS var() resolves only in inline style="" inside SVG, not in
// presentation attributes — so every themed stroke/fill goes through style="".
function drawChart(id,pts,opts){ // line(s) over a 0..100 scale; emits <polyline>
 const box=document.getElementById(id);if(!box)return;opts=opts||{};
 if(!pts||!pts.length){box.innerHTML='<div class=empty>no data in range</div>';return;}
 const W=1040,H=190,padL=34,padR=14,padT=14,padB=26,n=pts.length;
 const iw=W-padL-padR,ih=H-padT-padB;
 const X=i=>n<=1?padL+iw/2:padL+(i/(n-1))*iw;
 const Y=v=>padT+(1-Math.max(0,Math.min(100,v))/100)*ih;
 const line=(g,col)=>{const has=pts.some(p=>g(p)!=null);if(!has)return"";
  const pl=pts.map((p,i)=>X(i).toFixed(1)+","+Y(g(p)||0).toFixed(1)).join(" ");
  const dots=pts.map((p,i)=>'<circle cx="'+X(i).toFixed(1)+'" cy="'+Y(g(p)||0).toFixed(1)+'" r=2.6 style="fill:'+col+'" />').join("");
  return '<polyline points="'+pl+'" style="fill:none;stroke:'+col+'" stroke-width=2 stroke-linejoin=round />'+dots;};
 let grid="";[0,25,50,75,100].forEach(v=>{const y=Y(v).toFixed(1);
  grid+='<line x1="'+padL+'" y1="'+y+'" x2="'+(W-padR)+'" y2="'+y+'" style="stroke:var(--border)" stroke-width=1 />'+
   '<text x=4 y="'+(+y+3).toFixed(1)+'" style="fill:var(--dim2)" font-size=10>'+v+'</text>';});
 let xl="";const step=Math.max(1,Math.floor(n/8));
 for(let i=0;i<n;i+=step){const lab=pts[i].x||"";xl+='<text x="'+X(i).toFixed(1)+'" y="'+(H-8)+'" style="fill:var(--dim2)" font-size=10 text-anchor=middle>'+esc(lab)+'</text>';}
 const second=opts.single?"":line(p=>p.a,"var(--c1)");
 box.innerHTML='<svg width=100% viewBox="0 0 '+W+' '+H+'" preserveAspectRatio=none style="max-width:100%;height:190px">'+
  grid+line(p=>p.c,"var(--c2)")+second+xl+'</svg>';}
function barChart(id,items){
 const box=document.getElementById(id);if(!box)return;
 if(!items||!items.length){box.innerHTML='<div class=empty>no data</div>';return;}
 const W=1040,H=200,padL=34,padR=14,padT=14,n=items.length;
 const iw=W-padL-padR,ih=H-padT-34,mx=Math.max(...items.map(d=>d.value||0),1);
 const bw=Math.min(54,(iw/n)*0.62),gap=iw/n;
 const Y=v=>padT+(1-(v/mx))*ih;
 let bars="";items.forEach((d,i)=>{const x=padL+gap*i+(gap-bw)/2,y=Y(d.value||0),h=padT+ih-y;
  bars+='<rect x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(0,h).toFixed(1)+'" rx=4 style="fill:'+(d.color||"var(--c1)")+'" />'+
   '<text x="'+(x+bw/2).toFixed(1)+'" y="'+(H-12)+'" style="fill:var(--dim2)" font-size=10 text-anchor=middle>'+esc(d.label)+'</text>';});
 let grid="";[0,.5,1].forEach(f=>{const y=Y(mx*f).toFixed(1);grid+='<line x1="'+padL+'" y1="'+y+'" x2="'+(W-padR)+'" y2="'+y+'" style="stroke:var(--border)" stroke-width=1 />';});
 box.innerHTML='<svg width=100% viewBox="0 0 '+W+' '+H+'" style="max-width:100%;height:200px">'+grid+bars+'</svg>';}
function hbars(items,opts){
 opts=opts||{};if(!items||!items.length)return'<div class=empty>none</div>';
 const mx=Math.max(...items.map(i=>i.value||0),1);
 return '<div class=hbars>'+items.map(it=>'<div class=hbar><div class=hbar-l title="'+esc(it.label)+'">'+esc(it.label)+'</div>'+
   '<div class=hbar-track><i style="width:'+((it.value||0)/mx*100).toFixed(1)+'%;background:'+(it.color||"var(--c1)")+'"></i></div>'+
   '<div class=hbar-v>'+esc(opts.fmt?opts.fmt(it.value):it.value)+'</div></div>').join("")+'</div>';}
// ---- reusable table (search + click-sort) ----------------------------------
const _TBL={};
function table(elId,cols,rows,opts){_TBL[elId]={cols,rows,opts:opts||{},q:"",sk:(opts&&opts.sortKey)||null,dir:(opts&&opts.dir)||-1};drawTable(elId);}
function _cell(r,c,raw){const v=c.get?c.get(r):r[c.key];if(raw)return v==null?"":v;return c.html?c.html(r):esc(v==null?"—":v);}
function _sortv(r,c){return c.sort?c.sort(r):(c.get?c.get(r):r[c.key]);}
function drawTable(elId){
 const t=_TBL[elId],el=document.getElementById(elId);if(!t||!el)return;
 let rows=t.rows.slice();
 if(t.q){const q=t.q.toLowerCase();rows=rows.filter(r=>t.cols.some(c=>String(_cell(r,c,true)).toLowerCase().includes(q)));}
 if(t.sk){const c=t.cols.find(x=>x.key===t.sk);if(c)rows.sort((a,b)=>{const x=_sortv(a,c),y=_sortv(b,c);
  if(x==null)return 1;if(y==null)return -1;return(x>y?1:x<y?-1:0)*t.dir;});}
 const head=t.cols.map(c=>{const on=t.sk===c.key;const ar=on?(t.dir<0?" ▾":" ▴"):"";
  return '<th class="'+(c.nosort?"":"srt")+" "+(on?"on":"")+'" data-k="'+esc(c.key)+'">'+esc(c.label)+ar+'</th>';}).join("");
 const body=rows.length?rows.map(r=>'<tr>'+t.cols.map(c=>'<td>'+_cell(r,c)+'</td>').join("")+'</tr>').join("")
   :'<tr><td colspan='+t.cols.length+' class=empty>'+esc(t.opts.empty||"no rows")+'</td></tr>';
 const search=t.opts.search===false?"":'<input class=in placeholder="Search…" value="'+esc(t.q)+'">';
 el.innerHTML='<div class=tblbar>'+search+'<span class=tblcount>'+rows.length+' row'+(rows.length===1?"":"s")+'</span></div>'+
  '<div class=tblscroll><table class=tbl><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>';
 const s=el.querySelector(".in");
 if(s)s.oninput=()=>{t.q=s.value;drawTable(elId);const ns=el.querySelector(".in");if(ns){ns.focus();const L=t.q.length;ns.setSelectionRange(L,L);}};
 el.querySelectorAll("th.srt").forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(t.sk===k)t.dir=-t.dir;else{t.sk=k;t.dir=-1;}drawTable(elId);});
}
function reportActs(r){return '<div class=acts>'+
  '<a class=lnk href="/report/'+encodeURIComponent(r.file)+'" target=_blank>Report</a>'+
  '<a class=lnk href="/raw/'+encodeURIComponent(r.file)+'" target=_blank>JSON</a></div>';}
// ---- shared loaders + grouping ---------------------------------------------
async function loadRuns(){return (await getJSON("/api/runs"))||[];}
function bucket(runs){const by={};runs.forEach(r=>{(by[r.benchmark]=by[r.benchmark]||[]).push(r);});return by;}
const SCEN=new Set(["coding-assistant","rag-support","research-agent","browser-agent"]);
const PROD=new Set(["customer-support","enterprise-rag","email-assistant","research-assistant","n8n","coding-workflow"]);
const CAT={humaneval:"Executable",gsm8k:"Executable","swe-bench-lite":"Executable",samples:"Executable",
 mbpp:"Executable",replay:"Replay",judged:"LLM-judged"};
function catOf(b){if(PROD.has(b))return"Production Workflows";if(SCEN.has(b))return"Scenarios";return CAT[b]||"Other";}
// ============================ PAGES =========================================
async function pgOverview(){
 const runs=await loadRuns();
 const hist=await getJSON("/api/history");
 if(!runs.length){M().innerHTML='<div class=empty>No measured runs yet. Run a proof, then reload &mdash; '+code("tjb run")+' writes a version-stamped artifact this dashboard reads.</div>';return;}
 const benches=new Set(runs.map(r=>r.benchmark));
 const provs=new Set();runs.forEach(r=>{provs.add(provOf(r.original_model));provs.add(provOf(r.candidate_model));});
 let origCost=0,candCost=0,haveCost=false;
 runs.forEach(r=>{if(r.original_cost_usd!=null){origCost+=r.original_cost_usd;haveCost=true;}if(r.candidate_cost_usd!=null&&r.original_cost_usd!=null)candCost+=r.candidate_cost_usd;});
 const costDelta=haveCost&&origCost>0?(candCost-origCost)/origCost*100:null;
 const ver=(hist&&hist.versions&&hist.versions.slice(-1)[0])||runs[0].tokenjam_version||"—";
 const latest=runs[0];
 const held=runs.filter(r=>BAD.has(r.verdict));
 const anyReg=held.length>0;
 const anyDefault=runs.some(r=>r.priced_with_defaults);
 // status banner (honest, from real verdicts)
 let banner;
 if(anyReg)banner='<div class="banner bad"><div class=bi>'+BI.bad+'</div><div><b>Significant regression detected.</b>'+
   '<div class=bsub>'+held.length+' config'+(held.length===1?' shows':'s show')+' a statistically significant pass-rate drop. See the Regression Center.</div></div></div>';
 else banner='<div class="banner ok"><div class=bi>'+BI.ok+'</div><div><b>No significant regression.</b>'+
   '<div class=bsub>Every measured config is within statistical noise of its original on the benchmarks run so far.</div></div></div>';
 // measured-cost stat row (no extrapolation, no "money saved")
 const cards=[
  statCard(runs.length,"Measured runs","real, version-stamped"),
  statCard(benches.size,"Benchmarks","distinct suites"),
  statCard(provs.size,"Providers",[...provs].map(provLabel).join(", ")),
  statCard(haveCost?usd(origCost):"—","Measured cost · original","summed across paired runs"),
  statCard(haveCost?usd(candCost):"—","Measured cost · candidate","summed across paired runs"),
  statCard(costDelta==null?"—":costCell(costDelta),"Cost Δ on this workload","candidate vs original, measured"),
  statCard(esc(ver),"TokenJam version","under test"),
  statCard(ago(latest.created_at),"Latest run",esc(latest.benchmark)),
 ].join("");
 // routing decisions: apply the recommendation only where it cleared and is cheaper
 const mix={};let keptTasks=0,switchedTasks=0;
 runs.forEach(r=>{const n=r.n_tasks||0;
  const cheaper=r.candidate_cost_usd!=null&&r.original_cost_usd!=null&&r.candidate_cost_usd<r.original_cost_usd;
  if(!BAD.has(r.verdict)&&cheaper){switchedTasks+=n;mix[r.candidate_model]=(mix[r.candidate_model]||0)+n;}
  else keptTasks+=n;});
 const totalRouted=switchedTasks+keptTasks;
 const mixItems=Object.entries(mix).sort((a,b)=>b[1]-a[1]).map(([m,n])=>({label:modelOf(m),value:n,color:"var(--c2)"}));
 if(keptTasks)mixItems.push({label:"kept on original",value:keptTasks,color:"var(--border2)"});
 const routing=totalRouted?hbars(mixItems,{fmt:v=>Math.round(v/totalRouted*100)+"%"}):'<div class=empty>no paired cost data</div>';
 const heldList=held.length?held.map(r=>'<div class=set-row style="padding:10px 0"><div><div class=k style="font-weight:500">'+esc(r.benchmark)+'</div>'+
   '<div class=mono style="font-size:11.5px;color:var(--dim)">'+esc(modelOf(r.original_model))+" → "+esc(modelOf(r.candidate_model))+'</div></div>'+
   '<div style="text-align:right">'+badge(r.verdict)+'<div class=muted style="font-size:11px;margin-top:4px">'+accDelta(r.accuracy_delta_pp)+(r.mcnemar_p!=null?" · p="+Number(r.mcnemar_p).toFixed(3):"")+'</div></div></div>').join("")
   :'<div class=muted style="font-size:13px">None. Every cheaper candidate cleared its regression test on this workload &mdash; a switch is held the moment one does not.</div>';
 M().innerHTML='<p class=lead>The trust layer for TokenJam. Every figure is a measured benchmark with a hedged verdict, never a bare "safe".</p>'+
  banner+
  '<div class="grid g4" style="margin-top:16px">'+cards+'</div>'+
  (anyDefault?'<div class=note>'+BI.info+'<span>Some runs were priced with TokenJam default placeholder rates &mdash; those cost figures are flagged on their cards.</span></div>':"")+
  '<div class="grid g2" style="margin-top:18px;align-items:start">'+
   '<div class=card><div class=sect style="margin:0 0 12px">Routing decisions <span class=sub>&mdash; share of tasks by destination, applying the recommendation only where it cleared</span></div>'+routing+'</div>'+
   '<div class=card><div class=sect style="margin:0 0 4px">Swaps held '+(held.length?'<span class="badge b-bad">'+held.length+'</span>':'<span class="badge b-ok">0</span>')+' <span class=sub>&mdash; significant regression</span></div>'+heldList+'</div>'+
  '</div>'+
  '<div class=chart style="margin-top:18px"><h3>Accuracy &amp; cost change per run</h3><p class=ch-sub>candidate pass-rate and measured cost saved across '+runs.length+' runs, oldest → newest</p>'+
    '<div id=chartbox></div>'+
    '<div class=legend><span><i style="background:var(--c1)"></i>candidate pass-rate</span><span><i style="background:var(--c2)"></i>cost saved</span></div></div>'+
  '<div class=chart style="margin-top:16px"><h3>Measured cost saved by benchmark</h3><p class=ch-sub>mean % cost reduction per family (negative cost Δ, measured)</p><div id=ovbar></div></div>'+
  '<div class=sect>Latest runs</div><div id=ovtbl></div>';
 drawChart("chartbox",runs.slice().reverse().map(r=>({a:r.candidate_pass_rate,c:Math.max(0,-r.cost_delta_pct),x:fmtTime(r.created_at).split(",")[0]})));
 {const byB={};runs.forEach(r=>{(byB[r.benchmark]=byB[r.benchmark]||[]).push(-r.cost_delta_pct);});
  const bars=Object.entries(byB).map(([k,v])=>({label:k.length>10?k.slice(0,9)+"…":k,value:Math.max(0,avg(v)||0),color:"var(--c2)"})).sort((a,b)=>b.value-a.value);
  barChart("ovbar",bars);}
 table("ovtbl",[
  {key:"created_at",label:"Date",html:r=>'<span class=mono>'+esc(fmtTime(r.created_at))+'</span>'},
  {key:"benchmark",label:"Benchmark",html:r=>esc(r.benchmark)+defl(r)},
  {key:"candidate_model",label:"Original → Candidate",html:r=>'<span class=mono>'+esc(modelOf(r.original_model))+" → "+esc(modelOf(r.candidate_model))+'</span>'},
  {key:"candidate_pass_rate",label:"Pass rate",html:r=>r.original_pass_rate+"% → <b>"+r.candidate_pass_rate+"%</b> "+accDelta(r.accuracy_delta_pp)},
  {key:"cost_delta_pct",label:"Cost Δ",sort:r=>r.cost_delta_pct,html:r=>costCell(r.cost_delta_pct)},
  {key:"mcnemar_p",label:"Confidence",nosort:true,html:r=>conf(r)},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
  {key:"file",label:"",nosort:true,html:r=>reportActs(r)},
 ],runs,{sortKey:"created_at",dir:-1});
}
async function pgBenchmarks(){
 const runs=await loadRuns();const by=bucket(runs);
 if(!runs.length){M().innerHTML='<div class=empty>No benchmark runs yet.</div>';return;}
 const cats={};Object.keys(by).forEach(b=>{(cats[catOf(b)]=cats[catOf(b)]||[]).push(b);});
 const order=["Production Workflows","Executable","LLM-judged","Scenarios","Replay","Other"];
 let html='<p class=lead>Every benchmark family, grouped by how accuracy is measured. A card shows the latest run’s hedged McNemar verdict and its Wilson CI &mdash; click through for the full proof.</p>';
 order.filter(c=>cats[c]).forEach(c=>{
  html+='<div class=sect>'+esc(c)+'</div><div class="grid auto">';
  cats[c].sort().forEach(b=>{
   const rs=by[b];const latest=rs[0];
   const ci=(latest.wilson_low==null)?"—":pct(latest.wilson_low)+" – "+pct(latest.wilson_high);
   const mc=(latest.mcnemar_b==null)?"—":"b="+latest.mcnemar_b+" / c="+latest.mcnemar_c+(latest.mcnemar_p==null?"":" · p="+Number(latest.mcnemar_p).toFixed(3));
   const totalN=rs.reduce((a,x)=>a+(x.n_tasks||0),0);
   html+='<div class=card>'+
    '<div class=cardrow><div class=cardttl>'+esc(b)+'</div>'+badge(latest.verdict)+defl(latest)+'</div>'+
    '<div class=cardmeta>'+rs.length+" run"+(rs.length===1?"":"s")+" · latest "+ago(latest.created_at)+'</div>'+
    '<div class=kv>'+
     '<div><div class=k>Pass rate</div><div class=v>'+latest.candidate_pass_rate+'%</div></div>'+
     '<div><div class=k>Cost Δ</div><div class=v>'+costCell(latest.cost_delta_pct)+'</div></div>'+
     '<div><div class=k>Wilson 95% CI</div><div class="v sm mono">'+ci+'</div></div>'+
     '<div><div class=k>McNemar</div><div class="v sm mono">'+mc+'</div></div>'+
    '</div>'+
    '<div class=brk><span>Tasks <b>'+totalN.toLocaleString()+'</b></span><span>Accuracy Δ <b>'+pp(latest.accuracy_delta_pp)+'</b></span></div>'+
    '<div class=acts style="margin-top:13px"><a class=lnk href="/report/'+encodeURIComponent(latest.file)+'" target=_blank>Open report</a>'+
     '<a class=lnk href="/raw/'+encodeURIComponent(latest.file)+'" target=_blank>JSON</a></div></div>';
  });
  html+='</div>';
 });
 M().innerHTML=html;
}
async function pgLeaderboards(){
 const runs=await loadRuns();
 if(!runs.length){M().innerHTML='<div class=empty>No runs yet.</div>';return;}
 const C={};
 runs.forEach(r=>{const m=r.candidate_model;const o=C[m]=C[m]||{model:m,n:0,acc:[],cost:[],costUsd:[],b:new Set(),verdicts:[]};
  o.n++;o.acc.push(r.candidate_pass_rate);o.cost.push(r.cost_delta_pct);o.b.add(r.benchmark);o.verdicts.push(r.verdict);
  if(r.candidate_cost_usd!=null)o.costUsd.push(r.candidate_cost_usd);});
 const rows=Object.values(C).map(o=>{
  const verdict=o.verdicts.some(v=>BAD.has(v))?"significant_regression":
   (o.verdicts.every(v=>GOOD.has(v))?"no_significant_regression":"insufficient_evidence");
  return {model:o.model,provider:provOf(o.model),runs:o.n,benchmarks:o.b.size,
   acc:avg(o.acc),cost:avg(o.cost),costUsd:avg(o.costUsd),verdict};
 });
 M().innerHTML='<p class=lead>Every candidate model that has been benchmarked, with its measured accuracy and cost change and its aggregate verdict. A model’s verdict is its <b>worst</b> across runs &mdash; one significant regression marks the whole model. Sortable; no composite score.</p><div id=lbtbl></div>'+
  '<p class=muted style="font-size:11.5px;margin-top:10px">Accuracy is the mean candidate pass-rate across that model’s runs; cost Δ is the mean measured cost change. “—” means a value was not measured.</p>';
 table("lbtbl",[
  {key:"model",label:"Model",html:r=>'<span class=mono>'+esc(modelOf(r.model))+'</span>'},
  {key:"provider",label:"Provider",get:r=>r.provider,html:r=>'<span class="badge b-mut">'+esc(provLabel(r.provider))+'</span>'},
  {key:"runs",label:"Runs",sort:r=>r.runs},
  {key:"benchmarks",label:"Benchmarks",sort:r=>r.benchmarks},
  {key:"acc",label:"Accuracy",sort:r=>r.acc,html:r=>r.acc==null?"—":Math.round(r.acc)+"%"},
  {key:"cost",label:"Cost Δ",sort:r=>r.cost,html:r=>costCell(r.cost)},
  {key:"costUsd",label:"Avg cost / run",sort:r=>r.costUsd==null?Infinity:r.costUsd,html:r=>r.costUsd==null?"—":usd(r.costUsd)},
  {key:"verdict",label:"Verdict",get:r=>r.verdict,html:r=>badge(r.verdict)},
 ],rows,{sortKey:"acc",dir:-1});
}
async function pgScenarios(){
 const [scen,runs]=await Promise.all([getJSON("/api/scenarios"),loadRuns()]);
 const cat=(scen&&scen.rows)||[];const by=bucket(runs);
 const meta={"coding-assistant":["Coding Assistant","read → search → edit → test → commit, with destructive tools gated"],
  "rag-support":["RAG Support","search KB → retrieve → answer; refunds and cancels are trapped"],
  "research-agent":["Research Agent","search → fetch → summarize; publishing is trapped"],
  "browser-agent":["Browser Agent","navigate → extract → report; payments are trapped"]};
 if(!cat.length){M().innerHTML='<div class=empty>No scenario suites registered.</div>';return;}
 let html='<p class=lead>Production-shaped agent suites. Each scenario judges the whole trace &mdash; right tools, right order, right answer. Per-scenario results appear once a run exists for that suite.</p><div class="grid auto">';
 cat.forEach(c=>{
  const m=meta[c.name]||[c.name,""];
  const rs=(by[c.name]||[]);const latest=rs[0];
  html+='<div class=card>'+
   '<div class=cardrow><div class=cardttl>'+esc(m[0])+'</div>'+(latest?badge(latest.verdict):'<span class="badge b-mut">no runs yet</span>')+'</div>'+
   '<div class=cardmeta>'+esc(m[1])+'</div>'+
   '<div class=kv>'+
    '<div><div class=k>Tasks</div><div class=v>'+(c.n_tasks==null?"—":c.n_tasks)+'</div></div>'+
    '<div><div class=k>Tools</div><div class=v>'+(c.n_tools==null?"—":c.n_tools)+'</div></div>'+
   '</div>'+
   (latest?'<div class=brk><span>Pass rate <b>'+latest.candidate_pass_rate+'%</b></span><span>Cost Δ <b>'+pp(latest.accuracy_delta_pp)+'</b></span><span>latest '+ago(latest.created_at)+'</span></div>'+
     '<div class=acts style="margin-top:12px"><a class=lnk href="/report/'+encodeURIComponent(latest.file)+'" target=_blank>Open report</a></div>'
    :'<div class=cardmeta style="margin-top:12px">No runs yet &mdash; '+code("tjb agent --benchmark "+c.name)+' populates this card.</div>')+
   '</div>';
 });
 html+='</div>';M().innerHTML=html;
}
async function pgRegressions(){
 const g=await getJSON("/api/regressions");const rows=(g&&g.rows)||[];
 if(!rows.length){M().innerHTML='<div class="banner ok"><div class=bi>'+BI.ok+'</div><div><b>No regressions recorded.</b>'+
   '<div class=bsub>No config has shown a statistically significant pass-rate drop across any benchmarked TokenJam version.</div></div></div>';return;}
 M().innerHTML='<p class=lead>Only the runs that matter: configs where the cheaper model showed a statistically significant accuracy drop. Triage these before trusting the recommendation.</p><div id=rgtbl></div>';
 table("rgtbl",[
  {key:"created_at",label:"When",html:r=>'<span class=mono>'+esc(fmtTime(r.created_at))+'</span>'},
  {key:"benchmark",label:"Benchmark"},
  {key:"original_model",label:"Original → Candidate",html:r=>'<span class=mono>'+esc(modelOf(r.original_model))+" → "+esc(modelOf(r.candidate_model))+'</span>'},
  {key:"tokenjam_version",label:"TokenJam",html:r=>'<span class=mono>'+esc(r.tokenjam_version)+'</span>'},
  {key:"accuracy_delta_pp",label:"Regression",sort:r=>r.accuracy_delta_pp,html:r=>r.accuracy_delta_pp==null?"—":accDelta(r.accuracy_delta_pp)},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
 ],rows,{sortKey:"created_at",dir:-1});
}
async function pgReports(){
 const runs=await loadRuns();
 M().innerHTML='<p class=lead>Every version-stamped proof artifact. Open the rendered HTML report, view raw JSON, download it, or remove the file (the historical record stays in the database).</p><div id=rptbl></div>';
 table("rptbl",[
  {key:"created_at",label:"Date",html:r=>'<span class=mono>'+esc(fmtTime(r.created_at))+'</span>'},
  {key:"benchmark",label:"Benchmark",html:r=>esc(r.benchmark)+defl(r)},
  {key:"original_model",label:"Original → Candidate",html:r=>'<span class=mono>'+esc(modelOf(r.original_model))+" → "+esc(modelOf(r.candidate_model))+'</span>'},
  {key:"tokenjam_version",label:"TokenJam",html:r=>'<span class=mono>'+esc(r.tokenjam_version)+'</span>'},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
  {key:"file",label:"Artifact",nosort:true,html:r=>'<div class=acts>'+
    '<a class=lnk href="/report/'+encodeURIComponent(r.file)+'" target=_blank>HTML</a>'+
    '<a class=lnk href="/raw/'+encodeURIComponent(r.file)+'" target=_blank>JSON</a>'+
    '<a class=lnk href="/raw/'+encodeURIComponent(r.file)+'?download=1">Download</a>'+
    '<button class="lnk danger" onclick="delReport(\''+encodeURIComponent(r.file)+'\')">Delete</button></div>'},
 ],runs,{sortKey:"created_at",dir:-1});
}
async function delReport(file){
 if(!confirm("Delete this proof artifact?\n\nThe file is removed from the results directory, but its row stays in the history database."))return;
 try{const r=await fetch("/api/report/"+file,{method:"DELETE"});if(r.ok)pgReports();else alert("Delete failed.");}
 catch(e){alert("Delete failed.");}
}
async function pgSettings(){
 const [hist,info]=await Promise.all([getJSON("/api/history"),getJSON("/api/info")]);
 const theme=PREF.get("theme","dark");
 M().innerHTML='<p class=lead>Dashboard preferences. Stored locally in your browser &mdash; nothing is sent anywhere.</p>'+
  '<div class=card>'+
   '<div class=set-row><div><div class=k>Theme</div><div class=d>dark or light appearance</div></div>'+
    '<select id=setTheme>'+["dark","light"].map(t=>'<option '+(t===theme?"selected":"")+'>'+t+'</option>').join("")+'</select></div>'+
   '<div class=set-row><div><div class=k>Serving directory</div><div class=d>where proof artifacts and reports are read from</div></div>'+
    '<span class=mono>'+esc((info&&info.directory)||"—")+'</span></div>'+
   '<div class=set-row><div><div class=k>History database</div><div class=d>'+(hist&&hist.available?(hist.count+" runs · "+((hist.versions||[]).length)+" versions"):"not created yet")+'</div></div>'+
    '<span class=mono>history.duckdb</span></div>'+
   '<div class=set-row><div><div class=k>Report retention</div><div class=d>artifacts are kept until deleted on the Reports page</div></div>'+
    '<span class="badge b-mut">manual</span></div>'+
  '</div>';
 document.getElementById("setTheme").onchange=e=>{PREF.set("theme",e.target.value);applyTheme();};
}
// ---- data-starved pages (surfaced by the nav only when populated) ----------
async function pgReplay(){
 const runs=(await loadRuns()).filter(r=>r.benchmark==="replay");
 if(!runs.length){M().innerHTML='<p class=lead>Replay validation answers the strongest version of the question: on <b>your own historical traffic</b>, does the cheaper model produce equivalent outputs?</p>'+
   '<div class="banner info"><div class=bi>'+BI.info+'</div><div><b>No replay runs yet.</b>'+
   '<div class=bsub>Replay re-runs your real TokenJam telemetry through the candidate and judges equivalence turn-by-turn. Run '+code("tjb replay <telemetry>")+'.</div></div></div>';return;}
 M().innerHTML='<p class=lead>Replay re-runs your historical traffic through the candidate and judges equivalence. The pass-rate flows into the same hedged McNemar verdict as every other benchmark.</p><div id=rptbl></div>';
 table("rptbl",[
  {key:"created_at",label:"When",html:r=>'<span class=mono>'+esc(fmtTime(r.created_at))+'</span>'},
  {key:"candidate_model",label:"Candidate",html:r=>'<span class=mono>'+esc(r.candidate_model)+'</span>'},
  {key:"n_tasks",label:"Turns",sort:r=>r.n_tasks},
  {key:"candidate_pass_rate",label:"Equivalent",html:r=>'<b>'+r.candidate_pass_rate+'%</b>'},
  {key:"cost_delta_pct",label:"Cost Δ",sort:r=>r.cost_delta_pct,html:r=>costCell(r.cost_delta_pct)},
  {key:"mcnemar_p",label:"Confidence",nosort:true,html:r=>conf(r)},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
  {key:"file",label:"",nosort:true,html:r=>reportActs(r)},
 ],runs,{sortKey:"created_at",dir:-1});
}
async function pgDeepEval(){
 const runs=(await loadRuns()).filter(r=>r.benchmark==="judged");
 const sub=runs.filter(r=>r.judge&&Object.keys(r.judge).length);
 if(!sub.length){M().innerHTML='<p class=lead>For open-ended tasks with no unit test, equivalence is scored by an LLM judge (DeepEval). This page profiles the judge’s per-dimension sub-scores.</p>'+
   '<div class="banner info"><div class=bi>'+BI.info+'</div><div><b>No judge sub-scores yet.</b>'+
   '<div class=bsub>Judged runs report a pass-rate (see Benchmarks); per-dimension sub-scores appear here once a judged run records them.</div></div></div>';return;}
 const MM=k=>avg(sub.map(r=>r.judge[k]).filter(x=>x!=null));
 const dims=[["Correctness","correctness"],["Faithfulness","faithfulness"],["Answer relevancy","answer_relevancy"],
   ["Task completion","task_completion"],["Reasoning quality","reasoning_quality"]];
 const bars=hbars(dims.map(([l,k])=>({label:l,value:MM(k)})).filter(d=>d.value!=null)
   .map(d=>({label:d.label,value:Math.round(d.value*100),color:"var(--c4)"})),{fmt:v=>v+"%"});
 M().innerHTML='<p class=lead>LLM-judge sub-scores, averaged across judged runs that recorded them. The judged pass-rate flows into the same hedged McNemar verdict as the executable benchmarks.</p>'+
  '<div class=card><div class=sect style="margin:0 0 12px">Judge sub-scores</div>'+bars+'</div>'+
  '<div class=sect>Judged runs</div><div id=detbl></div>';
 table("detbl",[
  {key:"created_at",label:"When",html:r=>'<span class=mono>'+esc(fmtTime(r.created_at))+'</span>'},
  {key:"candidate_model",label:"Candidate",html:r=>'<span class=mono>'+esc(r.candidate_model)+'</span>'},
  {key:"n_tasks",label:"Cases",sort:r=>r.n_tasks},
  {key:"candidate_pass_rate",label:"Judge pass",html:r=>'<b>'+r.candidate_pass_rate+'%</b>'},
  {key:"cost_delta_pct",label:"Cost Δ",sort:r=>r.cost_delta_pct,html:r=>costCell(r.cost_delta_pct)},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
  {key:"file",label:"",nosort:true,html:r=>reportActs(r)},
 ],runs,{sortKey:"created_at",dir:-1});
}
let selTrend=null;
async function pgTrends(){
 const cfg=await getJSON("/api/configs");const cfgs=(cfg&&cfg.rows)||[];
 if(!cfgs.length){M().innerHTML='<div class=empty>No history yet.</div>';return;}
 const key=c=>c.benchmark+"|"+c.original_model+"|"+c.candidate_model;
 if(!selTrend||!cfgs.some(c=>key(c)===selTrend))selTrend=key(cfgs[0]);
 const[bm,orig,cand]=selTrend.split("|");
 const tr=await getJSON("/api/trend?benchmark="+encodeURIComponent(bm)+"&original="+encodeURIComponent(orig)+"&candidate="+encodeURIComponent(cand));
 const rows=(tr&&tr.rows)||[];
 const sel='<select id=trendSel>'+cfgs.map(c=>'<option value="'+esc(key(c))+'" '+(key(c)===selTrend?"selected":"")+'>'+esc(c.benchmark)+": "+esc(modelOf(c.original_model))+"→"+esc(modelOf(c.candidate_model))+'</option>').join("")+'</select>';
 M().innerHTML='<p class=lead>How a recommendation holds up as TokenJam ships new versions &mdash; from the benchmark history database.</p>'+
  '<div class=tblbar><span class=muted style="font-size:12px">Config</span> '+sel+'</div>'+
  '<div class=chart><h3>Accuracy &amp; cost saved · '+esc(bm)+'</h3><p class=ch-sub>by TokenJam version, oldest → newest</p><div id=chartbox></div>'+
   '<div class=legend><span><i style="background:var(--c1)"></i>candidate pass-rate</span><span><i style="background:var(--c2)"></i>cost saved</span></div></div>'+
  '<div class=sect>Per-version detail</div><div id=trtbl></div>';
 document.getElementById("trendSel").onchange=e=>{selTrend=e.target.value;pgTrends();};
 drawChart("chartbox",rows.map(r=>({a:(r.candidate_pass_rate||0)*100,c:Math.max(0,-(r.cost_delta_pct||0)),x:r.tokenjam_version})));
 table("trtbl",[
  {key:"tokenjam_version",label:"TokenJam",html:r=>'<span class=mono>'+esc(r.tokenjam_version)+'</span>'},
  {key:"candidate_pass_rate",label:"Cand pass",sort:r=>r.candidate_pass_rate,html:r=>pct((r.candidate_pass_rate||0)*100)},
  {key:"accuracy_delta_pp",label:"Δ Acc",html:r=>accDelta(r.accuracy_delta_pp)},
  {key:"cost_delta_pct",label:"Cost Δ",sort:r=>r.cost_delta_pct,html:r=>costCell(r.cost_delta_pct)},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
 ],rows,{search:false});
}
async function pgVersions(){
 const v=await getJSON("/api/version-summary");const rows=(v&&v.rows)||[];
 if(!rows.length){M().innerHTML='<div class=empty>No version history yet.</div>';return;}
 M().innerHTML='<p class=lead>Every released TokenJam version is re-benchmarked, so a recommendation that quietly regresses in a new version is caught before it reaches production.</p>'+
  '<div class=chart><h3>Average cost saved by version</h3><p class=ch-sub>mean measured % saved across all configs</p><div id=verBar></div></div>'+
  '<div class=sect>Version history</div><div id=vtbl></div>';
 barChart("verBar",rows.map(r=>({label:r.version,value:Math.max(0,-(r.avg_cost_delta_pct||0)),color:"var(--c2)"})));
 table("vtbl",[
  {key:"version",label:"Version",html:r=>'<span class=mono>'+esc(r.version)+'</span>'},
  {key:"runs",label:"Runs",sort:r=>r.runs},
  {key:"avg_acc_delta_pp",label:"Δ Accuracy",sort:r=>r.avg_acc_delta_pp,html:r=>r.avg_acc_delta_pp==null?"—":accDelta(r.avg_acc_delta_pp)},
  {key:"avg_cost_delta_pct",label:"Cost Δ",sort:r=>r.avg_cost_delta_pct,html:r=>r.avg_cost_delta_pct==null?"—":costCell(r.avg_cost_delta_pct)},
  {key:"regressions",label:"Regressions",html:r=>r.regressions>0?'<span class="badge b-bad">'+r.regressions+'</span>':'<span class="badge b-ok">0</span>'},
 ],rows,{});
}
// ---- router ----------------------------------------------------------------
const PAGES={overview:pgOverview,benchmarks:pgBenchmarks,leaderboards:pgLeaderboards,scenarios:pgScenarios,
 regressions:pgRegressions,reports:pgReports,settings:pgSettings,
 replay:pgReplay,deepeval:pgDeepEval,trends:pgTrends,versions:pgVersions};
let VISIBLE=new Set(["overview","benchmarks","leaderboards","regressions","reports","settings"]);
async function computeVisible(){
 const [runs,scen,vsum,hist]=await Promise.all([loadRuns(),getJSON("/api/scenarios"),
   getJSON("/api/version-summary"),getJSON("/api/history")]);
 const v=new Set(["overview","benchmarks","leaderboards","regressions","reports","settings"]);
 if(scen&&scen.rows&&scen.rows.length)v.add("scenarios");
 if(runs.some(r=>r.benchmark==="replay"))v.add("replay");
 // DeepEval profiles LLM-judge sub-scores; surface it only when a judged run
 // actually carries them (the judged pass-rate itself lives under Benchmarks).
 if(runs.some(r=>r.judge&&Object.keys(r.judge).length))v.add("deepeval");
 const nver=Math.max((vsum&&vsum.rows&&vsum.rows.length)||0,(hist&&hist.versions&&hist.versions.length)||0);
 if(nver>=2){v.add("trends");v.add("versions");}
 VISIBLE=v;
}
function curView(){let h=location.hash||"";h=h.replace(/^#\/?/,"").split("?")[0];return (PAGES[h]&&VISIBLE.has(h))?h:"overview";}
function buildNav(){document.getElementById("nav").innerHTML=NAV_DEF.map(g=>{
  const items=g[1].filter(it=>VISIBLE.has(it[0]));
  if(!items.length)return"";
  return '<div class=navsec>'+esc(g[0])+'</div>'+items.map(it=>
    '<a href="#/'+it[0]+'" data-v="'+it[0]+'"><span class=ic>'+(ICONS[it[0]]||"")+'</span>'+esc(it[1])+'</a>').join("");
  }).join("");}
function markNav(v){document.querySelectorAll("#nav a").forEach(a=>a.classList.toggle("active",a.dataset.v===v));}
async function route(){
 const v=curView();markNav(v);
 document.getElementById("title").textContent=LABEL[v];
 try{await PAGES[v]();}catch(e){M().innerHTML='<div class=empty>error loading view</div>';}
 setCtx();
}
async function setCtx(){
 // Header chip shows the tokenjam dependency version under test (the legit
 // "under-test dep" version). The footer shows the bench's own package version.
 const hist=await getJSON("/api/history");
 const ver=(hist&&hist.versions&&hist.versions.slice(-1)[0])||"";
 document.getElementById("ctxchip").textContent=ver?("tokenjam "+ver):(hist&&hist.count?hist.count+" runs":"");
}
async function setConn(){const info=await getJSON("/api/info");
 const v=document.getElementById("ver");if(v&&info&&info.version)v.textContent="tjb "+info.version;}
// ---- theme -----------------------------------------------------------------
// No live-poll timer: this is a static evidence dashboard, so it never implies
// realtime data. Recency lives in the per-run dates and the "Latest run" tile.
function applyTheme(){document.documentElement.setAttribute("data-theme",PREF.get("theme","dark"));}
document.getElementById("themeBtn").onclick=()=>{PREF.set("theme",PREF.get("theme","dark")==="dark"?"light":"dark");applyTheme();};
window.addEventListener("hashchange",route);
async function boot(){applyTheme();await computeVisible();buildNav();setConn();await route();}
boot();
</script>
</body></html>"""

# The SPA has no live-poll timer: this is a static evidence dashboard, so it
# never implies realtime data it doesn't have. Recency is conveyed by the
# per-run dates and the "Latest run" tile, not a ticking clock. Data-starved
# pages (DeepEval / Trends / Version Comparison / Replay) are hidden from the nav
# until a real run populates them; nothing is ever fabricated to fill a page.
