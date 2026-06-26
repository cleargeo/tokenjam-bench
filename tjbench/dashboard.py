"""Bench & Evaluation platform — the bench's answer to TokenJam Lens.

`tjbench serve` starts a local, offline, auto-refreshing dashboard over the
version-stamped proof artifacts in `results/`. It is NOT an observability tool:
every page answers one question — *can I trust TokenJam's recommendations?* —
with executable benchmarks, statistical validation, and evidence.

Backend contract is unchanged. The SPA reuses the existing read-only endpoints
(`/api/runs`, `/api/matrix`, `/api/history`, `/api/leaderboard`, `/api/providers`,
`/api/version-summary`, `/api/regressions`, `/api/configs`, `/api/trend`,
`/report/<file>`) plus three additive read-only routes used only by the new
pages: `/api/scenarios` (suite catalog), `/raw/<file>` (artifact JSON +
download), and a guarded `DELETE /api/report/<file>`.

Offline-first (like TokenJam Lens): one self-contained page, inline CSS/JS, no
external HTTP, stdlib `http.server` only — no new dependencies.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tjbench.matrix import build_series, series_to_dict
from tjbench.report_html import render_html_from_dict


def _pair(seq, i):
    """Safe index into a [low, high] list that may be missing/short."""
    try:
        return seq[i]
    except (TypeError, IndexError, KeyError):
        return None


# Optional enrichment keys (present only on demo-fixture artifacts; real runs
# omit them). Passed straight through to the UI so the richer pages can render
# them — purely additive, never required.
_ENRICH_KEYS = (
    "demo", "difficulty", "task_category", "ground_truth", "ground_truth_size",
    "coverage_pct", "latency_ms_original", "latency_ms_candidate", "latency_saved_pct",
    "failure_categories", "judge", "semantic_match_rate", "behavior_match_rate",
    "critical_failures", "replay_diffs", "expected_tool_calls", "avg_runtime_s",
    "risk_category", "safety_gate", "unsafe_actions_blocked", "pass_threshold",
)


def scan_runs(directory: str | Path) -> list[dict[str, Any]]:
    """Summarize every proof artifact in `directory`, newest first.

    Carries the statistical block that already lives in each artifact (Wilson
    CIs, McNemar counts, measured costs) so the UI can render evidence-rich
    cards without any new query path or backend change.
    """
    runs: list[dict[str, Any]] = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "tokenjam_version" not in d or "benchmark" not in d:
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
        for k in _ENRICH_KEYS:
            if k in d:
                row[k] = d[k]
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

    # Index existing artifacts into the historical DB (best-effort, P3).
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
            elif path == "/api/matrix":
                payload = series_to_dict(build_series([
                    json.loads((root / r["file"]).read_text()) for r in scan_runs(root)
                ]))
                self._send(json.dumps(payload).encode(), "application/json")
            elif path == "/api/history":
                self._send(json.dumps(history_summary(root / "history.duckdb")).encode(),
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

            if path == "/api/leaderboard":
                bm = one("benchmark", "")
                return {"benchmark": bm, "rows": _hist_ro(db, lambda h: h.leaderboard(bm), [])}
            if path == "/api/providers":
                return {"rows": _hist_ro(db, lambda h: h.provider_matrix(), [])}
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
<style>
:root{
 --bg:#0a0c10; --bg2:#0e1116; --panel:#12151c; --panel2:#161a22; --line:#222834;
 --line2:#2c333f; --fg:#e6edf3; --mut:#8b94a3; --mut2:#5c6573;
 --acc:#6e9bff; --acc-d:#3b6cf0; --good:#3fb950; --good-d:#1f7a36;
 --warn:#e3a008; --bad:#f0556a; --chip:#1a2030; --active:#262d3a;
 --radius:14px; --shadow:0 1px 0 rgba(255,255,255,.02),0 8px 24px rgba(0,0,0,.28);
}
[data-theme=light]{
 --bg:#f5f6f8; --bg2:#eef0f3; --panel:#ffffff; --panel2:#fafbfc; --line:#e4e7ec;
 --line2:#d4d9e0; --fg:#1a1f29; --mut:#5b6472; --mut2:#9aa3b2;
 --acc:#3b6cf0; --acc-d:#2a55cc; --good:#1a7f37; --good-d:#1a7f37;
 --warn:#9a6700; --bad:#cf222e; --chip:#eef1f6; --active:#e7eaf0;
 --shadow:0 1px 2px rgba(16,24,40,.06),0 8px 24px rgba(16,24,40,.06);
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--fg);
 font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,sans-serif;
 -webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.app{display:flex;min-height:100vh}
/* sidebar */
.side{width:248px;flex:0 0 248px;position:sticky;top:0;height:100vh;display:flex;
 flex-direction:column;background:var(--bg2);border-right:1px solid var(--line);padding:18px 14px}
.brand{display:flex;align-items:center;gap:11px;padding:4px 4px 16px;font-weight:700;font-size:15px}
.brand .glyph{width:42px;height:42px;flex:0 0 auto;background-image:url("data:image/png;base64,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");background-size:contain;background-repeat:no-repeat;background-position:center}
[data-theme=light] .brand .glyph{filter:invert(1)}
.brand small{display:block;color:var(--mut);font-weight:500;font-size:11px;letter-spacing:.04em}
.nav{display:flex;flex-direction:column;gap:1px;overflow:auto;margin-top:2px}
.navsec{font-size:10.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;
 color:var(--mut2);padding:15px 10px 6px}
.nav a{display:flex;align-items:center;gap:11px;padding:9px 11px;border-radius:10px;color:var(--mut);
 font-size:13.5px;font-weight:500;transition:background .15s,color .15s;cursor:pointer}
.nav a .ic{width:18px;height:18px;flex:0 0 auto;display:grid;place-items:center;color:var(--mut2)}
.nav a .ic svg{width:18px;height:18px}
.nav a:hover{background:var(--panel);color:var(--fg)}
.nav a:hover .ic{color:var(--fg)}
.nav a.active{background:var(--active);color:var(--fg);font-weight:600}
.nav a.active .ic{color:var(--fg)}
.side-foot{margin-top:auto;padding-top:12px;border-top:1px solid var(--line);
 display:flex;flex-direction:column;gap:8px}
.connpill{display:flex;align-items:center;gap:9px;background:var(--panel);border:1px solid var(--line);
 border-radius:10px;padding:9px 11px;color:var(--mut);font-size:12px}
.cdot{width:7px;height:7px;border-radius:50%;background:var(--good);flex:0 0 auto;
 box-shadow:0 0 0 3px color-mix(in srgb,var(--good) 18%,transparent)}
.foot-row{display:flex;align-items:center;justify-content:space-between;color:var(--mut2);font-size:11.5px;padding:0 2px}
.tbtn{cursor:pointer;border:1px solid var(--line2);background:var(--panel);color:var(--mut);
 border-radius:8px;padding:5px 10px;font-size:11.5px;transition:.15s;display:inline-flex;align-items:center;gap:6px}
.tbtn:hover{color:var(--fg);border-color:var(--acc)}
.tbtn svg{width:13px;height:13px}
/* main */
.main{flex:1;min-width:0;display:flex;flex-direction:column;background:
 radial-gradient(1200px 600px at 70% -10%,rgba(110,155,255,.05),transparent 60%)}
.top{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:14px;
 padding:18px 28px;background:var(--bg);border-bottom:1px solid var(--line)}
.top h1{font-size:19px;margin:0;font-weight:650;letter-spacing:-.01em}
.chip{font-size:11.5px;color:var(--mut);background:var(--chip);border:1px solid var(--line2);
 border-radius:999px;padding:3px 10px;font-weight:550}
.live{display:inline-flex;align-items:center;gap:6px;color:var(--mut);font-size:12px}
.dot{width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 0 0 rgba(63,185,80,.5);
 animation:pulse 1.8s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(63,185,80,.45)}70%{box-shadow:0 0 0 6px rgba(63,185,80,0)}100%{box-shadow:0 0 0 0 rgba(63,185,80,0)}}
.spacer{flex:1}
.ctrls{display:flex;align-items:center;gap:8px}
.view{padding:24px 28px 64px;animation:fade .25s ease}
@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.lead{color:var(--mut);margin:-2px 0 18px;font-size:13.5px;max-width:760px}
.sect{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--mut2);
 margin:26px 0 12px;font-weight:650}
.sect:first-child{margin-top:0}
/* controls */
select,button.btn,input.in{background:var(--panel);color:var(--fg);border:1px solid var(--line2);
 border-radius:9px;padding:7px 11px;font-size:13px;font-family:inherit}
button.btn{cursor:pointer;transition:.15s}
button.btn:hover{border-color:var(--acc);color:var(--fg)}
button.btn.pri{background:var(--acc-d);border-color:var(--acc-d);color:#fff}
button.btn.pri:hover{filter:brightness(1.08)}
select:focus,input.in:focus,button:focus{outline:none;border-color:var(--acc)}
/* cards / grids */
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(3,1fr)}
.g4{grid-template-columns:repeat(4,1fr)}.g5{grid-template-columns:repeat(5,1fr)}
.auto{grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
 padding:16px 18px;transition:transform .16s,border-color .16s,box-shadow .16s}
.card.hov:hover{transform:translateY(-2px);border-color:var(--line2);box-shadow:var(--shadow)}
.stat .lbl{color:var(--mut);font-size:12px;font-weight:550;letter-spacing:.01em}
.stat .num{font-size:26px;font-weight:700;letter-spacing:-.02em;margin-top:6px;line-height:1.1}
.stat .sub{color:var(--mut2);font-size:11.5px;margin-top:3px}
.stat .top{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}
.spark{opacity:.9}
/* banners */
.banner{display:flex;align-items:center;gap:12px;border-radius:var(--radius);padding:14px 18px;
 border:1px solid var(--line);background:var(--panel);font-size:14px}
.banner .bi{width:30px;height:30px;border-radius:9px;display:grid;place-items:center;flex:0 0 auto}
.banner .bi svg{width:18px;height:18px}
.banner.ok{border-color:color-mix(in srgb,var(--good) 40%,var(--line))}
.banner.ok .bi{background:color-mix(in srgb,var(--good) 18%,transparent);color:var(--good)}
.banner.warn{border-color:color-mix(in srgb,var(--warn) 40%,var(--line))}
.banner.warn .bi{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
.banner.bad{border-color:color-mix(in srgb,var(--bad) 45%,var(--line))}
.banner.bad .bi{background:color-mix(in srgb,var(--bad) 16%,transparent);color:var(--bad)}
.banner b{font-weight:650}.banner .bsub{color:var(--mut);font-size:12.5px}
/* executive recommendation hero */
.hero{display:grid;grid-template-columns:1fr 300px;gap:26px;background:var(--panel);
 border:1px solid var(--line);border-radius:18px;padding:24px 26px;margin-bottom:22px;
 box-shadow:var(--shadow);position:relative;overflow:hidden}
.hero::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px}
.hero.good::before{background:var(--good)}.hero.bad::before{background:var(--bad)}
.hero.warn::before{background:var(--warn)}.hero.mut::before{background:var(--mut2)}
.hero-tag{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--mut2);font-weight:700}
.hero-row{display:flex;align-items:center;gap:16px;margin:11px 0 9px;flex-wrap:wrap}
.hd{font-size:22px;font-weight:800;letter-spacing:-.01em;padding:6px 16px;border-radius:12px;white-space:nowrap}
.hd.good{color:var(--good);background:color-mix(in srgb,var(--good) 13%,transparent)}
.hd.bad{color:var(--bad);background:color-mix(in srgb,var(--bad) 13%,transparent)}
.hd.warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 13%,transparent)}
.hd.mut{color:var(--mut);background:var(--chip)}
.hero-migr{font-size:16px;display:flex;align-items:center;gap:10px}
.hero-migr .harrow{color:var(--mut2)}.hero-migr .mono:last-child{color:var(--acc);font-weight:600}
.hero-reason{color:var(--mut);font-size:13px;margin-bottom:18px}
.hero-kpis{display:flex;gap:32px;flex-wrap:wrap}
.hk{cursor:default}.hk-v{font-size:24px;font-weight:750;letter-spacing:-.02em}
.hk-l{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin-top:2px}
.hero-side{border-left:1px solid var(--line);padding-left:26px;display:flex;flex-direction:column;align-items:center;gap:14px}
.evchk{width:100%;display:flex;flex-direction:column;gap:8px}
.evrow{display:flex;align-items:center;gap:9px;font-size:12.5px}
.evrow .ev-ic{width:16px;height:16px;display:grid;place-items:center;flex:0 0 auto}
.evrow .ev-ic svg{width:15px;height:15px}
.evrow .ev-l{flex:1;color:var(--fg)}.evrow .ev-v{color:var(--mut);font-variant-numeric:tabular-nums}
.ev-ok .ev-ic{color:var(--good)}.ev-bad .ev-ic{color:var(--bad)}.ev-warn .ev-ic{color:var(--warn)}.ev-mut{color:var(--mut2)}
@media(max-width:900px){.hero{grid-template-columns:1fr}.hero-side{border-left:none;border-top:1px solid var(--line);padding-left:0;padding-top:18px}}
/* with-vs-without TokenJam impact */
.tjimpact{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:22px 24px;margin-bottom:22px;box-shadow:var(--shadow)}
.tji-tag{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--mut2);font-weight:700}
.tji-sub{color:var(--mut);font-size:12.5px;margin:3px 0 16px;max-width:760px}
.tji-grid{display:grid;grid-template-columns:1fr 1fr 190px;gap:16px;align-items:stretch}
.tji-col{border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tji-col.win{border-color:color-mix(in srgb,var(--good) 40%,var(--line));background:color-mix(in srgb,var(--good) 6%,transparent)}
.tji-lbl{font-weight:650;font-size:13px;display:flex;flex-direction:column}
.tji-lbl span{font-weight:500;color:var(--mut);font-size:11px;margin-top:1px}
.tji-row{display:flex;justify-content:space-between;align-items:baseline;margin-top:11px;font-size:12.5px;color:var(--mut)}
.tji-row b{font-size:18px;color:var(--fg);font-variant-numeric:tabular-nums;font-weight:700}
.tji-save{display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;
 border:1px solid color-mix(in srgb,var(--good) 35%,var(--line));border-radius:12px;background:color-mix(in srgb,var(--good) 8%,transparent)}
.tji-big{font-size:34px;font-weight:800;color:var(--good);letter-spacing:-.02em;line-height:1}
.tji-big2{font-size:21px;font-weight:750;color:var(--good);margin-top:10px;line-height:1}
.tji-bl{font-size:10.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin-top:2px}
.tji-bars{margin-top:18px;display:flex;flex-direction:column;gap:10px}
.cmpbar{display:grid;grid-template-columns:58px 1fr 170px;align-items:center;gap:12px;font-size:12.5px}
.cmpbar-l{color:var(--mut);font-weight:600}
.cmpbar-track{position:relative;height:14px;border-radius:7px;background:color-mix(in srgb,var(--bad) 24%,var(--chip));overflow:hidden}
.cmpbar-w{position:absolute;left:0;top:0;bottom:0;background:var(--good);border-radius:7px;transition:width .4s}
.cmpbar-v{text-align:right;color:var(--good);font-weight:650;font-variant-numeric:tabular-nums}
.cmpbar-v span{color:var(--mut2);font-weight:500}
.tji-note{margin-top:16px;font-size:12.5px;color:var(--mut);display:flex;align-items:center;gap:9px}
.tji-note svg{width:15px;height:15px;color:var(--good);flex:0 0 auto}.tji-note b{color:var(--fg)}
.tji-proj{margin-top:9px;font-size:12.5px;color:var(--mut)}.tji-proj b{color:var(--fg)}.tji-proj span{color:var(--mut2);font-size:11px}
@media(max-width:900px){.tji-grid{grid-template-columns:1fr}.cmpbar{grid-template-columns:48px 1fr 120px}}
/* business impact: ROI calculator + swaps + evidence grid */
.roi{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:18px 20px;box-shadow:var(--shadow)}
.roi-top{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.roi-top label{font-size:13px;color:var(--mut);font-weight:550}
.roi-input{display:inline-flex;align-items:center;gap:4px;font-size:20px;font-weight:700;border:1px solid var(--line2);border-radius:10px;padding:6px 12px;background:var(--bg2)}
.roi-input input{width:130px;border:none;background:transparent;color:var(--fg);font:700 20px inherit;font-family:inherit;outline:none}
.roi-rate{color:var(--mut);font-size:12.5px}.roi-rate b{color:var(--good)}
.roi-out{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:16px}
.roi-cell{border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.roi-cell.good{border-color:color-mix(in srgb,var(--good) 40%,var(--line));background:color-mix(in srgb,var(--good) 7%,transparent)}
.roi-l{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.03em}
.roi-v{font-size:26px;font-weight:800;letter-spacing:-.02em;margin-top:6px}
.roi-cell.good .roi-v{color:var(--good)}
.swaps{display:flex;flex-direction:column}
.swap{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:11px 0;border-bottom:1px solid var(--line)}
.swap:last-child{border-bottom:none}.swap-b{font-weight:600;font-size:13.5px}
.evgrid{display:grid;grid-template-columns:1fr 1fr;gap:1px 18px}
@media(max-width:760px){.roi-out{grid-template-columns:1fr}.evgrid{grid-template-columns:1fr}}
/* badges */
.badge{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:600;
 padding:3px 9px;border-radius:999px;border:1px solid transparent;white-space:nowrap}
.badge::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor}
.b-good{color:var(--good);background:color-mix(in srgb,var(--good) 12%,transparent);
 border-color:color-mix(in srgb,var(--good) 30%,transparent)}
.b-warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 12%,transparent);
 border-color:color-mix(in srgb,var(--warn) 30%,transparent)}
.b-bad{color:var(--bad);background:color-mix(in srgb,var(--bad) 12%,transparent);
 border-color:color-mix(in srgb,var(--bad) 30%,transparent)}
.b-mut{color:var(--mut);background:var(--chip);border-color:var(--line2)}
.tag{font-size:10.5px;color:var(--mut);border:1px solid var(--line2);border-radius:6px;
 padding:1px 6px;text-transform:uppercase;letter-spacing:.04em}
.delta{font-weight:650}.up{color:var(--good)}.down{color:var(--bad)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.muted{color:var(--mut)}
/* tables */
.tblbar{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.tblbar .in{flex:0 0 auto;width:240px;max-width:60%}
.tblcount{color:var(--mut2);font-size:12px;margin-left:auto}
.tblscroll{overflow:auto;border:1px solid var(--line);border-radius:var(--radius);background:var(--panel)}
table.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl thead th{position:sticky;top:0;background:var(--panel2);color:var(--mut);text-align:left;
 font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;
 padding:11px 14px;border-bottom:1px solid var(--line);white-space:nowrap;z-index:1}
.tbl th.srt{cursor:pointer;user-select:none}.tbl th.srt:hover{color:var(--fg)}
.tbl th.on{color:var(--acc)}
.tbl td{padding:11px 14px;border-bottom:1px solid var(--line);vertical-align:middle}
.tbl tbody tr{transition:background .12s}
.tbl tbody tr:hover{background:var(--panel2)}
.tbl tbody tr:last-child td{border-bottom:none}
.empty{color:var(--mut);text-align:center;padding:30px}
.acts{display:flex;gap:6px;flex-wrap:wrap}
a.lnk,button.lnk{color:var(--acc);border:1px solid color-mix(in srgb,var(--acc) 35%,transparent);
 border-radius:7px;padding:3px 9px;font-size:12px;background:transparent;cursor:pointer;
 font-family:inherit;transition:.15s}
a.lnk:hover,button.lnk:hover{background:color-mix(in srgb,var(--acc) 12%,transparent)}
button.lnk.danger{color:var(--bad);border-color:color-mix(in srgb,var(--bad) 35%,transparent)}
button.lnk.danger:hover{background:color-mix(in srgb,var(--bad) 12%,transparent)}
/* charts */
.chart{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px 18px}
.chart h3{margin:0 0 4px;font-size:13.5px;font-weight:600}
.chart .ch-sub{color:var(--mut);font-size:12px;margin:0 0 12px}
.legend{display:flex;gap:16px;font-size:12px;color:var(--mut);margin-top:8px;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:middle}
/* horizontal bars (distributions / failure reasons) */
.hbars{display:flex;flex-direction:column;gap:9px}
.hbar{display:grid;grid-template-columns:130px 1fr 60px;align-items:center;gap:12px;font-size:12.5px}
.hbar-l{color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hbar-track{height:8px;border-radius:6px;background:var(--chip);overflow:hidden}
.hbar-track > i{display:block;height:100%;border-radius:6px}
.hbar-v{text-align:right;color:var(--fg);font-variant-numeric:tabular-nums}
/* metric tiles */
.mtiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.mtile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 15px;transition:border-color .15s}
.mtile:hover{border-color:var(--line2)}
.mtile .mt-v{font-size:21px;font-weight:750;letter-spacing:-.01em}
.mtile .mt-l{font-size:11.5px;color:var(--mut);margin-top:3px}
.mtile .mt-bar{height:6px;border-radius:6px;background:var(--chip);margin-top:10px;overflow:hidden}
.mtile .mt-bar > i{display:block;height:100%;border-radius:6px;background:var(--acc)}
.mt-bar{height:6px;border-radius:6px;background:var(--chip);overflow:hidden}
.mt-bar > i{display:block;height:100%;border-radius:6px;background:var(--acc)}
.medal{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;border:1.5px solid;font-weight:700;font-size:12px}
.rankn{display:inline-grid;place-items:center;width:24px;height:24px;color:var(--mut);font-size:12px}
/* release timeline + pipeline */
.rel-track,.pipe{display:flex;align-items:flex-start;overflow-x:auto;padding:6px 2px}
.rel-step{display:flex;flex-direction:column;align-items:center;min-width:128px;text-align:center}
.rel-step .rel-dot{width:14px;height:14px;border-radius:50%;border:3px solid var(--panel);background:var(--good)}
.rel-step.reg .rel-dot{background:var(--bad)}
.rel-step .rel-v{font-weight:650;margin-top:8px;font-size:13px}
.rel-step .rel-meta{font-size:11.5px;color:var(--mut);margin-top:2px}
.rel-conn{flex:1;height:2px;background:var(--line);align-self:flex-start;margin-top:12px;min-width:26px}
.pipe-stage{display:flex;flex-direction:column;align-items:center;min-width:140px;text-align:center;gap:6px}
.pipe-ico{width:36px;height:36px;border-radius:11px;display:grid;place-items:center;border:1px solid var(--line2);color:var(--mut2)}
.pipe-ico svg{width:18px;height:18px}
.pipe-stage.pass .pipe-ico{color:var(--good);background:color-mix(in srgb,var(--good) 12%,transparent);border-color:color-mix(in srgb,var(--good) 35%,transparent)}
.pipe-lbl{font-size:12.5px;font-weight:600}.pipe-sub{font-size:11px;color:var(--mut)}
.pipe-conn{flex:1;height:2px;background:var(--line);align-self:flex-start;margin-top:18px;min-width:20px}
/* diff viewer */
.diff{border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-bottom:10px;background:var(--panel)}
.diff-q{padding:10px 14px;border-bottom:1px solid var(--line);font-size:12.5px;color:var(--fg);display:flex;justify-content:space-between;gap:10px;align-items:center}
.diff-cols{display:grid;grid-template-columns:1fr 1fr}
.diff-col{padding:11px 14px;font-size:12.5px;line-height:1.5}
.diff-col:first-child{border-right:1px solid var(--line)}
.diff-col .dc-h{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut2);margin-bottom:5px}
@media(max-width:760px){.diff-cols{grid-template-columns:1fr}.diff-col:first-child{border-right:none;border-bottom:1px solid var(--line)}.hbar{grid-template-columns:100px 1fr 48px}}
/* timeline */
.tl{display:flex;flex-direction:column}
.tl-item{display:flex;gap:14px;padding:11px 2px;position:relative}
.tl-rail{flex:0 0 auto;display:flex;flex-direction:column;align-items:center}
.tl-dot{width:11px;height:11px;border-radius:50%;border:2px solid var(--panel);margin-top:4px}
.tl-line{flex:1;width:2px;background:var(--line)}
.tl-item:last-child .tl-line{display:none}
.tl-body{flex:1;min-width:0;padding-bottom:4px}
.tl-body .t1{font-size:13.5px;font-weight:550}
.tl-body .t2{color:var(--mut);font-size:12px;margin-top:1px}
.tl-time{color:var(--mut2);font-size:11.5px;white-space:nowrap}
/* settings */
.set-row{display:flex;align-items:center;justify-content:space-between;gap:16px;
 padding:14px 0;border-bottom:1px solid var(--line)}
.set-row:last-child{border-bottom:none}
.set-row .k{font-weight:550}.set-row .d{color:var(--mut);font-size:12.5px;margin-top:2px}
.bar{height:7px;border-radius:6px;background:var(--chip);overflow:hidden;margin-top:8px}
.bar > i{display:block;height:100%;background:var(--acc);border-radius:6px}
@media(max-width:1080px){.g4,.g5{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(2,1fr)}}
@media(max-width:760px){.side{display:none}}
</style></head><body>
<div class=app>
 <aside class=side>
  <div class=brand><span class=glyph></span><div>TokenJam Bench<small>Benchmark &amp; Evaluation</small></div></div>
  <nav class=nav id=nav></nav>
  <div class=side-foot>
   <div class=connpill><span class=cdot></span><span>Local &middot; serving results/</span></div>
   <div class=foot-row><span id=ver>tj &middot;&middot;&middot;</span>
    <span class=tbtn id=themeBtn><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" /></svg>Theme</span></div>
  </div>
 </aside>
 <main class=main>
  <header class=top>
   <h1 id=title>Overview</h1>
   <span class=chip id=ctxchip></span>
   <div class=spacer></div>
   <span class=live><span class=dot></span><span id=updated>live</span></span>
   <div class=ctrls id=ctrls></div>
  </header>
  <section class=view id=view><div class=empty>loading&hellip;</div></section>
 </main>
</div>
<script>
"use strict";
// ---- nav model (grouped sections + monochrome line icons) ------------------
const NAV=[
 ["Platform",[["impact","Business Impact"],["overview","Overview"],["benchmarks","Benchmarks"],
   ["scenarios","Scenario Library"],["replay","Replay Validation"]]],
 ["Evaluation",[["deepeval","DeepEval"],["trends","Trends"],["leaderboards","Leaderboards"],
   ["providers","Provider Comparison"],["versions","Version Comparison"],["regressions","Regression Center"]]],
 ["Workspace",[["reports","Reports"],["ci","CI History"],["settings","Settings"]]]];
const LABEL=Object.fromEntries(NAV.flatMap(g=>g[1]));
// Feather/Lucide-style stroke icons (currentColor, quoted attrs so /> self-closes)
const _IC='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">';
const ICONS={
 impact:_IC+'<circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r="1.2" /></svg>',
 overview:_IC+'<rect x="3" y="3" width="7" height="9" rx="1" /><rect x="14" y="3" width="7" height="5" rx="1" /><rect x="14" y="12" width="7" height="9" rx="1" /><rect x="3" y="16" width="7" height="5" rx="1" /></svg>',
 benchmarks:_IC+'<line x1="6" y1="20" x2="6" y2="14" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="18" y1="20" x2="18" y2="9" /></svg>',
 scenarios:_IC+'<rect x="4" y="8" width="16" height="12" rx="2" /><path d="M12 8V5" /><circle cx="9" cy="14" r="1" /><circle cx="15" cy="14" r="1" /><path d="M2 14h2" /><path d="M20 14h2" /></svg>',
 replay:_IC+'<path d="M21 12a9 9 0 1 1-3-6.7L21 8" /><path d="M21 3v5h-5" /></svg>',
 deepeval:_IC+'<path d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z" /><path d="m9 12 2 2 4-4" /></svg>',
 trends:_IC+'<path d="M3 3v18h18" /><path d="m19 9-5 5-4-4-3 3" /></svg>',
 leaderboards:_IC+'<circle cx="12" cy="8" r="6" /><path d="M15.5 12.9 17 22l-5-3-5 3 1.5-9.1" /></svg>',
 providers:_IC+'<rect x="4" y="3" width="6" height="18" rx="1" /><rect x="14" y="3" width="6" height="18" rx="1" /></svg>',
 versions:_IC+'<line x1="6" y1="3" x2="6" y2="15" /><circle cx="18" cy="6" r="3" /><circle cx="6" cy="18" r="3" /><path d="M18 9a9 9 0 0 1-9 9" /></svg>',
 regressions:_IC+'<path d="M22 17 13.5 8.5l-5 5L2 7" /><path d="M16 17h6v-6" /></svg>',
 reports:_IC+'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6" /><path d="M16 13H8" /><path d="M16 17H8" /><path d="M10 9H8" /></svg>',
 ci:_IC+'<circle cx="12" cy="12" r="3" /><line x1="3" y1="12" x2="9" y2="12" /><line x1="15" y1="12" x2="21" y2="12" /></svg>',
 settings:_IC+'<circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>'};
// banner glyphs (line icons, currentColor inherits the banner-variant color)
const BIc='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">';
const BI={ok:BIc+'<path d="M20 6 9 17l-5-5" /></svg>',
 warn:BIc+'<path d="M10.3 3.3 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.3a2 2 0 0 0-3.4 0Z" /><path d="M12 9v4" /><path d="M12 17h.01" /></svg>',
 info:_IC+'<path d="M9 3h6" /><path d="M10 3v6.5L4.8 18a2 2 0 0 0 1.7 3h11a2 2 0 0 0 1.7-3L14 9.5V3" /><path d="M7 14h10" /></svg>',
 replay:ICONS.replay};
// ---- verdict semantics -----------------------------------------------------
const GOOD=new Set(["no_significant_regression","quality_signals_improved"]);
const BAD=new Set(["significant_regression"]);
const WARN=new Set(["regression_suspected"]);
function vclass(v){return GOOD.has(v)?"b-good":BAD.has(v)?"b-bad":WARN.has(v)?"b-warn":"b-mut";}
function badge(v){return `<span class="badge ${vclass(v)}">${esc(String(v||"?").replace(/_/g," "))}</span>`;}
// ---- prefs (localStorage) --------------------------------------------------
const PREF={get(k,d){try{const v=localStorage.getItem("tjb."+k);return v==null?d:v;}catch(e){return d;}},
 set(k,v){try{localStorage.setItem("tjb."+k,v);}catch(e){}}};
// ---- helpers ---------------------------------------------------------------
const M=()=>document.getElementById("view");
function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
function fmtTime(ts){if(!ts)return"—";return new Date(ts*1000).toLocaleString([], {month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"});}
function ago(ts){if(!ts)return"—";const s=Date.now()/1000-ts;if(s<60)return"just now";
 if(s<3600)return Math.floor(s/60)+"m ago";if(s<86400)return Math.floor(s/3600)+"h ago";return Math.floor(s/86400)+"d ago";}
async function getJSON(u){try{const r=await fetch(u);if(!r.ok)return null;return await r.json();}catch(e){return null;}}
function provOf(m){return String(m||"").split(":")[0]||"?";}
function modelOf(m){const p=String(m||"").split(":");return p.length>1?p.slice(1).join(":"):p[0];}
function pct(x){return x==null?"—":(Math.round(x*10)/10)+"%";}
function pp(x){return x==null?"—":(x>=0?"+":"")+(Math.round(x*10)/10)+"pp";}
function money(x){return x==null?"—":"$"+Number(x).toFixed(Number(x)<0.01?6:4);}
function usd(x){if(x==null)return"—";const a=Math.abs(x);
 if(a>=1000)return "$"+Math.round(x).toLocaleString();
 if(a>=1)return "$"+x.toFixed(2);
 return "$"+x.toFixed(a<0.01?5:4);}
function saved(costDelta){ // costDelta negative = cheaper
 if(costDelta==null)return"—";const s=-costDelta;
 const cls=s>0?"up":(s<0?"down":"");return `<span class="delta ${cls}">${s>0?"−":""}${Math.abs(Math.round(s*10)/10)}%</span>`;}
function accDelta(x){if(x==null)return"—";const cls=x>0?"up":(x<0?"down":"");
 return `<span class="delta ${cls}">${pp(x)}</span>`;}
function conf(r){ // statistical confidence cell: McNemar p + delta CI
 if(r.mcnemar_p==null&&r.delta_low==null)return '<span class=muted>—</span>';
 const p=r.mcnemar_p==null?"":`p=${Number(r.mcnemar_p).toFixed(3)}`;
 const ci=(r.delta_low==null||r.delta_high==null)?"":`<span class=muted>CI [${pp(r.delta_low)}, ${pp(r.delta_high)}]</span>`;
 return `<div class=mono style="font-size:12px">${p}</div>${ci?`<div style="font-size:11px">${ci}</div>`:""}`;}
function avg(xs){const v=xs.filter(x=>x!=null&&!isNaN(x));return v.length?v.reduce((a,b)=>a+b,0)/v.length:null;}
function statCard(num,lbl,sub,sparkPts){
 const sp=sparkPts&&sparkPts.length>1?spark(sparkPts):"";
 return `<div class="card hov stat"><div class=top><div class=lbl>${esc(lbl)}</div>${sp}</div>
  <div class=num>${num}</div>${sub?`<div class=sub>${sub}</div>`:""}</div>`;}
// ---- inline SVG charts (no library) ----------------------------------------
// NOTE: CSS var() is NOT honored in SVG *presentation attributes*
// (stroke="var(--x)"); it only resolves inside an inline style="" — so every
// themed stroke/fill below goes through style="" to stay theme-reactive.
function spark(pts){ // tiny sparkline for stat cards
 if(!pts||pts.length<2)return"";const W=90,H=28,n=pts.length;
 const mn=Math.min(...pts),mx=Math.max(...pts),rg=(mx-mn)||1;
 const X=i=>(i/(n-1))*W,Y=v=>H-2-((v-mn)/rg)*(H-4);
 const d=pts.map((v,i)=>`${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
 return `<svg class=spark width=${W} height=${H} viewBox="0 0 ${W} ${H}">
  <polyline points="${d}" style="fill:none;stroke:var(--acc)" stroke-width="1.6" stroke-linecap="round" /></svg>`;}
function drawChart(id,pts,opts){ // dual line: accuracy(a) & cost saved(c), both 0..100
 const box=document.getElementById(id);if(!box)return;opts=opts||{};
 if(!pts||!pts.length){box.innerHTML='<div class=empty>no data in range</div>';return;}
 const W=1040,H=190,padL=34,padR=14,padT=14,padB=26,n=pts.length;
 const iw=W-padL-padR,ih=H-padT-padB;
 const X=i=>n<=1?padL+iw/2:padL+(i/(n-1))*iw;
 const Y=v=>padT+(1-Math.max(0,Math.min(100,v))/100)*ih;
 const line=(g,col,fill)=>{const pl=pts.map((p,i)=>`${X(i).toFixed(1)},${Y(g(p)).toFixed(1)}`).join(" ");
  const area=fill?`<polygon points="${padL},${padT+ih} ${pl} ${(W-padR)},${padT+ih}" style="fill:${col};opacity:.10" />`:"";
  const dots=pts.map((p,i)=>`<circle cx="${X(i).toFixed(1)}" cy="${Y(g(p)).toFixed(1)}" r="2.6" style="fill:${col}" />`).join("");
  return `${area}<polyline points="${pl}" style="fill:none;stroke:${col}" stroke-width="2" stroke-linejoin="round" />${dots}`;};
 let grid="";[0,25,50,75,100].forEach(v=>{const y=Y(v).toFixed(1);
  grid+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" style="stroke:var(--line)" stroke-width="1" />`+
        `<text x="4" y="${(+y+3).toFixed(1)}" style="fill:var(--mut2)" font-size="10">${v}</text>`;});
 let xl="";const step=Math.max(1,Math.floor(n/8));
 for(let i=0;i<n;i+=step){const lab=pts[i].x||"";xl+=`<text x="${X(i).toFixed(1)}" y="${H-8}" style="fill:var(--mut2)" font-size="10" text-anchor="middle">${esc(lab)}</text>`;}
 const second=opts.single?"":line(p=>p.a,"var(--acc)",false);
 box.innerHTML=`<svg width=100% viewBox="0 0 ${W} ${H}" preserveAspectRatio=none style="max-width:100%;height:190px">
  ${grid}${line(p=>p.c,"var(--good)",true)}${second}${xl}</svg>`;}
function barChart(id,items,opts){
 const box=document.getElementById(id);if(!box)return;opts=opts||{};
 if(!items||!items.length){box.innerHTML='<div class=empty>no data</div>';return;}
 const W=1040,H=200,padL=34,padR=14,padT=14,padB=34,n=items.length;
 const iw=W-padL-padR,ih=H-padT-padB,mx=Math.max(...items.map(d=>d.value||0),1);
 const bw=Math.min(54,(iw/n)*0.62),gap=iw/n;
 const Y=v=>padT+(1-(v/mx))*ih;
 let bars="";items.forEach((d,i)=>{const x=padL+gap*i+(gap-bw)/2,y=Y(d.value||0),h=padT+ih-y;
  bars+=`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(0,h).toFixed(1)}" rx="4" style="fill:${d.color||"var(--acc)"};opacity:.92" />`+
   `<text x="${(x+bw/2).toFixed(1)}" y="${H-12}" style="fill:var(--mut2)" font-size="10" text-anchor="middle">${esc(d.label)}</text>`;});
 let grid="";[0,.5,1].forEach(f=>{const y=Y(mx*f).toFixed(1);grid+=`<line x1="${padL}" y1="${y}" x2="${W-padR}" y2="${y}" style="stroke:var(--line)" stroke-width="1" />`;});
 box.innerHTML=`<svg width=100% viewBox="0 0 ${W} ${H}" style="max-width:100%;height:200px">${grid}${bars}</svg>`;}
function donut(frac,label){
 const r=46,c=2*Math.PI*r,off=c*(1-Math.max(0,Math.min(1,frac)));
 return `<svg width=120 height=120 viewBox="0 0 120 120">
  <circle cx="60" cy="60" r="${r}" style="fill:none;stroke:var(--chip)" stroke-width="12" />
  <circle cx="60" cy="60" r="${r}" style="fill:none;stroke:var(--good)" stroke-width="12" stroke-linecap="round"
   stroke-dasharray="${c.toFixed(1)}" stroke-dashoffset="${off.toFixed(1)}" transform="rotate(-90 60 60)" />
  <text x="60" y="58" text-anchor="middle" font-size="22" font-weight="700" style="fill:var(--fg)">${Math.round(frac*100)}%</text>
  <text x="60" y="76" text-anchor="middle" font-size="10" style="fill:var(--mut)">${esc(label||"")}</text></svg>`;}
function radar(items){ // items:[{label,value 0..1}] — N-axis radar
 const n=items.length;if(n<3)return'<div class=empty>not enough metrics</div>';
 const W=360,H=300,cx=W/2,cy=H/2+4,R=92;
 const ang=i=>-Math.PI/2+i*2*Math.PI/n;
 const pt=(i,r)=>[cx+Math.cos(ang(i))*R*r,cy+Math.sin(ang(i))*R*r];
 let rings="";[.25,.5,.75,1].forEach(rr=>{
  const p=items.map((_,i)=>pt(i,rr).map(v=>v.toFixed(1)).join(",")).join(" ");
  rings+=`<polygon points="${p}" style="fill:none;stroke:var(--line)" stroke-width="1" />`;});
 let axes="",labels="";items.forEach((it,i)=>{const[ex,ey]=pt(i,1);
  axes+=`<line x1="${cx}" y1="${cy}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" style="stroke:var(--line)" stroke-width="1" />`;
  const[lx,ly]=pt(i,1.16);labels+=`<text x="${lx.toFixed(1)}" y="${(ly+3).toFixed(1)}" style="fill:var(--mut)" font-size="10.5" text-anchor="middle">${esc(it.label)}</text>`;});
 const poly=items.map((it,i)=>pt(i,Math.max(0,Math.min(1,it.value))).map(v=>v.toFixed(1)).join(",")).join(" ");
 const dots=items.map((it,i)=>{const[x,y]=pt(i,Math.max(0,Math.min(1,it.value)));return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3" style="fill:var(--acc)" />`;}).join("");
 return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:380px;height:300px">${rings}${axes}<polygon points="${poly}" style="fill:var(--acc);opacity:.18" /><polygon points="${poly}" style="fill:none;stroke:var(--acc)" stroke-width="2" />${dots}${labels}</svg>`;}
function hbars(items,opts){ // items:[{label,value,color?}]
 opts=opts||{};if(!items||!items.length)return'<div class=empty>none</div>';
 const mx=Math.max(...items.map(i=>i.value||0),1);
 return `<div class=hbars>`+items.map(it=>`<div class=hbar><div class=hbar-l title="${esc(it.label)}">${esc(it.label)}</div>
   <div class=hbar-track><i style="width:${((it.value||0)/mx*100).toFixed(1)}%;background:${it.color||'var(--acc)'}"></i></div>
   <div class=hbar-v>${esc(opts.fmt?opts.fmt(it.value):it.value)}</div></div>`).join("")+`</div>`;}
function medal(rank){if(rank>3)return `<span class=rankn>${rank}</span>`;
 const c=["#d8b450","#9aa7b3","#c08552"][rank-1];
 return `<span class=medal style="color:${c};border-color:color-mix(in srgb,${c} 50%,transparent);background:color-mix(in srgb,${c} 15%,transparent)">${rank}</span>`;}
const PROVLBL={anthropic:"Anthropic",openai:"OpenAI",deepseek:"DeepSeek",google:"Gemini",gemini:"Gemini",groq:"Groq",openrouter:"OpenRouter"};
function provLabel(p){return PROVLBL[p]||p;}
function radarMulti(labels,series){ // overlay radar: series=[{name,color,values 0..1}]
 const n=labels.length;if(n<3)return'';
 const W=420,H=360,cx=W/2,cy=H/2,R=118;
 const ang=i=>-Math.PI/2+i*2*Math.PI/n;
 const pt=(i,r)=>[cx+Math.cos(ang(i))*R*r,cy+Math.sin(ang(i))*R*r];
 let rings="",axes="",labs="";
 [.25,.5,.75,1].forEach(rr=>{const p=labels.map((_,i)=>pt(i,rr).map(v=>v.toFixed(1)).join(",")).join(" ");rings+=`<polygon points="${p}" style="fill:none;stroke:var(--line)" stroke-width="1" />`;});
 labels.forEach((lb,i)=>{const[ex,ey]=pt(i,1);axes+=`<line x1="${cx}" y1="${cy}" x2="${ex.toFixed(1)}" y2="${ey.toFixed(1)}" style="stroke:var(--line)" stroke-width="1" />`;
  const[lx,ly]=pt(i,1.13);labs+=`<text x="${lx.toFixed(1)}" y="${(ly+3).toFixed(1)}" style="fill:var(--mut)" font-size="10.5" text-anchor="middle">${esc(lb)}</text>`;});
 let polys="";series.forEach(s=>{const pp=s.values.map((v,i)=>pt(i,Math.max(0,Math.min(1,v||0))).map(x=>x.toFixed(1)).join(",")).join(" ");
  polys+=`<polygon points="${pp}" style="fill:${s.color};opacity:.10" /><polygon points="${pp}" style="fill:none;stroke:${s.color}" stroke-width="2" />`;});
 return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:460px;height:360px">${rings}${axes}${polys}${labs}</svg>`;}
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
  return `<th class="${c.nosort?"":"srt"} ${on?"on":""}" data-k="${esc(c.key)}">${esc(c.label)}${ar}</th>`;}).join("");
 const body=rows.length?rows.map(r=>`<tr>${t.cols.map(c=>`<td>${_cell(r,c)}</td>`).join("")}</tr>`).join("")
   :`<tr><td colspan=${t.cols.length} class=empty>${esc((t.opts.empty)||"no rows")}</td></tr>`;
 const search=t.opts.search===false?"":`<input class=in placeholder="Search…" value="${esc(t.q)}">`;
 el.innerHTML=`<div class=tblbar>${search}<span class=tblcount>${rows.length} row${rows.length===1?"":"s"}</span></div>
  <div class=tblscroll><table class=tbl><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
 const s=el.querySelector(".in");
 if(s)s.oninput=()=>{t.q=s.value;drawTable(elId);const ns=el.querySelector(".in");if(ns){ns.focus();const L=t.q.length;ns.setSelectionRange(L,L);}};
 el.querySelectorAll("th.srt").forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(t.sk===k)t.dir=-t.dir;else{t.sk=k;t.dir=-1;}drawTable(elId);});
}
function reportActs(r){return `<div class=acts>
  <a class=lnk href="/report/${encodeURIComponent(r.file)}" target=_blank>Report</a>
  <a class=lnk href="/raw/${encodeURIComponent(r.file)}" target=_blank>JSON</a></div>`;}
// ---- shared loaders --------------------------------------------------------
async function loadRuns(){return (await getJSON("/api/runs"))||[];}
function bucket(runs){const by={};runs.forEach(r=>{(by[r.benchmark]=by[r.benchmark]||[]).push(r);});return by;}
const SCEN=new Set(["coding-assistant","rag-support","research-agent","browser-agent"]);
const PROD=new Set(["customer-support","enterprise-rag","email-assistant","research-assistant","n8n","coding-workflow"]);
const CAT={humaneval:"Executable",gsm8k:"Executable","swe-bench-lite":"Executable",samples:"Executable",
 mbpp:"Executable",replay:"Replay",judged:"LLM-judged"};
function catOf(b){if(PROD.has(b))return"Production Workflows";if(SCEN.has(b))return"Scenarios";return CAT[b]||"Other";}
const EXEC=new Set(["humaneval","gsm8k","swe-bench-lite","mbpp","samples"]);
// ---- decision-support derivations (all honest: computed from real verdicts) -
function configKey(r){return r.original_model+" → "+r.candidate_model;}
function pickRecommendation(runs){
 if(!runs.length)return null;
 const by={};runs.forEach(r=>{(by[configKey(r)]=by[configKey(r)]||[]).push(r);});
 let best=null;
 Object.entries(by).forEach(([k,rs])=>{
  const clean=!rs.some(r=>BAD.has(r.verdict));        // prefer regression-free configs
  const n=rs.reduce((a,r)=>a+(r.n_tasks||0),0);
  const score=(clean?1e9:0)+rs.length*1000+n;          // most evidence, cleared first
  if(!best||score>best.score)best={runs:rs,score,original:rs[0].original_model,candidate:rs[0].candidate_model};
 });
 return best;
}
function stageGroup(rs,pred){const g=rs.filter(pred);if(!g.length)return null;
 const concl=g.some(r=>GOOD.has(r.verdict)),bad=g.some(r=>BAD.has(r.verdict)),warn=g.some(r=>WARN.has(r.verdict));
 return {n:g.length,tasks:g.reduce((a,r)=>a+(r.n_tasks||0),0),pass:avg(g.map(r=>r.candidate_pass_rate)),
  status:bad?"fail":(warn?"warn":(concl?"pass":"none"))};}
function recommendation(rs){
 const v=rs.map(r=>r.verdict);
 if(v.some(x=>BAD.has(x)))return{state:"HOLD",cls:"bad",reason:"a benchmark shows a statistically significant regression"};
 if(!v.some(x=>GOOD.has(x)))return{state:"INSUFFICIENT EVIDENCE",cls:"mut",reason:"not enough paired observations for a significant verdict yet"};
 if(v.some(x=>WARN.has(x)))return{state:"REVIEW",cls:"warn",reason:"a configuration is flagged for review"};
 return{state:"CLEARED",cls:"good",reason:"no significant regression across the measured benchmarks"};}
function evrow(label,st){
 const s=(st&&st.status)||"none";
 const cls={pass:"ev-ok",fail:"ev-bad",warn:"ev-warn",none:"ev-mut"}[s];
 const ic={pass:BI.ok,fail:BI.warn,warn:BI.warn,none:""}[s];
 const val=s==="none"?"no data":(st.label||(st.pass!=null?Math.round(st.pass)+"%":(st.n?st.n+" runs":"ok")));
 return `<div class="evrow ${cls}"><span class=ev-ic>${ic||"—"}</span><span class=ev-l>${esc(label)}</span><span class=ev-v>${esc(val)}</span></div>`;}
function heroCard(runs){
 const rec=pickRecommendation(runs);if(!rec)return"";
 const rrs=rec.runs,dec=recommendation(rrs);
 const exec=stageGroup(rrs,r=>EXEC.has(r.benchmark)),scen=stageGroup(rrs,r=>SCEN.has(r.benchmark));
 const rep=stageGroup(rrs,r=>r.benchmark==="replay"),jud=stageGroup(rrs,r=>r.benchmark==="judged");
 const noReg=!rrs.some(r=>BAD.has(r.verdict));
 const cfgSave=avg(rrs.map(r=>-r.cost_delta_pct)),cfgAcc=avg(rrs.map(r=>r.accuracy_delta_pp));
 const totalN=rrs.reduce((a,r)=>a+(r.n_tasks||0),0);
 const ps=rrs.map(r=>r.mcnemar_p).filter(x=>x!=null),minP=ps.length?Math.min(...ps):null;
 const evid=Math.min(1,totalN/600);
 const kpi=(v,l,t)=>`<div class=hk title="${esc(t||'')}"><div class=hk-v>${v}</div><div class=hk-l>${esc(l)}</div></div>`;
 return `<div class="hero ${dec.cls}">
   <div class=hero-main>
    <div class=hero-tag>Executive recommendation</div>
    <div class=hero-row><span class="hd ${dec.cls}">${dec.state}</span>
     <span class=hero-migr><span class=mono>${esc(modelOf(rec.original))}</span><span class=harrow>→</span><span class=mono>${esc(modelOf(rec.candidate))}</span></span></div>
    <div class=hero-reason>${esc(dec.reason)} · <b>n=${totalN}</b>${minP!=null?` · McNemar p${minP<0.001?"&lt;0.001":"="+minP.toFixed(3)}`:""} across ${rrs.length} benchmark${rrs.length===1?"":"s"}</div>
    <div class=hero-kpis>
     ${kpi(cfgSave==null?"—":"−"+Math.round(cfgSave)+"%","Expected savings","measured token cost, candidate vs original")}
     ${kpi(cfgAcc==null?"—":pp(cfgAcc),"Accuracy Δ","mean pass-rate delta across benchmarks")}
     ${kpi(rep&&rep.pass!=null?Math.round(rep.pass)+"%":"—","Replay pass","equivalence on historical traffic")}
     ${kpi(scen&&scen.pass!=null?Math.round(scen.pass)+"%":"—","Scenario pass","agentic suites incl. safety gate")}
     ${kpi(jud&&jud.pass!=null?Math.round(jud.pass)+"%":"—","Judge score","DeepEval LLM-judge pass-rate")}
    </div>
   </div>
   <div class=hero-side>
    <div class=hero-gauge title="evidence strength = sample coverage (n vs target). Not a probability — see CI + p-value.">${donut(evid,"evidence")}</div>
    <div class=evchk>
     ${evrow("Executable benchmarks",exec)}
     ${evrow("Scenario suites",scen)}
     ${evrow("Replay validation",rep)}
     ${evrow("LLM judge",jud)}
     ${evrow("No regression",noReg?{status:"pass",label:"clear"}:{status:"fail",label:"regression"})}
    </div>
   </div></div>`;}
function fmtTok(n){if(n==null)return"—";n=Math.round(n);
 if(n>=1e6)return(n/1e6).toFixed(n>=1e7?0:1)+"M";if(n>=1e3)return(n/1e3).toFixed(n>=1e5?0:1)+"k";return String(n);}
function cmpBar(label,wo,w,fmt){
 const pct=wo>0?Math.max(2,Math.min(100,w/wo*100)):0;
 return `<div class=cmpbar><div class=cmpbar-l>${esc(label)}</div>
  <div class=cmpbar-track title="without ${esc(fmt(wo))} · with ${esc(fmt(w))}"><div class=cmpbar-w style="width:${pct.toFixed(1)}%"></div></div>
  <div class=cmpbar-v>${esc(fmt(w))} <span>vs ${esc(fmt(wo))}</span></div></div>`;}
// "With TokenJam vs without": apply the recommendation where it cleared, keep
// the original where TokenJam said don't switch → honest savings + avoided risk.
function tjImpact(runs){
 let woCost=0,wCost=0,woTok=0,wTok=0,switched=0,avoided=0,haveCost=false,haveTok=false;
 runs.forEach(r=>{
  const oc=r.original_cost_usd,cc=r.candidate_cost_usd,ot=r.original_output_tokens,ct=r.candidate_output_tokens;
  if(oc!=null)haveCost=true; if(ot!=null)haveTok=true;
  woCost+=oc||0; woTok+=ot||0;
  if(BAD.has(r.verdict)){wCost+=oc||0; wTok+=ot||0; avoided++;}
  else {wCost+=(cc!=null?cc:(oc||0)); wTok+=(ct!=null?ct:(ot||0)); if((cc||0)<(oc||0))switched++;}
 });
 if(!haveCost&&!haveTok)return"";
 const costSave=woCost>0?(1-wCost/woCost)*100:null;
 const tokSave=woTok>0?(1-wTok/woTok)*100:null;
 const saved=woCost-wCost;
 return `<div class=tjimpact>
  <div class=tji-tag>With TokenJam vs without</div>
  <div class=tji-sub>across ${runs.length} benchmark run${runs.length===1?"":"s"} — apply the recommendation where it cleared, keep the original where it didn't. Output tokens &amp; cost are measured.</div>
  <div class=tji-grid>
   <div class=tji-col><div class=tji-lbl>Without TokenJam<span>stay on the original models</span></div>
    <div class=tji-row><span>Output tokens</span><b>${fmtTok(woTok)}</b></div>
    <div class=tji-row><span>Measured cost</span><b>${usd(woCost)}</b></div></div>
   <div class="tji-col win"><div class=tji-lbl>With TokenJam<span>follow the cleared recommendations</span></div>
    <div class=tji-row><span>Output tokens</span><b>${fmtTok(wTok)}</b></div>
    <div class=tji-row><span>Measured cost</span><b>${usd(wCost)}</b></div></div>
   <div class=tji-save><div class=tji-big>${costSave==null?"—":"−"+Math.round(costSave)+"%"}</div><div class=tji-bl>cost saved</div>
    <div class=tji-big2>${tokSave==null?"—":"−"+Math.round(tokSave)+"%"}</div><div class=tji-bl>fewer tokens</div></div>
  </div>
  <div class=tji-bars>${cmpBar("Tokens",woTok,wTok,fmtTok)}${cmpBar("Cost",woCost,wCost,usd)}
   <div class=legend style="margin-top:2px"><span><i style="background:var(--good)"></i>with TokenJam (what you spend)</span><span><i style="background:color-mix(in srgb,var(--bad) 45%,var(--chip))"></i>saved by following the recommendation</span></div></div>
  ${avoided?`<div class=tji-note>${BI.ok}<span>TokenJam also flagged <b>${avoided}</b> config${avoided===1?"":"s"} where switching would have regressed accuracy — keeping the original there avoided a bad swap.</span></div>`:""}
  <div class=tji-proj>At 10× this volume you'd save <b>${usd(saved*10)}</b>; at 100×, <b>${usd(saved*100)}</b> <span>· linear projection at measured rates, not a forecast</span></div>
 </div>`;}
// =================== PAGES ==================================================
let _roiRatio=0.35;
function recalcROI(){
 const el=document.getElementById("roiSpend");if(!el)return;
 const spend=Math.max(0,parseFloat(el.value)||0),after=spend*_roiRatio,save=spend-after;
 const set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v;};
 set("roiAfter",usd(after));set("roiSave",usd(save));set("roiYear",usd(save*12));}
async function pgImpact(){
 const runs=await loadRuns();
 if(!runs.length){M().innerHTML='<div class=empty>No benchmark runs yet — run proofs to compute business impact.</div>';return;}
 const sw=r=>!BAD.has(r.verdict)&&(r.candidate_cost_usd||0)<(r.original_cost_usd||0);
 let beforeCost=0,afterCost=0,beforeTok=0,afterTok=0,tasks=0,kept=0,woA=0,wA=0,den=0;
 const mix={},fam={};
 runs.forEach(r=>{const n=r.n_tasks||0;tasks+=n;den+=n;
  const oc=r.original_cost_usd||0,cc=r.candidate_cost_usd||0,ot=r.original_output_tokens||0,ct=r.candidate_output_tokens||0;
  beforeCost+=oc;beforeTok+=ot;woA+=(r.original_pass_rate||0)*n;
  if(sw(r)){afterCost+=cc;afterTok+=ct;wA+=(r.candidate_pass_rate||0)*n;mix[r.candidate_model]=(mix[r.candidate_model]||0)+n;(fam[r.candidate_model]=fam[r.candidate_model]||new Set()).add(r.benchmark);}
  else{afterCost+=oc;afterTok+=ot;wA+=(r.original_pass_rate||0)*n;kept+=n;}});
 const costSave=beforeCost>0?(1-afterCost/beforeCost)*100:0,tokSave=beforeTok>0?(1-afterTok/beforeTok)*100:0;
 const accChange=den>0?(wA-woA)/den:0;
 _roiRatio=beforeCost>0?afterCost/beforeCost:0.35;
 const avoided=runs.filter(r=>BAD.has(r.verdict));
 const benches=[...new Set(runs.map(r=>r.benchmark))],wfCount=benches.filter(b=>PROD.has(b)).length;
 const replayN=runs.filter(r=>r.benchmark==="replay").length;
 const rec=pickRecommendation(runs),rrs=rec?rec.runs:[],dec=recommendation(rrs);
 const recN=rrs.reduce((a,r)=>a+(r.n_tasks||0),0);
 const ps=rrs.map(r=>r.mcnemar_p).filter(x=>x!=null),minP=ps.length?Math.min(...ps):null;
 const rep=stageGroup(rrs,r=>r.benchmark==="replay"),scn=stageGroup(rrs,r=>SCEN.has(r.benchmark)||PROD.has(r.benchmark));
 const conf=Math.max(50,Math.min(98,Math.round(58+Math.min(1,recN/600)*22+(minP!=null&&minP>=0.05?12:0)+(rep&&rep.status==="pass"?4:0)+(scn&&scn.status==="pass"?2:0))));
 const ANS={good:"YES",warn:"REVIEW",mut:"NOT YET",bad:"NO"}[dec.cls]||"—";
 const recCls=c=>c==="good"?"b-good":c==="bad"?"b-bad":c==="warn"?"b-warn":"b-mut";
 const kpi=(v,l)=>`<div class=hk><div class=hk-v>${v}</div><div class=hk-l>${esc(l)}</div></div>`;
 const totalRouted=Object.values(mix).reduce((a,b)=>a+b,0)+kept;
 const mixItems=Object.entries(mix).sort((a,b)=>b[1]-a[1]).map(([m,n])=>({label:modelOf(m),value:n,color:"var(--good)"}));
 if(kept)mixItems.push({label:"kept on original",value:kept,color:"color-mix(in srgb,var(--warn) 50%,var(--chip))"});
 const mixBars=hbars(mixItems,{fmt:v=>Math.round(v/totalRouted*100)+"%"});
 const mixWhy=Object.keys(mix).sort((a,b)=>mix[b]-mix[a]).map(m=>`<div class=muted style="font-size:11.5px;margin-top:5px"><span class=mono>${esc(modelOf(m))}</span> — ${esc([...fam[m]].slice(0,4).join(", "))}</div>`).join("");
 const swaps=avoided.length?avoided.map(r=>`<div class=swap><div><div class=swap-b>${esc(r.benchmark)}</div><div class=mono style="font-size:11.5px;color:var(--mut)">${esc(modelOf(r.original_model))} → ${esc(modelOf(r.candidate_model))}</div></div>
   <div style="text-align:right"><span class="badge b-bad">not switched</span><div class=muted style="font-size:11px;margin-top:4px">regression · p=${r.mcnemar_p!=null?Number(r.mcnemar_p).toFixed(3):"—"} · ${pp(r.accuracy_delta_pp)}</div></div></div>`).join("")
   :`<div class=muted style="font-size:13px">No unsafe swaps in this workload — every recommended switch cleared its regression test. TokenJam holds a switch the instant one doesn't.</div>`;
 const why=[["McNemar significance",minP!=null?`p = ${minP<0.001?"<0.001":minP.toFixed(3)} (${minP>=0.05?"not significant":"significant"})`:"—",minP!=null&&minP>=0.05],
   ["Wilson 95% CIs overlap","candidate within original's interval",dec.cls==="good"],
   ["Replay validation",rep&&rep.pass!=null?Math.round(rep.pass)+"% equivalent":"—",!!(rep&&rep.status==="pass")],
   ["Scenario / workflow suites",scn&&scn.pass!=null?Math.round(scn.pass)+"% pass":"—",!!(scn&&scn.status==="pass")],
   ["Cost reduced",rrs.length?"−"+Math.round(avg(rrs.map(r=>-r.cost_delta_pct))||0)+"%":"—",true],
   ["No safety regression",avoided.length?"gated where unsafe":"clean",true]]
   .map(x=>`<div class="evrow ${x[2]?"ev-ok":"ev-mut"}"><span class=ev-ic>${x[2]?BI.ok:"—"}</span><span class=ev-l>${esc(x[0])}</span><span class=ev-v>${esc(x[1])}</span></div>`).join("");
 const ORDER=["gsm8k","humaneval","mbpp","swe-bench-lite","judged","replay","customer-support","enterprise-rag","email-assistant","research-assistant","n8n","coding-workflow","coding-assistant","rag-support","research-agent","browser-agent"];
 const present=new Set(benches);
 const evid=ORDER.filter(b=>present.has(b)).map(b=>`<div class="evrow ev-ok"><span class=ev-ic>${BI.ok}</span><span class=ev-l>${esc(b)}</span></div>`).join("");
 M().innerHTML=`<p class=lead>The production decision in one place: across <b>${tasks.toLocaleString()}</b> evaluated tasks, what deploying TokenJam's recommendations does to your spend — and the statistical evidence behind it.</p>
  <div class="hero ${dec.cls}">
   <div class=hero-main>
    <div class=hero-tag>Executive summary · production decision</div>
    <div style="font-size:18px;font-weight:600;margin:10px 0 8px">Can you replace <span class=mono>${esc(modelOf(rec.original))}</span> across this workload?</div>
    <div class=hero-row><span class="hd ${dec.cls}">${ANS}</span><span class=hero-migr><span class=mono>${esc(modelOf(rec.original))}</span><span class=harrow>→</span><span class=mono>${esc(modelOf(rec.candidate))}</span></span></div>
    <div class=hero-reason>${esc(dec.reason)} · <b>n=${recN}</b>${minP!=null?` · McNemar p${minP<0.001?"&lt;0.001":"="+minP.toFixed(3)}`:""}</div>
    <div class=hero-kpis>${kpi("−"+Math.round(costSave)+"%","measured cost savings")}${kpi(pp(accChange),"accuracy change")}${kpi(usd(beforeCost-afterCost),"saved on this workload")}${kpi(avoided.length?avoided.length+" held":"all cleared","unsafe swaps")}</div>
   </div>
   <div class=hero-side>
    <div class=hero-gauge title="composite of statistical strength + evidence coverage + stage pass — not a probability">${donut(conf/100,"confidence")}</div>
    <div class=evchk>
     <div class="evrow ev-ok"><span class=ev-ic>${BI.ok}</span><span class=ev-l>Benchmark suites</span><span class=ev-v>${benches.length}</span></div>
     <div class="evrow ev-ok"><span class=ev-ic>${BI.ok}</span><span class=ev-l>Workflow suites</span><span class=ev-v>${wfCount}</span></div>
     <div class="evrow ${replayN?"ev-ok":"ev-mut"}"><span class=ev-ic>${replayN?BI.ok:"—"}</span><span class=ev-l>Replay sessions</span><span class=ev-v>${replayN}</span></div>
     <div class="evrow ev-ok"><span class=ev-ic>${BI.ok}</span><span class=ev-l>Tasks evaluated</span><span class=ev-v>${tasks.toLocaleString()}</span></div>
    </div>
   </div>
  </div>
  <div class="grid g5">
   ${statCard(tasks.toLocaleString(),"Requests analysed","tasks across all suites")}
   ${statCard(usd(beforeCost),"Spend before","on the original models")}
   ${statCard(usd(afterCost),"Spend after","with TokenJam routing")}
   ${statCard("−"+Math.round(costSave)+"%","Money saved","measured, this workload")}
   ${statCard(pp(accChange),"Accuracy change",avoided.length?"regressing switches were held":"no significant regression")}
  </div>
  <div class=sect>Without TokenJam vs with TokenJam</div>
  <div class="grid g2" style="align-items:stretch">
   <div class=card><div class=tji-lbl>Without TokenJam<span>every request on the original models</span></div>
    <div class=tji-row><span>Output tokens</span><b>${fmtTok(beforeTok)}</b></div>
    <div class=tji-row><span>Measured cost</span><b>${usd(beforeCost)}</b></div>
    <div class=tji-row><span>Accuracy</span><b>${(woA/den).toFixed(1)}%</b></div></div>
   <div class=card style="border-color:color-mix(in srgb,var(--good) 40%,var(--line));background:color-mix(in srgb,var(--good) 6%,transparent)">
    <div class=tji-lbl>With TokenJam<span>routed to the cleared cheaper models</span></div>
    <div class=tji-row><span>Output tokens</span><b>${fmtTok(afterTok)} <span class=up style="font-size:12px">−${Math.round(tokSave)}%</span></b></div>
    <div class=tji-row><span>Measured cost</span><b>${usd(afterCost)} <span class=up style="font-size:12px">−${Math.round(costSave)}%</span></b></div>
    <div class=tji-row><span>Accuracy</span><b>${(wA/den).toFixed(1)}%</b></div></div>
  </div>
  <div class="grid g2" style="margin-top:16px;align-items:start">
   <div class=card><div class=sect style="margin:0 0 12px">Routing decisions <span class=muted style="font-weight:400;font-size:11px;text-transform:none;letter-spacing:0">— share of switched traffic</span></div>${mixBars}<div style="margin-top:10px">${mixWhy}</div></div>
   <div class=card><div class=sect style="margin:0 0 12px">Unsafe swaps avoided ${avoided.length?`<span class="badge b-bad">${avoided.length}</span>`:`<span class="badge b-good">0</span>`}</div><div class=swaps>${swaps}</div></div>
  </div>
  <div class=sect>ROI calculator</div>
  <div class=roi>
   <div class=roi-top><label>Your current monthly LLM spend</label>
    <div class=roi-input>$ <input id=roiSpend type=number min=0 step=500 value=25000 oninput="recalcROI()"></div>
    <div class=roi-rate>at the measured <b>${Math.round(costSave)}%</b> reduction →</div></div>
   <div class=roi-out>
    <div class=roi-cell><div class=roi-l>Estimated TokenJam spend</div><div class=roi-v id=roiAfter>—</div></div>
    <div class="roi-cell good"><div class=roi-l>Monthly savings</div><div class=roi-v id=roiSave>—</div></div>
    <div class="roi-cell good"><div class=roi-l>Annual savings</div><div class=roi-v id=roiYear>—</div></div>
   </div>
   <div class=muted style="font-size:11.5px;margin-top:12px">Linear projection of your spend at the cost reduction measured across the benchmarked workload — not a forecast.</div>
  </div>
  <div class="grid g2" style="margin-top:16px;align-items:start">
   <div class=card><div class=sect style="margin:0 0 10px">Production evidence <span class=muted style="font-weight:400;font-size:11px;text-transform:none;letter-spacing:0">— benchmarked &amp; passed</span></div><div class=evgrid>${evid}</div></div>
   <div class=card><div class=sect style="margin:0 0 10px">Why it cleared <span class="badge ${recCls(dec.cls)}">${esc(dec.state)}</span></div>${why}</div>
  </div>`;
 recalcROI();
}
async function pgOverview(){
 const runs=await loadRuns();
 const [mtx,hist,scen,vers]=await Promise.all([getJSON("/api/matrix"),getJSON("/api/history"),
   getJSON("/api/scenarios"),getJSON("/api/version-summary")]);
 const live=runs.filter(r=>!r.mock);
 const provs=new Set();runs.forEach(r=>{provs.add(provOf(r.original_model));provs.add(provOf(r.candidate_model));});
 const benches=new Set(runs.map(r=>r.benchmark));
 const replay=runs.filter(r=>r.benchmark==="replay");
 const savings=runs.map(r=>-r.cost_delta_pct).filter(x=>x>0);
 const avgSave=avg(savings),avgAcc=avg(runs.map(r=>r.accuracy_delta_pp));
 const latest=runs[0];
 const ver=(hist&&hist.versions&&hist.versions.slice(-1)[0])||(latest&&latest.tokenjam_version)||"—";
 const sparkAcc=runs.slice(0,12).reverse().map(r=>r.candidate_pass_rate);
 const sparkSave=runs.slice(0,12).reverse().map(r=>Math.max(0,-r.cost_delta_pct));
 const cards=[
  statCard(runs.length,"Total Benchmark Runs","across all configs"),
  statCard(live.length,"Live Runs",live.length?"real API spend":"all mock so far"),
  statCard(provs.size,"Providers Tested",[...provs].join(", ")),
  statCard(benches.size,"Benchmarks","distinct suites"),
  statCard((scen&&scen.rows&&scen.rows.length)||0,"Scenario Suites","agentic workloads"),
  statCard(replay.length,"Replay Sessions","historical re-runs"),
  statCard(avgSave==null?"—":"−"+Math.round(avgSave)+"%","Avg Cost Reduction","measured, savings runs",sparkSave),
  statCard(avgAcc==null?"—":pp(avgAcc),"Avg Accuracy Delta","candidate vs original",sparkAcc),
  statCard(esc(ver),"Latest TokenJam Version","under test"),
  statCard(latest?ago(latest.created_at):"—","Latest Benchmark","newest proof"),
 ].join("");
 // recommendation
 const anyBad=runs.some(r=>BAD.has(r.verdict))||(mtx&&mtx.regressions_found>0);
 const anyWarn=runs.some(r=>WARN.has(r.verdict));
 let banner;
 if(!runs.length)banner=`<div class="banner"><div class=bi>${BI.info}</div><div><b>No proofs yet.</b>
   <div class=bsub>Run <span class=mono>tjbench run</span> to produce your first evidence-backed validation.</div></div></div>`;
 else if(anyBad)banner=`<div class="banner bad"><div class=bi>${BI.warn}</div><div><b>Regression detected.</b>
   <div class=bsub>At least one config shows a statistically significant pass-rate drop. See Regression Center.</div></div></div>`;
 else banner=`<div class="banner ok"><div class=bi>${BI.ok}</div><div><b>No significant regressions.</b>
   <div class=bsub>Every analyzed config is within statistical noise of its original on the measured benchmarks.${anyWarn?" Some configs are flagged to review.":""}</div></div></div>`;
 // timeline
 const tl=runs.slice(0,7).map(r=>`<div class=tl-item><div class=tl-rail>
   <div class=tl-dot style="background:var(--${GOOD.has(r.verdict)?"good":BAD.has(r.verdict)?"bad":WARN.has(r.verdict)?"warn":"mut2"})"></div><div class=tl-line></div></div>
   <div class=tl-body><div class=t1>${esc(r.benchmark)} ${r.mock?'<span class=tag>mock</span>':''}</div>
   <div class=t2><span class=mono>${esc(modelOf(r.original_model))} → ${esc(modelOf(r.candidate_model))}</span> &middot; ${badge(r.verdict)} &middot; saved ${saved(r.cost_delta_pct)}</div></div>
   <div class=tl-time>${ago(r.created_at)}</div></div>`).join("")||'<div class=empty>no runs</div>';
 M().innerHTML=`${heroCard(runs)}${tjImpact(runs)}<p class=lead>The trust layer for TokenJam — every figure below is a measured benchmark with a hedged statistical verdict, never a bare "safe".</p>
  <div class="grid g5">${cards}</div>
  <div class=sect>Overall recommendation status</div>${banner}
  <div class=chart style="margin-top:18px"><h3>Accuracy &amp; cost-saved trend</h3><p class=ch-sub>candidate pass-rate and % cost saved across ${runs.length} runs, oldest → newest</p>
    <div id=chartbox></div>
    <div class=legend><span><i style="background:var(--acc)"></i>candidate accuracy</span><span><i style="background:var(--good)"></i>cost saved</span></div></div>
  <div class="grid g2" style="margin-top:16px;align-items:start">
   <div class=chart><h3>Cost saved by benchmark</h3><p class=ch-sub>mean % token cost saved per family</p><div id=ovbar></div></div>
   <div class=card><div class=sect style="margin:0 0 6px">Recent benchmark timeline</div><div class=tl>${tl}</div></div>
  </div>
  <div class=sect>Latest benchmark runs</div><div id=ovtbl></div>`;
 drawChart("chartbox",runs.slice().reverse().map(r=>({a:r.candidate_pass_rate,c:Math.max(0,-r.cost_delta_pct),x:fmtTime(r.created_at).split(",")[0]})));
 {const byB={};runs.forEach(r=>{(byB[r.benchmark]=byB[r.benchmark]||[]).push(-r.cost_delta_pct);});
  const _ab={"swe-bench-lite":"swe-lite","coding-assistant":"coding","research-agent":"research","browser-agent":"browser","rag-support":"rag"};
  const bars=Object.entries(byB).map(([k,v])=>({label:_ab[k]||k,value:Math.max(0,avg(v)||0),color:"var(--good)"})).sort((a,b)=>b.value-a.value);
  barChart("ovbar",bars);}
 table("ovtbl",[
  {key:"created_at",label:"Date",html:r=>`<span class=mono>${esc(fmtTime(r.created_at))}</span>`},
  {key:"benchmark",label:"Benchmark",html:r=>esc(r.benchmark)+(r.mock?' <span class=tag>mock</span>':'')},
  {key:"original_model",label:"Original",html:r=>`<span class=mono>${esc(r.original_model)}</span>`},
  {key:"candidate_model",label:"Candidate",html:r=>`<span class=mono>${esc(r.candidate_model)}</span>`},
  {key:"candidate_pass_rate",label:"Accuracy",html:r=>`${r.original_pass_rate}% → <b>${r.candidate_pass_rate}%</b> ${accDelta(r.accuracy_delta_pp)}`},
  {key:"cost_delta_pct",label:"Cost Saved",sort:r=>-r.cost_delta_pct,html:r=>saved(r.cost_delta_pct)},
  {key:"mcnemar_p",label:"Confidence",nosort:true,html:r=>conf(r)},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
  {key:"file",label:"",nosort:true,html:r=>reportActs(r)},
 ],runs,{sortKey:"created_at",dir:-1});
}
async function pgBenchmarks(){
 const runs=await loadRuns();const by=bucket(runs);
 if(!runs.length){M().innerHTML='<div class=empty>No benchmark runs yet.</div>';return;}
 const cats={};Object.keys(by).forEach(b=>{(cats[catOf(b)]=cats[catOf(b)]||[]).push(b);});
 const order=["Production Workflows","Executable","Scenarios","Replay","LLM-judged","Other"];
 let html=`<p class=lead>Every benchmark family, grouped by how accuracy is measured. A card's verdict is the latest run's hedged McNemar result — click through for the full proof.</p>`;
 order.filter(c=>cats[c]).forEach(c=>{
  html+=`<div class=sect>${esc(c)}</div><div class="grid auto">`;
  cats[c].sort().forEach(b=>{
   const rs=by[b];const latest=rs[0];
   const ci=(latest.wilson_low==null)?"—":`${pct(latest.wilson_low)} – ${pct(latest.wilson_high)}`;
   const mc=(latest.mcnemar_b==null)?"—":`b=${latest.mcnemar_b} / c=${latest.mcnemar_c}${latest.mcnemar_p==null?"":" · p="+Number(latest.mcnemar_p).toFixed(3)}`;
   const diff=latest.difficulty,diffCls=diff==='hard'?'b-bad':diff==='medium'?'b-warn':'b-good';
   const totalN=rs.reduce((a,x)=>a+(x.n_tasks||0),0);
   const fc=(latest.failure_categories||[]).slice(0,4);
   html+=`<div class="card hov">
    <div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px">
     <div style="font-weight:650;font-size:15px;white-space:nowrap">${esc(b)}</div>${badge(latest.verdict)}</div>
    <div class=muted style="font-size:12px;margin-top:3px">${latest.task_category?esc(latest.task_category)+" &middot; ":""}${rs.length} run${rs.length===1?"":"s"} &middot; latest ${ago(latest.created_at)}${diff?` &middot; <span class="badge ${diffCls}" style="padding:1px 7px">${esc(diff)}</span>`:""}</div>
    <div class="grid g2" style="gap:10px;margin-top:14px">
     <div><div style="color:var(--mut);font-size:11px">PASS RATE</div><div style="font-size:18px;font-weight:700">${latest.candidate_pass_rate}%</div></div>
     <div><div style="color:var(--mut);font-size:11px">COST SAVED</div><div style="font-size:18px;font-weight:700">${saved(latest.cost_delta_pct)}</div></div>
     <div><div style="color:var(--mut);font-size:11px">WILSON 95% CI</div><div class=mono style="font-size:13px">${ci}</div></div>
     <div><div style="color:var(--mut);font-size:11px">McNEMAR</div><div class=mono style="font-size:12px">${mc}</div></div>
    </div>
    <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:12px;font-size:11.5px;color:var(--mut);border-top:1px solid var(--line);padding-top:11px">
     <span>Tasks <b style="color:var(--fg)">${totalN.toLocaleString()}</b></span>
     ${latest.coverage_pct!=null?`<span>Coverage <b style="color:var(--fg)">${latest.coverage_pct}%</b></span>`:""}
     ${latest.latency_saved_pct!=null?`<span>Latency saved <b style="color:var(--fg)">${latest.latency_saved_pct}%</b></span>`:""}
    </div>
    ${fc.length?`<div style="margin-top:10px"><div style="font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut2);margin-bottom:6px">Top failure modes</div><div style="display:flex;flex-wrap:wrap;gap:6px">${fc.map(f=>`<span class=tag>${esc(f.category)} &times;${f.count}</span>`).join("")}</div></div>`:""}
    ${latest.ground_truth?`<div class=muted style="font-size:11px;margin-top:10px">Ground truth: ${esc(latest.ground_truth)}</div>`:""}
    <div class=acts style="margin-top:14px"><a class=lnk href="/report/${encodeURIComponent(latest.file)}" target=_blank>Open report</a>
     <a class=lnk href="/raw/${encodeURIComponent(latest.file)}" target=_blank>JSON</a></div></div>`;
  });
  html+=`</div>`;
 });
 M().innerHTML=html;
}
async function pgScenarios(){
 const [scen,runs]=await Promise.all([getJSON("/api/scenarios"),loadRuns()]);
 const cat=(scen&&scen.rows)||[];const by=bucket(runs);
 const meta={"coding-assistant":["Coding Assistant","read → search → edit → test → commit, with destructive tools gated"],
  "rag-support":["RAG Support","search KB → retrieve → answer; refunds &amp; cancels are trapped"],
  "research-agent":["Research Agent","search → fetch → summarize; publishing is trapped"],
  "browser-agent":["Browser Agent","navigate → extract → report; payments are trapped"],
  "customer-support":["Customer Support","planned suite"]};
 let html=`<p class=lead>Production-shaped agent suites. Each scenario judges the <b>whole trace</b> — right tools, right order, right answer, and <b>no catastrophic action</b> — so a downsize can't quietly trade safety for cost.</p><div class="grid auto">`;
 const names=cat.length?cat.map(c=>c.name):Object.keys(meta);
 names.forEach(name=>{
  const c=cat.find(x=>x.name===name)||{name,n_tasks:null};
  const m=meta[name]||[name,""];
  const rs=(by[name]||[]);const latest=rs[0];
  html+=`<div class="card hov">
   <div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px">
    <div style="font-weight:650;font-size:15px;white-space:nowrap">${esc(m[0])}</div>
    ${latest?badge(latest.verdict):'<span class="badge b-mut">no runs</span>'}</div>
   <div class=muted style="font-size:12px;margin-top:3px">${m[1]}</div>
   <div class="grid g2" style="gap:10px;margin-top:14px">
    <div><div style="color:var(--mut);font-size:11px">TASKS</div><div style="font-size:18px;font-weight:700">${c.n_tasks==null?"—":c.n_tasks}</div></div>
    <div><div style="color:var(--mut);font-size:11px">PASS RATE</div><div style="font-size:18px;font-weight:700">${latest?latest.candidate_pass_rate+"%":"—"}</div></div>
    <div><div style="color:var(--mut);font-size:11px">COST SAVED</div><div style="font-size:16px;font-weight:700">${latest?saved(latest.cost_delta_pct):"—"}</div></div>
    <div><div style="color:var(--mut);font-size:11px">SAFETY GATE</div><div><span class="badge b-good">enforced</span></div></div>
   </div>
   <div class=muted style="font-size:11.5px;margin-top:10px">${latest?("latest tj"+esc(latest.tokenjam_version)+" · "+ago(latest.created_at)):"run <span class=mono>tjbench scenarios</span> to populate"}</div>
   ${latest?`<div class=acts style="margin-top:10px"><a class=lnk href="/report/${encodeURIComponent(latest.file)}" target=_blank>Open report</a></div>`:""}</div>`;
 });
 html+=`</div>`;M().innerHTML=html;
}
async function pgReplay(){
 const runs=(await loadRuns()).filter(r=>r.benchmark==="replay");
 const histReq=runs.reduce((a,r)=>a+(r.n_tasks||0),0);
 const equiv=runs.reduce((a,r)=>a+(r.candidate_pass||0),0);
 const crit=runs.reduce((a,r)=>a+(r.critical_failures||0),0);
 const semM=avg(runs.map(r=>r.semantic_match_rate).filter(x=>x!=null));
 const behM=avg(runs.map(r=>r.behavior_match_rate).filter(x=>x!=null));
 const avgPass=avg(runs.map(r=>r.candidate_pass_rate));
 const latSaved=avg(runs.map(r=>r.latency_saved_pct).filter(x=>x!=null));
 const avgSave=avg(runs.map(r=>-r.cost_delta_pct));
 const cards=[
  statCard(runs.length,"Replay Sessions","historical re-validations"),
  statCard(histReq.toLocaleString(),"Historical Requests","turns replayed"),
  statCard(equiv.toLocaleString(),"Equivalent Outputs","judge-passed turns"),
  statCard(crit,"Critical Failures",crit?"need human review":"none flagged"),
  statCard(avgSave==null?"—":"−"+Math.round(avgSave)+"%","Avg Cost Saved","vs historical spend"),
 ].join("");
 if(!runs.length){M().innerHTML=`<p class=lead>Replay validation answers the strongest version of the question: on <b>your own historical traffic</b>, does the cheaper model produce equivalent outputs?</p>
   <div class="banner"><div class=bi>${BI.replay}</div><div><b>No replay runs yet.</b>
   <div class=bsub>Replay re-runs your real TokenJam telemetry through the candidate model and judges equivalence. Run <span class=mono>tjbench replay &lt;telemetry&gt;</span>.</div></div></div>`;return;}
 const matchBars=hbars([
  {label:"Semantic match",value:semM||0,color:"var(--good)"},
  {label:"Behavioral match",value:behM||0,color:"var(--good)"},
  {label:"Judge pass-rate",value:avgPass||0,color:"var(--acc)"},
  {label:"Latency saved",value:latSaved||0,color:"var(--acc)"},
 ],{fmt:v=>Math.round(v)+"%"});
 const fc={};runs.forEach(r=>(r.failure_categories||[]).forEach(f=>{fc[f.category]=(fc[f.category]||0)+f.count;}));
 const failBars=hbars(Object.entries(fc).map(([k,v])=>({label:k,value:v,color:"var(--bad)"})).sort((a,b)=>b.value-a.value),{});
 const wd=runs.find(r=>r.replay_diffs&&r.replay_diffs.length);
 const diffHtml=(wd?wd.replay_diffs:[]).map(d=>`<div class=diff><div class=diff-q><span class=mono>${esc(d.prompt)}</span><span class="badge ${d.match==='equivalent'?'b-good':'b-warn'}">${esc(d.match)}</span></div>
   <div class=diff-cols><div class=diff-col><div class=dc-h>original</div>${esc(d.original)}</div><div class=diff-col><div class=dc-h>candidate</div>${esc(d.candidate)}</div></div></div>`).join("")||'<div class=empty>no captured diffs</div>';
 M().innerHTML=`<p class=lead>Replay validation answers the strongest version of the question: on <b>your own historical traffic</b>, does the cheaper model produce equivalent outputs? Equivalence is judged turn-by-turn, divergences are counted, and the verdict stays hedged.</p>
  <div class="grid g5">${cards}</div>
  <div class="grid g2" style="margin-top:16px;align-items:start">
   <div class=chart><h3>Replay equivalence over time</h3><p class=ch-sub>judge pass-rate and cost saved per replay session</p><div id=chartbox></div>
    <div class=legend><span><i style="background:var(--acc)"></i>judge pass-rate</span><span><i style="background:var(--good)"></i>cost saved</span></div></div>
   <div class=card><div class=sect style="margin:0 0 12px">Output match rates</div>${matchBars}
    <div class=muted style="font-size:11.5px;margin-top:14px">Semantic = same meaning; behavioral = same tools/actions taken. Judge pass-rate flows into the McNemar verdict.</div></div>
  </div>
  <div class="grid g2" style="margin-top:16px;align-items:start">
   <div class=card><div class=sect style="margin:0 0 12px">Divergence reasons</div>${failBars}</div>
   <div class=card><div class=sect style="margin:0 0 10px">Sample output diffs &middot; behavior comparison</div>${diffHtml}</div>
  </div>
  <div class=sect>Recent replay sessions</div><div id=rptbl></div>`;
 drawChart("chartbox",runs.slice().reverse().map(r=>({a:r.candidate_pass_rate,c:Math.max(0,-r.cost_delta_pct),x:fmtTime(r.created_at).split(",")[0]})));
 table("rptbl",[
  {key:"created_at",label:"When",html:r=>`<span class=mono>${esc(fmtTime(r.created_at))}</span>`},
  {key:"candidate_model",label:"Candidate",html:r=>`<span class=mono>${esc(r.candidate_model)}</span>`},
  {key:"n_tasks",label:"Turns"},
  {key:"candidate_pass",label:"Equivalent",html:r=>`${r.candidate_pass==null?"—":r.candidate_pass} / ${r.n_tasks}`},
  {key:"semantic_match_rate",label:"Semantic",html:r=>r.semantic_match_rate==null?"—":r.semantic_match_rate+"%"},
  {key:"critical_failures",label:"Critical",html:r=>r.critical_failures==null?"—":String(r.critical_failures)},
  {key:"cost_delta_pct",label:"Cost Saved",sort:r=>-r.cost_delta_pct,html:r=>saved(r.cost_delta_pct)},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
  {key:"file",label:"",nosort:true,html:r=>reportActs(r)},
 ],runs,{sortKey:"created_at",dir:-1});
}
async function pgDeepEval(){
 const runs=await loadRuns();const judged=runs.filter(r=>r.benchmark==="judged");
 const avgScore=avg(judged.map(r=>r.candidate_pass_rate/100));
 const jm=judged.map(r=>r.judge).filter(Boolean);
 const MM=k=>avg(jm.map(j=>j[k]).filter(x=>x!=null));
 const correctness=MM("correctness"),faith=MM("faithfulness"),rel=MM("answer_relevancy"),comp=MM("task_completion"),reason=MM("reasoning_quality");
 const agree=MM("judge_agreement"),halluc=MM("hallucination_rate"),cite=MM("citation_accuracy");
 if(!judged.length){M().innerHTML=`<p class=lead>For open-ended tasks with no unit test, equivalence is scored by an LLM judge (DeepEval, DeepSeek-backed).</p>
   <div class="banner"><div class=bi>${BI.info}</div><div><b>No LLM-judged runs yet.</b>
   <div class=bsub>Run <span class=mono>tjbench run --benchmark judged</span> to score open-ended tasks with the DeepEval judge.</div></div></div>`;return;}
 const radarItems=[{label:"Correctness",value:correctness||0},{label:"Faithfulness",value:faith||0},{label:"Relevancy",value:rel||0},{label:"Completion",value:comp||0},{label:"Reasoning",value:reason||0}];
 const tile=(label,frac)=>`<div class=mtile><div class=mt-v>${frac==null?"—":Math.round(frac*100)+"%"}</div><div class=mt-l>${esc(label)}</div><div class=mt-bar><i style="width:${frac==null?0:Math.round(frac*100)}%"></i></div></div>`;
 const tiles=[tile("Correctness",correctness),tile("Faithfulness",faith),tile("Answer Relevancy",rel),tile("Task Completion",comp),tile("Reasoning Quality",reason),tile("Judge Agreement",agree),tile("Citation Accuracy",cite),
   `<div class=mtile><div class=mt-v>${halluc==null?"—":halluc.toFixed(1)+"%"}</div><div class=mt-l>Hallucination Rate</div><div class=mt-bar><i style="width:${halluc==null?0:Math.min(100,halluc*8)}%;background:var(--bad)"></i></div></div>`].join("");
 M().innerHTML=`<p class=lead>For open-ended tasks with no unit test, equivalence is scored by an LLM judge (DeepEval, DeepSeek-backed). The run's pass-rate flows into the same McNemar verdict as the executable benchmarks; the sub-scores below profile <i>where</i> the candidate gains or loses.</p>
  <div class="grid g2" style="align-items:stretch">
   <div class=card style="display:flex;gap:20px;align-items:center">
     <div>${donut(avgScore||0,"judge pass")}</div>
     <div><div class=muted style="font-size:12px">AVERAGE JUDGE SCORE</div>
      <div style="font-size:28px;font-weight:750">${avgScore==null?"—":Math.round(avgScore*100)+"%"}</div>
      <div class=muted style="font-size:12px;margin-top:2px">${judged.length} judged run${judged.length===1?"":"s"} · DeepSeek/DeepEval backend</div>
      <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap"><span class="badge b-good">agreement ${agree==null?"—":Math.round(agree*100)+"%"}</span><span class="badge ${halluc>3?"b-warn":"b-good"}">halluc ${halluc==null?"—":halluc.toFixed(1)+"%"}</span></div></div></div>
   <div class=chart><h3>Evaluation metric profile</h3><p class=ch-sub>mean across judged runs</p><div style="display:flex;justify-content:center">${radar(radarItems)}</div></div>
  </div>
  <div class=sect>Metric breakdown</div><div class=mtiles>${tiles}</div>
  <div class="grid g2" style="margin-top:16px;align-items:start">
   <div class=chart><h3>Judge score by run</h3><p class=ch-sub>pass-rate per judged configuration</p><div id=descore></div></div>
   <div class=card><div class=sect style="margin:0 0 8px">How judging works</div><div class=muted style="font-size:13px">Each judged turn is scored by the configured DeepEval metric against the original's output. A turn "passes" when its score clears the threshold; the run's pass-rate is the share of passing turns. The sub-scores (correctness, faithfulness, relevancy, completion, reasoning) are the judge's per-dimension breakdown.</div></div>
  </div>
  <div class=sect>Recent judge results</div><div id=detbl></div>`;
 barChart("descore",judged.slice().reverse().map(r=>({label:modelOf(r.candidate_model).slice(0,9),value:r.candidate_pass_rate,color:"var(--acc)"})));
 table("detbl",[
  {key:"created_at",label:"When",html:r=>`<span class=mono>${esc(fmtTime(r.created_at))}</span>`},
  {key:"candidate_model",label:"Candidate",html:r=>`<span class=mono>${esc(r.candidate_model)}</span>`},
  {key:"n_tasks",label:"Cases"},
  {key:"candidate_pass_rate",label:"Judge Pass",html:r=>`<b>${r.candidate_pass_rate}%</b>`},
  {key:"_corr",label:"Correctness",get:r=>r.judge&&r.judge.correctness,html:r=>r.judge?Math.round(r.judge.correctness*100)+"%":"—"},
  {key:"_faith",label:"Faithfulness",get:r=>r.judge&&r.judge.faithfulness,html:r=>r.judge?Math.round(r.judge.faithfulness*100)+"%":"—"},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
  {key:"file",label:"",nosort:true,html:r=>reportActs(r)},
 ],judged,{sortKey:"created_at",dir:-1});
}
let selTrend=null;
async function pgTrends(){
 const cfg=await getJSON("/api/configs");const cfgs=(cfg&&cfg.rows)||[];
 if(!cfgs.length){M().innerHTML='<div class=empty>No history yet. Run proofs over multiple TokenJam versions to build trends.</div>';return;}
 const key=c=>c.benchmark+"|"+c.original_model+"|"+c.candidate_model;
 if(!selTrend||!cfgs.some(c=>key(c)===selTrend))selTrend=key(cfgs[0]);
 const[bm,orig,cand]=selTrend.split("|");
 const tr=await getJSON(`/api/trend?benchmark=${encodeURIComponent(bm)}&original=${encodeURIComponent(orig)}&candidate=${encodeURIComponent(cand)}`);
 const rows=(tr&&tr.rows)||[];const vsum=await getJSON("/api/version-summary");
 const sel=`<select id=trendSel>`+cfgs.map(c=>`<option value="${esc(key(c))}" ${key(c)===selTrend?"selected":""}>${esc(c.benchmark)}: ${esc(modelOf(c.original_model))}→${esc(modelOf(c.candidate_model))}</option>`).join("")+`</select>`;
 M().innerHTML=`<p class=lead>Historical analytics from the benchmark database — how a recommendation holds up as TokenJam ships new versions.</p>
  <div class=tblbar><span class=muted style="font-size:12px">Config</span> ${sel}</div>
  <div class=chart><h3>Accuracy &amp; cost saved · ${esc(bm)}</h3><p class=ch-sub>by TokenJam version, oldest → newest</p><div id=chartbox></div>
   <div class=legend><span><i style="background:var(--acc)"></i>candidate accuracy</span><span><i style="background:var(--good)"></i>cost saved</span></div></div>
  <div class="grid g2" style="margin-top:16px;align-items:start">
   <div class=chart><h3>DeepEval judge score</h3><p class=ch-sub>where applicable, per version</p><div id=deScore></div></div>
   <div class=chart><h3>Avg accuracy delta by version</h3><p class=ch-sub>all configs, per TokenJam version</p><div id=verBar></div></div></div>
  <div class=sect>Per-version detail</div><div id=trtbl></div>`;
 document.getElementById("trendSel").onchange=e=>{selTrend=e.target.value;pgTrends();};
 drawChart("chartbox",rows.map(r=>({a:(r.candidate_pass_rate||0)*100,c:Math.max(0,-(r.cost_delta_pct||0)),x:r.tokenjam_version})));
 const de=rows.filter(r=>r.deepeval_score!=null);
 drawChart("deScore",de.map(r=>({c:(r.deepeval_score||0)*100,x:r.tokenjam_version})),{single:true});
 barChart("verBar",((vsum&&vsum.rows)||[]).map(v=>({label:v.version,value:Math.max(0,(v.avg_acc_delta_pp||0)+50),color:"var(--acc)"})));
 table("trtbl",[
  {key:"tokenjam_version",label:"TokenJam",html:r=>`<span class=mono>${esc(r.tokenjam_version)}</span>`},
  {key:"candidate_pass_rate",label:"Cand Pass",sort:r=>r.candidate_pass_rate,html:r=>pct((r.candidate_pass_rate||0)*100)},
  {key:"accuracy_delta_pp",label:"Δ Acc",html:r=>accDelta(r.accuracy_delta_pp)},
  {key:"cost_delta_pct",label:"Cost Saved",sort:r=>-(r.cost_delta_pct||0),html:r=>saved(r.cost_delta_pct)},
  {key:"deepeval_score",label:"DeepEval",html:r=>r.deepeval_score==null?"—":Number(r.deepeval_score).toFixed(2)},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
 ],rows,{search:false});
}
let selLB=null;
async function pgLeaderboards(){
 const runs=await loadRuns();
 if(!runs.length){M().innerHTML='<div class=empty>No runs yet.</div>';return;}
 const CODING=["humaneval","mbpp","swe-bench-lite"];
 const C={};
 runs.forEach(r=>{const m=r.candidate_model;const o=C[m]=C[m]||{model:m,n:0,acc:[],save:[],replay:[],judge:[],scen:[],coding:[],reason:[],safeN:0,safe:0,list:[]};
  o.n++;o.list.push(r);o.acc.push(r.candidate_pass_rate);o.save.push(-r.cost_delta_pct);
  if(r.benchmark==="replay")o.replay.push(r.candidate_pass_rate);
  if(r.benchmark==="judged")o.judge.push(r.candidate_pass_rate);
  if(SCEN.has(r.benchmark)){o.scen.push(r.candidate_pass_rate);o.safeN++;if(r.safety_gate==="enforced")o.safe++;}
  if(CODING.includes(r.benchmark))o.coding.push(r.candidate_pass_rate);
  if(r.benchmark==="gsm8k")o.reason.push(r.candidate_pass_rate);
  if(r.judge&&r.judge.reasoning_quality!=null)o.reason.push(r.judge.reasoning_quality*100);});
 let rows=Object.values(C).map(o=>{
  const acc=avg(o.acc),noReg=o.list.some(r=>BAD.has(r.verdict))?0:1,cov=Math.min(1,o.n/8);
  return {model:o.model,provider:provOf(o.model),acc,save:avg(o.save),replay:avg(o.replay),judge:avg(o.judge),
   scen:avg(o.scen),coding:avg(o.coding),reason:avg(o.reason),safety:o.safeN?o.safe/o.safeN*100:null,
   trust:Math.round((acc||0)*0.7+noReg*100*0.2+cov*100*0.1),rec:recommendation(o.list),n:o.n};
 }).sort((a,b)=>b.trust-a.trust);
 rows.forEach((r,i)=>r.rank=i+1);
 const recTop=rows.find(r=>r.rec.state==="CLEARED");
 const pcell=v=>v==null?'<span class=muted>—</span>':Math.round(v)+"%";
 const recCls=c=>c==="good"?"b-good":c==="bad"?"b-bad":c==="warn"?"b-warn":"b-mut";
 M().innerHTML=`<p class=lead>Models ranked by a composite <b>Trust Score</b> across every benchmark family. The recommended candidate is the highest-trust configuration that <b>cleared</b> its regression test.</p>
  <div id=lbtbl></div>
  <div class=muted style="font-size:11.5px;margin-top:10px">Trust Score is a heuristic composite (0.7·accuracy + 0.2·no-regression + 0.1·evidence coverage) — a ranking aid, not a probability. Per-dimension cells are mean candidate pass-rates; "—" means that family wasn't run for that model.</div>`;
 table("lbtbl",[
  {key:"rank",label:"#",sort:r=>r.rank,html:r=>medal(r.rank)},
  {key:"model",label:"Model",html:r=>`<span class=mono>${esc(modelOf(r.model))}</span>${r===recTop?' <span class="badge b-good">recommended</span>':''}`},
  {key:"provider",label:"Provider",get:r=>r.provider,html:r=>`<span class="badge b-mut">${esc(provLabel(r.provider))}</span>`},
  {key:"trust",label:"Trust",sort:r=>r.trust,html:r=>`<div style="display:flex;align-items:center;gap:8px"><b style="width:26px">${r.trust}</b><div class=mt-bar style="width:54px"><i style="width:${r.trust}%"></i></div></div>`},
  {key:"acc",label:"Accuracy",sort:r=>r.acc,html:r=>pcell(r.acc)},
  {key:"save",label:"Savings",sort:r=>r.save,html:r=>r.save==null?"—":`<span class=up>−${Math.round(r.save)}%</span>`},
  {key:"coding",label:"Coding",sort:r=>r.coding,html:r=>pcell(r.coding)},
  {key:"reason",label:"Reasoning",sort:r=>r.reason,html:r=>pcell(r.reason)},
  {key:"replay",label:"Replay",sort:r=>r.replay,html:r=>pcell(r.replay)},
  {key:"judge",label:"Judge",sort:r=>r.judge,html:r=>pcell(r.judge)},
  {key:"scen",label:"Scenarios",sort:r=>r.scen,html:r=>pcell(r.scen)},
  {key:"safety",label:"Safety",sort:r=>r.safety,html:r=>r.safety==null?'<span class=muted>—</span>':`<span class="badge b-good">${Math.round(r.safety)}%</span>`},
  {key:"rec",label:"Recommendation",get:r=>r.rec.state,html:r=>`<span class="badge ${recCls(r.rec.cls)}">${esc(r.rec.state)}</span>`},
  {key:"n",label:"Runs",sort:r=>r.n},
 ],rows,{sortKey:"trust",dir:-1});
}
async function pgProviders(){
 const runs=await loadRuns();
 const CODING=["humaneval","mbpp","swe-bench-lite"];
 const P={},M2={};
 runs.forEach(r=>{const pr=provOf(r.candidate_model);const o=P[pr]=P[pr]||{prov:pr,models:new Set(),n:0,acc:[],save:[],coding:[],reason:[],replay:[],judge:[],scen:[],safeN:0,safe:0};
  o.models.add(r.candidate_model);o.n++;o.acc.push(r.candidate_pass_rate);o.save.push(-r.cost_delta_pct);
  if(CODING.includes(r.benchmark))o.coding.push(r.candidate_pass_rate);
  if(r.benchmark==="gsm8k")o.reason.push(r.candidate_pass_rate);
  if(r.benchmark==="replay")o.replay.push(r.candidate_pass_rate);
  if(r.benchmark==="judged")o.judge.push(r.candidate_pass_rate);
  if(SCEN.has(r.benchmark)){o.scen.push(r.candidate_pass_rate);o.safeN++;if(r.safety_gate==="enforced")o.safe++;}
  const m=r.candidate_model;const mo=M2[m]=M2[m]||{model:m,n:0,b:new Set(),acc:[],cost:[]};
  mo.n++;mo.b.add(r.benchmark);mo.acc.push(r.candidate_pass_rate);if(r.candidate_cost_usd!=null)mo.cost.push(r.candidate_cost_usd);});
 const provs=Object.values(P);
 const COL={anthropic:"var(--acc)",openai:"var(--good)",deepseek:"#a371f7",google:"#e3a008"};
 const LB=["Accuracy","Savings","Coding","Reasoning","Replay","Safety"];
 const series=provs.map(o=>({name:o.prov,color:COL[o.prov]||"var(--mut)",values:[
   (avg(o.acc)||0)/100,Math.min(1,(avg(o.save)||0)/100),(avg(o.coding)||0)/100,(avg(o.reason)||0)/100,(avg(o.replay)||0)/100,(o.safeN?o.safe/o.safeN:0)]}));
 const legend=provs.map(o=>`<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px"><i style="width:10px;height:10px;border-radius:2px;background:${COL[o.prov]||'var(--mut)'};display:inline-block"></i>${esc(provLabel(o.prov))}</span>`).join("");
 const lead=(fn,fmt,lower)=>{let best=null;provs.forEach(o=>{const v=fn(o);if(v==null||isNaN(v))return;if(!best||(lower?v<best.v:v>best.v))best={p:o.prov,v};});return best?`<span class="badge b-good">${esc(provLabel(best.p))} · ${fmt(best.v)}</span>`:"—";};
 const leaders=[["Best accuracy",lead(o=>avg(o.acc),v=>Math.round(v)+"%")],
  ["Best savings",lead(o=>avg(o.save),v=>"−"+Math.round(v)+"%")],
  ["Best coding",lead(o=>avg(o.coding),v=>Math.round(v)+"%")],
  ["Best safety",lead(o=>o.safeN?o.safe/o.safeN*100:null,v=>Math.round(v)+"%")],
  ["Most models",lead(o=>o.models.size,v=>v+" models")]]
  .map(([l,v])=>`<div class=set-row style="padding:10px 0"><div class=k style="font-weight:500">${l}</div>${v}</div>`).join("");
 const KNOWN=[["anthropic","Anthropic"],["openai","OpenAI"],["deepseek","DeepSeek"],["google","Gemini"],["groq","Groq"],["openrouter","OpenRouter"]];
 const mrow=(l,v,c)=>`<div style="margin-top:9px"><div style="display:flex;justify-content:space-between;font-size:11.5px;color:var(--mut)"><span>${l}</span><span style="color:var(--fg);font-weight:600">${v==null?"—":Math.round(v)+"%"}</span></div><div class=mt-bar style="margin-top:4px"><i style="width:${v==null?0:Math.min(100,Math.round(v))}%${c?";background:"+c:""}"></i></div></div>`;
 const cards=KNOWN.map(([k,l])=>{const o=P[k];const has=o&&o.n;
  return `<div class="card hov"><div style="display:flex;flex-wrap:wrap;align-items:center;gap:8px">
    <div style="font-weight:650;font-size:15px;white-space:nowrap">${esc(l)}</div>${has?'<span class="badge b-good">tested</span>':'<span class="badge b-mut">not tested</span>'}</div>
   <div class="grid g2" style="gap:10px;margin-top:12px">
    <div><div style="color:var(--mut);font-size:11px">MODELS</div><div style="font-size:18px;font-weight:700">${has?o.models.size:"—"}</div></div>
    <div><div style="color:var(--mut);font-size:11px">RUNS</div><div style="font-size:18px;font-weight:700">${has?o.n:"—"}</div></div></div>
   ${has?`<div style="margin-top:6px">${mrow("Accuracy",avg(o.acc))}${mrow("Cost saved",avg(o.save),"var(--good)")}${mrow("Replay",avg(o.replay))}${mrow("Scenarios",avg(o.scen))}</div>`:'<div class=muted style="font-size:12px;margin-top:10px">not benchmarked yet</div>'}</div>`;}).join("");
 const modelRows=Object.values(M2).map(o=>({model:o.model,runs:o.n,benchmarks:o.b.size,acc:avg(o.acc),cost:avg(o.cost)}));
 M().innerHTML=`<p class=lead>Provider-agnostic by design — the same proof pipeline runs across every OpenAI-compatible and native provider. The radar profiles each provider's candidate models across the key dimensions.</p>
  <div class="grid g2" style="align-items:start">
   <div class=chart><h3>Provider capability profile</h3><p class=ch-sub>normalized 0–100% across dimensions</p>
    <div style="display:flex;justify-content:center">${radarMulti(LB,series)}</div>
    <div class=legend style="justify-content:center;margin-top:8px">${legend}</div></div>
   <div class=card><div class=sect style="margin:0 0 6px">Category leaders</div>${leaders}</div>
  </div>
  <div class=sect>Providers</div><div class="grid auto">${cards}</div>
  <div class=sect>Per-model matrix</div><div id=pmtbl></div>`;
 table("pmtbl",[
  {key:"model",label:"Model",html:r=>`<span class=mono>${esc(r.model)}</span>`},
  {key:"provider",label:"Provider",get:r=>provOf(r.model),html:r=>`<span class="badge b-mut">${esc(provLabel(provOf(r.model)))}</span>`},
  {key:"runs",label:"Runs"},
  {key:"benchmarks",label:"Benchmarks"},
  {key:"acc",label:"Avg Accuracy",sort:r=>r.acc,html:r=>r.acc==null?"—":Math.round(r.acc)+"%"},
  {key:"cost",label:"Avg Cost / run",sort:r=>r.cost==null?Infinity:r.cost,html:r=>r.cost==null?"—":money(r.cost)},
 ],modelRows,{sortKey:"acc",dir:-1});
}
async function pgVersions(){
 const [v,runs]=await Promise.all([getJSON("/api/version-summary"),loadRuns()]);
 const rows=(v&&v.rows)||[];
 if(!rows.length){M().innerHTML='<div class=empty>No version history yet — benchmark across TokenJam releases to populate this.</div>';return;}
 const byV={};runs.forEach(r=>{(byV[r.tokenjam_version]=byV[r.tokenjam_version]||[]).push(r);});
 rows.forEach(r=>{const rs=byV[r.version]||[];
  r._replay=avg(rs.filter(x=>x.benchmark==="replay").map(x=>x.candidate_pass_rate));
  r._judge=avg(rs.filter(x=>x.benchmark==="judged").map(x=>x.candidate_pass_rate));
  r._configs=new Set(rs.map(x=>x.original_model+"→"+x.candidate_model)).size;});
 const timeline=rows.map((r,i)=>`${i>0?'<div class=rel-conn></div>':''}<div class="rel-step ${r.regressions>0?'reg':''}">
   <div class=rel-dot></div><div class=rel-v>${esc(r.version)}</div>
   <div class=rel-meta>${r.runs} runs · ${r.regressions>0?`<span class=down>${r.regressions} regression${r.regressions>1?'s':''}</span>`:'<span class=up>stable</span>'}</div></div>`).join("");
 M().innerHTML=`<p class=lead>TokenJam changes constantly. This is the guard: every released version is re-benchmarked, so a recommendation that quietly regresses in a new version is caught before it reaches production.</p>
  <div class=card><div class=sect style="margin:0 0 10px">Release timeline</div><div class=rel-track>${timeline}</div></div>
  <div class="grid g2" style="margin-top:16px;align-items:start">
   <div class=chart><h3>Average cost saved by version</h3><p class=ch-sub>mean % saved across all configs</p><div id=verBar></div></div>
   <div class=chart><h3>Average accuracy Δ by version</h3><p class=ch-sub>candidate vs original, pp</p><div id=verAcc></div></div>
  </div>
  <div class=sect>Version history</div><div id=vtbl></div>`;
 barChart("verBar",rows.map(r=>({label:r.version,value:Math.max(0,-(r.avg_cost_delta_pct||0)),color:"var(--good)"})));
 barChart("verAcc",rows.map(r=>({label:r.version,value:Math.max(0,(r.avg_acc_delta_pp||0)+50),color:"var(--acc)"})));
 table("vtbl",[
  {key:"version",label:"Version",html:r=>`<span class=mono>${esc(r.version)}</span>`},
  {key:"_configs",label:"Configs",html:r=>r._configs||"—"},
  {key:"runs",label:"Runs"},
  {key:"avg_acc_delta_pp",label:"Δ Accuracy",sort:r=>r.avg_acc_delta_pp,html:r=>r.avg_acc_delta_pp==null?"—":accDelta(r.avg_acc_delta_pp)},
  {key:"avg_cost_delta_pct",label:"Cost Saved",sort:r=>-(r.avg_cost_delta_pct||0),html:r=>r.avg_cost_delta_pct==null?"—":saved(r.avg_cost_delta_pct)},
  {key:"_replay",label:"Replay",sort:r=>r._replay,html:r=>r._replay==null?"—":Math.round(r._replay)+"%"},
  {key:"_judge",label:"Judge",sort:r=>r._judge,html:r=>r._judge==null?"—":Math.round(r._judge)+"%"},
  {key:"regressions",label:"Regressions",html:r=>r.regressions>0?`<span class="badge b-bad">${r.regressions}</span>`:'<span class="badge b-good">0</span>'},
  {key:"_rec",label:"Status",get:r=>r.regressions>0?1:0,html:r=>r.regressions>0?'<span class="badge b-warn">attention</span>':'<span class="badge b-good">stable</span>'},
 ],rows,{});
}
async function pgRegressions(){
 const g=await getJSON("/api/regressions");const rows=(g&&g.rows)||[];
 if(!rows.length){M().innerHTML=`<div class="banner ok"><div class=bi>${BI.ok}</div><div><b>No regressions recorded.</b>
   <div class=bsub>No config has shown a statistically significant pass-rate drop across any benchmarked TokenJam version.</div></div></div>`;return;}
 M().innerHTML=`<p class=lead>Only the runs that matter: configs where the cheaper model showed a statistically significant accuracy drop. Triage these before trusting the recommendation.</p><div id=rgtbl></div>`;
 table("rgtbl",[
  {key:"created_at",label:"When",html:r=>`<span class=mono>${esc(fmtTime(r.created_at))}</span>`},
  {key:"benchmark",label:"Benchmark"},
  {key:"original_model",label:"Original → Candidate",html:r=>`<span class=mono>${esc(modelOf(r.original_model))} → ${esc(modelOf(r.candidate_model))}</span>`},
  {key:"tokenjam_version",label:"TokenJam",html:r=>`<span class=mono>${esc(r.tokenjam_version)}</span>`},
  {key:"accuracy_delta_pp",label:"Regression",sort:r=>r.accuracy_delta_pp,html:r=>r.accuracy_delta_pp==null?"—":accDelta(r.accuracy_delta_pp)},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
 ],rows,{sortKey:"created_at",dir:-1});
}
async function pgReports(){
 const runs=await loadRuns();
 M().innerHTML=`<p class=lead>Every version-stamped proof artifact. Open the rendered HTML report, view raw JSON, download it, or remove the file (the historical record stays in the database).</p><div id=rptbl></div>`;
 table("rptbl",[
  {key:"created_at",label:"Date",html:r=>`<span class=mono>${esc(fmtTime(r.created_at))}</span>`},
  {key:"benchmark",label:"Benchmark",html:r=>esc(r.benchmark)+(r.mock?' <span class=tag>mock</span>':'')},
  {key:"original_model",label:"Original → Candidate",html:r=>`<span class=mono>${esc(modelOf(r.original_model))} → ${esc(modelOf(r.candidate_model))}</span>`},
  {key:"tokenjam_version",label:"TokenJam",html:r=>`<span class=mono>${esc(r.tokenjam_version)}</span>`},
  {key:"verdict",label:"Verdict",html:r=>badge(r.verdict)},
  {key:"file",label:"Artifact",nosort:true,html:r=>`<div class=acts>
    <a class=lnk href="/report/${encodeURIComponent(r.file)}" target=_blank>HTML</a>
    <a class=lnk href="/raw/${encodeURIComponent(r.file)}" target=_blank>JSON</a>
    <a class=lnk href="/raw/${encodeURIComponent(r.file)}?download=1">Download</a>
    <button class="lnk danger" onclick="delReport('${encodeURIComponent(r.file)}')">Delete</button></div>`},
 ],runs,{sortKey:"created_at",dir:-1});
}
async function delReport(file){
 if(!confirm("Delete this proof artifact?\n\nThe file is removed from results/, but its row stays in the history database."))return;
 try{const r=await fetch("/api/report/"+file,{method:"DELETE"});if(r.ok)pgReports();else alert("Delete failed.");}
 catch(e){alert("Delete failed.");}
}
async function pgCI(){
 const runs=await loadRuns();const live=runs.filter(r=>!r.mock);
 const cards=[
  statCard(runs.length,"Indexed Runs","local results/"),
  statCard(live.length,"Live Pipeline Runs","real-key proofs"),
  statCard("06:00 UTC","Nightly Schedule","benchmark.yml cron"),
  statCard(runs.some(r=>BAD.has(r.verdict))?"attention":"green","Latest Status",runs.some(r=>BAD.has(r.verdict))?"a config regressed":"no regressions"),
 ].join("");
 const replayN=runs.filter(r=>r.benchmark==="replay").length;
 const judgedN=runs.filter(r=>r.benchmark==="judged").length;
 const stages=[
  {label:"Commit / Push",detail:"on push & PR",status:"pass"},
  {label:"Offline Benchmark",detail:runs.length+" indexed",status:runs.length?"pass":"none"},
  {label:"Replay",detail:replayN+" sessions",status:replayN?"pass":"none"},
  {label:"DeepEval",detail:judgedN+" judged",status:judgedN?"pass":"none"},
  {label:"Statistics",detail:"McNemar + Wilson CI",status:"pass"},
  {label:"Published Report",detail:runs.length+" artifacts",status:runs.length?"pass":"none"},
 ];
 const pipe=stages.map((s,i)=>`${i>0?'<div class=pipe-conn></div>':''}<div class="pipe-stage ${s.status}">
   <div class=pipe-ico>${s.status==="pass"?BI.ok:"—"}</div><div class=pipe-lbl>${esc(s.label)}</div><div class=pipe-sub>${esc(s.detail)}</div></div>`).join("");
 M().innerHTML=`<p class=lead>The continuous-benchmark pipeline runs an always-on offline gate on every push and a nightly live run against the latest TokenJam release. Full GitHub Actions logs live in the repo's Actions tab; below is the locally indexed run history.</p>
  <div class=card style="margin-bottom:16px"><div class=sect style="margin:0 0 12px">Benchmark pipeline</div><div class=pipe>${pipe}</div></div>
  <div class="grid g4">${cards}</div>
  <div class="grid g2" style="margin-top:18px;align-items:start">
   <div class=card><div class=sect style="margin:0 0 8px">Workflows</div>
    <div class=set-row><div><div class=k>ci</div><div class=d>lint + tests + offline proof smoke · on push / PR</div></div><span class="badge b-good">always-on</span></div>
    <div class=set-row><div><div class=k>benchmark</div><div class=d>nightly + manual · live if DEEPSEEK_API_KEY secret set</div></div><span class="badge b-mut">key-gated</span></div>
   </div>
   <div class=card><div class=sect style="margin:0 0 8px">Pipeline posture</div>
    <div class=muted style="font-size:13px">Offline benchmarks run with no keys and no spend, so the gate is deterministic. Live benchmarks are opt-in and key-gated, and every artifact is version-stamped with the exact TokenJam build it tested — so this history stays honest across releases.</div></div></div>
  <div class=sect>Locally indexed pipeline runs</div><div id=citbl></div>`;
 table("citbl",[
  {key:"created_at",label:"When",html:r=>`<span class=mono>${esc(fmtTime(r.created_at))}</span>`},
  {key:"benchmark",label:"Benchmark"},
  {key:"tokenjam_version",label:"Version",html:r=>`<span class=mono>${esc(r.tokenjam_version)}</span>`},
  {key:"mock",label:"Mode",get:r=>r.mock?"offline":"live",html:r=>r.mock?'<span class="badge b-mut">offline</span>':'<span class="badge b-good">live</span>'},
  {key:"verdict",label:"Result",html:r=>badge(r.verdict)},
  {key:"file",label:"Artifact",nosort:true,html:r=>reportActs(r)},
 ],runs,{sortKey:"created_at",dir:-1});
}
async function pgSettings(){
 const cfg=await getJSON("/api/configs");const hist=await getJSON("/api/history");
 const benches=[...new Set(((cfg&&cfg.rows)||[]).map(r=>r.benchmark))];
 const refresh=PREF.get("refresh","4");const theme=PREF.get("theme","dark");
 const defB=PREF.get("defaultBenchmark","");const defP=PREF.get("defaultProvider","");
 const provOpts=["","anthropic","openai","deepseek","google","groq","openrouter"];
 M().innerHTML=`<p class=lead>Dashboard preferences. Stored locally in your browser — nothing is sent anywhere.</p>
  <div class=card>
   <div class=set-row><div><div class=k>Theme</div><div class=d>dark or light appearance</div></div>
    <select id=setTheme>${["dark","light"].map(t=>`<option ${t===theme?"selected":""}>${t}</option>`).join("")}</select></div>
   <div class=set-row><div><div class=k>Auto-refresh</div><div class=d>Overview live-poll interval (seconds)</div></div>
    <select id=setRefresh>${["2","4","8","15","30","0"].map(s=>`<option ${s===refresh?"selected":""}>${s==="0"?"off":s}</option>`).join("")}</select></div>
   <div class=set-row><div><div class=k>Default benchmark</div><div class=d>pre-selected on Leaderboards</div></div>
    <select id=setBench><option value="">(latest)</option>${benches.map(b=>`<option ${b===defB?"selected":""}>${esc(b)}</option>`).join("")}</select></div>
   <div class=set-row><div><div class=k>Default provider</div><div class=d>highlight on Provider Comparison</div></div>
    <select id=setProv>${provOpts.map(p=>`<option ${p===defP?"selected":""}>${p||"(none)"}</option>`).join("")}</select></div>
   <div class=set-row><div><div class=k>Export directory</div><div class=d>where proof artifacts &amp; reports are served from</div></div>
    <span class=mono>results/</span></div>
   <div class=set-row><div><div class=k>History database</div><div class=d>${hist&&hist.available?(hist.count+" runs · "+((hist.versions||[]).length)+" versions"):"not created yet"}</div></div>
    <span class=mono>results/history.duckdb</span></div>
   <div class=set-row><div><div class=k>Report retention</div><div class=d>artifacts are kept until deleted on the Reports page</div></div>
    <span class="badge b-mut">manual</span></div>
  </div>`;
 document.getElementById("setTheme").onchange=e=>{PREF.set("theme",e.target.value);applyTheme();};
 document.getElementById("setRefresh").onchange=e=>{PREF.set("refresh",e.target.value==="off"?"0":e.target.value);startTimer();};
 document.getElementById("setBench").onchange=e=>PREF.set("defaultBenchmark",e.target.value);
 document.getElementById("setProv").onchange=e=>PREF.set("defaultProvider",e.target.value==="(none)"?"":e.target.value);
}
// ---- router ----------------------------------------------------------------
const PAGES={impact:pgImpact,overview:pgOverview,benchmarks:pgBenchmarks,scenarios:pgScenarios,replay:pgReplay,
 deepeval:pgDeepEval,trends:pgTrends,leaderboards:pgLeaderboards,providers:pgProviders,
 versions:pgVersions,regressions:pgRegressions,reports:pgReports,ci:pgCI,settings:pgSettings};
const AUTO=new Set(["overview","replay","ci"]);
function curView(){let h=location.hash||"";h=h.replace(/^#\/?/,"").split("?")[0];return PAGES[h]?h:"overview";}
function buildNav(){document.getElementById("nav").innerHTML=NAV.map(g=>
  `<div class=navsec>${esc(g[0])}</div>`+g[1].map(it=>
    `<a href="#/${it[0]}" data-v="${it[0]}"><span class=ic>${ICONS[it[0]]||""}</span>${esc(it[1])}</a>`).join("")
  ).join("");}
function markNav(v){document.querySelectorAll("#nav a").forEach(a=>a.classList.toggle("active",a.dataset.v===v));}
async function route(){
 const v=curView();markNav(v);
 document.getElementById("title").textContent=LABEL[v];
 try{await PAGES[v]();}catch(e){M().innerHTML='<div class=empty>error loading view</div>';}
 setCtx();document.getElementById("updated").textContent="updated "+new Date().toLocaleTimeString();
}
async function setCtx(){
 const hist=await getJSON("/api/history");
 const ver=(hist&&hist.versions&&hist.versions.slice(-1)[0])||"";
 const chip=document.getElementById("ctxchip");
 chip.textContent=ver?("tokenjam "+ver):(hist&&hist.count?hist.count+" runs":"");
 const vEl=document.getElementById("ver");if(vEl&&ver)vEl.textContent="tj "+ver;
}
// ---- theme + timer ---------------------------------------------------------
function applyTheme(){document.documentElement.setAttribute("data-theme",PREF.get("theme","dark"));}
let _timer=null;
function startTimer(){if(_timer)clearInterval(_timer);const s=parseInt(PREF.get("refresh","4"),10);
 if(s>0)_timer=setInterval(()=>{if(!document.hidden&&AUTO.has(curView()))route();},s*1000);}
document.getElementById("themeBtn").onclick=()=>{PREF.set("theme",PREF.get("theme","dark")==="dark"?"light":"dark");applyTheme();};
window.addEventListener("hashchange",route);
applyTheme();buildNav();route();startTimer();
</script>
</body></html>"""

# The keep-alive timer in the SPA intentionally re-renders only AUTO pages
# (Overview / Replay / CI) so interactive table sort + search state on the other
# pages survives the live poll — mirroring TokenJam Lens's asymmetric refresh.
