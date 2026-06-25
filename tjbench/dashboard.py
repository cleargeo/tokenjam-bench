"""Live proof dashboard — the bench's answer to TokenJam Lens.

`tjbench serve` starts a local, offline, auto-refreshing web dashboard over the
version-stamped proof artifacts in `results/`. It lists every run (accuracy,
cost delta, McNemar verdict, TokenJam version), surfaces the cross-version
regression matrix, and renders each run's full HTML report on demand.

Phase 4 adds analytics views backed by the historical DB (read-only):
leaderboards, provider matrix, version history, regression timeline, and trends.

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


def scan_runs(directory: str | Path) -> list[dict[str, Any]]:
    """Summarize every proof artifact in `directory`, newest first."""
    runs: list[dict[str, Any]] = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "tokenjam_version" not in d or "benchmark" not in d:
            continue
        s = d.get("stats", {}) or {}
        runs.append({
            "file": p.name,
            "benchmark": d.get("benchmark", "?"),
            "original_model": d.get("original_model", "?"),
            "candidate_model": d.get("candidate_model", "?"),
            "tokenjam_version": d.get("tokenjam_version", "?"),
            "n_tasks": d.get("n_tasks", 0),
            "original_pass_rate": round(d.get("original_pass_rate", 0.0) * 100, 1),
            "candidate_pass_rate": round(d.get("candidate_pass_rate", 0.0) * 100, 1),
            "accuracy_delta_pp": d.get("accuracy_delta_pp", 0.0),
            "cost_delta_pct": d.get("cost_delta_pct", 0.0),
            "verdict": s.get("verdict", "?"),
            "mock": d.get("mock", False),
            "created_at": d.get("created_at", 0.0),
        })
    runs.sort(key=lambda r: r["created_at"], reverse=True)
    return runs


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

        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(_DASHBOARD_HTML.encode(), "text/html; charset=utf-8")
            elif path == "/api/runs":
                self._send(json.dumps(scan_runs(root)).encode(), "application/json")
            elif path == "/api/matrix":
                payload = series_to_dict(build_series([
                    json.loads((root / r["file"]).read_text()) for r in scan_runs(root)
                ]))
                self._send(json.dumps(payload).encode(), "application/json")
            elif path == "/api/history":
                self._send(json.dumps(history_summary(root / "history.duckdb")).encode(),
                           "application/json")
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


_DASHBOARD_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>tokenjam-bench · analytics</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#c9d1d9;--mut:#8b949e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:24px 20px}
h1{font-size:20px;margin:0} .sub{color:var(--mut);margin:4px 0 14px}
.live{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2ea043;margin-right:6px;animation:pulse 1.6s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.nav{display:flex;gap:4px;flex-wrap:wrap;margin:0 0 18px;border-bottom:1px solid var(--line);padding-bottom:8px}
.navlink{color:var(--mut);text-decoration:none;padding:6px 12px;border-radius:6px;font-size:13px}
.navlink:hover{color:var(--fg);background:#1c2230}
.navlink.active{color:#58a6ff;background:#1f6feb22}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:130px}
.tile .v{font-size:20px;font-weight:700} .tile .l{color:var(--mut);font-size:12px}
.banner{border-radius:8px;padding:10px 14px;margin-bottom:16px;border-left:4px solid #2ea043;background:var(--panel)}
.banner.bad{border-left-color:#d29922}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
tr:hover td{background:#1c2230}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.v-green{color:#3fb950} .v-yellow{color:#d29922} .v-grey{color:#8b949e}
.neg{color:#3fb950} .pos{color:#f85149}
a.btn{color:#58a6ff;text-decoration:none;border:1px solid #1f6feb55;border-radius:6px;padding:2px 8px;font-size:12px}
a.btn:hover{background:#1f6feb22}
.tag{font-size:11px;color:var(--mut);border:1px solid var(--line);border-radius:10px;padding:1px 7px}
.empty{color:var(--mut);padding:24px;text-align:center}
.chart{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin-bottom:16px}
.legend{font-size:12px;color:var(--mut);margin-top:4px}
.lg-acc{color:#58a6ff}.lg-cost{color:#3fb950}
.ttl{font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:.03em;margin:0 0 10px}
select{background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:13px}
</style></head><body><div class=wrap>
<h1>tokenjam-bench &middot; analytics</h1>
<p class=sub><span class=live></span>live &middot; auto-refresh 4s &middot; <span id=updated></span></p>
<div class=nav id=nav></div>
<div id=main><div class=empty>loading…</div></div>
<script>
const V={no_significant_regression:'v-green',quality_signals_improved:'v-green',
  regression_suspected:'v-yellow',significant_regression:'v-yellow',insufficient_evidence:'v-grey'};
const VIEWS=[['overview','Overview'],['leaderboards','Leaderboards'],['providers','Providers'],
  ['versions','Versions'],['regressions','Regressions'],['trends','Trends'],['reports','Reports']];
let selBench=null, selConfig=null;
const M=()=>document.getElementById('main');
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function fmtTime(ts){if(!ts)return'';return new Date(ts*1000).toLocaleString();}
async function getJSON(u){try{return await (await fetch(u)).json();}catch(e){return null;}}
function tile(v,l){return `<div class=tile><div class=v>${esc(v)}</div><div class=l>${esc(l)}</div></div>`;}
function vtag(v){return `<span class=${V[v]||'v-grey'}>${esc(String(v).replace(/_/g,' '))}</span>`;}
function dcost(x){if(x==null)return'—';const c=x<0?'neg':(x>0?'pos':'');return `<span class=${c}>${x>0?'+':''}${x}%</span>`;}
function pct(x){return x==null?'—':(Math.round(x*10)/10)+'%';}
function tbl(headers,rows){
  if(!rows.length) return '<div class=empty>no data yet</div>';
  return `<table><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>`+
    rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')+'</tbody></table>';
}
function drawChart(id,pts){
  const box=document.getElementById(id); if(!box)return;
  if(!pts.length){box.innerHTML='<div class=empty>no data</div>';return;}
  const W=1040,H=150,padL=30,padR=10,padT=10,padB=14,n=pts.length;
  const iw=W-padL-padR, ih=H-padT-padB;
  const X=i=> n<=1? padL+iw/2 : padL+(i/(n-1))*iw;
  const Y=v=> padT+(1-Math.max(0,Math.min(100,v))/100)*ih;
  const series=(g,col)=>{
    const pl=pts.map((p,i)=>`${X(i).toFixed(1)},${Y(g(p)).toFixed(1)}`).join(' ');
    const dots=pts.map((p,i)=>`<circle cx=${X(i).toFixed(1)} cy=${Y(g(p)).toFixed(1)} r=3 fill=${col}/>`).join('');
    return `<polyline points="${pl}" fill=none stroke=${col} stroke-width=2/>${dots}`;};
  let grid='';[0,50,100].forEach(v=>{const y=Y(v).toFixed(1);
    grid+=`<line x1=${padL} y1=${y} x2=${W-padR} y2=${y} stroke=#30363d stroke-width=1/>`+
          `<text x=2 y=${(+y+3).toFixed(1)} fill=#8b949e font-size=10>${v}</text>`;});
  box.innerHTML=`<svg width=100% viewBox="0 0 ${W} ${H}" preserveAspectRatio=none `+
    `style="max-width:100%;height:150px">${grid}${series(p=>p.c,'#3fb950')}${series(p=>p.a,'#58a6ff')}</svg>`;
}

async function renderOverview(){
  const runs=(await getJSON('/api/runs'))||[];
  const mtx=(await getJSON('/api/matrix'))||{regressions_found:0};
  const hist=(await getJSON('/api/history'))||{count:0,versions:[]};
  const real=runs.filter(r=>!r.mock).length, latest=runs[0];
  const tiles=tile(runs.length,'proof runs')+tile(real,'live (non-mock)')+
    tile(mtx.regressions_found||0,'cross-version regressions')+tile(hist.count||0,'runs in history')+
    tile((hist.versions||[]).length,'tokenjam versions')+
    (latest?tile((latest.cost_delta_pct>0?'+':'')+latest.cost_delta_pct+'%','latest Δ cost'):'');
  const banner=mtx.regressions_found>0
    ? `<div class="banner bad">&#9888; ${mtx.regressions_found} cross-version regression(s) detected.</div>`
    : (runs.length?`<div class=banner>&#10003; no cross-version regressions.</div>`:'');
  const rows=runs.map(r=>[`<span class=mono>${esc(fmtTime(r.created_at))}</span>`,
    esc(r.benchmark)+(r.mock?' <span class=tag>mock</span>':''),
    `<span class=mono>${esc(r.original_model)} → ${esc(r.candidate_model)}</span>`,
    `<span class=mono>${esc(r.tokenjam_version)}</span>`, r.n_tasks,
    `${r.original_pass_rate}% → <b>${r.candidate_pass_rate}%</b>`, dcost(r.cost_delta_pct),
    vtag(r.verdict), `<a class=btn href="/report/${encodeURIComponent(r.file)}" target=_blank>report</a>`]);
  M().innerHTML=`<div class=tiles>${tiles}</div>${banner}
    <div class=chart><div class=ttl>Trend · accuracy & cost saved over time</div><div id=chartbox></div>
    <div class=legend><span class=lg-acc>&#9632;</span> candidate accuracy &nbsp;&nbsp;
    <span class=lg-cost>&#9632;</span> cost saved</div></div>
    ${tbl(['when','benchmark','original → candidate','tokenjam','n','accuracy','Δ cost','verdict',''],rows)}`;
  drawChart('chartbox', runs.slice().reverse().map(r=>({a:r.candidate_pass_rate,c:Math.max(0,-r.cost_delta_pct)})));
}

async function renderLeaderboards(){
  const cfg=(await getJSON('/api/configs'))||{rows:[]};
  const benches=[...new Set((cfg.rows||[]).map(r=>r.benchmark))];
  if(!benches.length){M().innerHTML='<div class=empty>No history yet. Run some proofs.</div>';return;}
  if(!selBench||!benches.includes(selBench)) selBench=benches[0];
  const lb=(await getJSON('/api/leaderboard?benchmark='+encodeURIComponent(selBench)))||{rows:[]};
  const sel=`<select onchange="selBench=this.value;renderLeaderboards()">`+
    benches.map(b=>`<option ${b===selBench?'selected':''}>${esc(b)}</option>`).join('')+`</select>`;
  const rows=(lb.rows||[]).map((r,i)=>[`#${i+1}`,`<span class=mono>${esc(r.model)}</span>`,
    pct((r.pass_rate||0)*100), r.cost_usd==null?'—':'$'+(+r.cost_usd).toFixed(6),
    `<span class=mono>${esc(r.tokenjam_version)}</span>`]);
  M().innerHTML=`<div class=ttl>Leaderboard &nbsp; ${sel}</div>${tbl(['rank','model','pass-rate','cost','tokenjam'],rows)}`;
}

async function renderProviders(){
  const p=(await getJSON('/api/providers'))||{rows:[]};
  const rows=(p.rows||[]).map(r=>[`<span class=mono>${esc(r.model)}</span>`, r.runs, r.benchmarks,
    pct((r.avg_accuracy||0)*100), r.avg_cost_usd==null?'—':'$'+(+r.avg_cost_usd).toFixed(6)]);
  M().innerHTML=`<div class=ttl>Provider / model matrix</div>${tbl(['model','runs','benchmarks','avg accuracy','avg cost'],rows)}`;
}

async function renderVersions(){
  const v=(await getJSON('/api/version-summary'))||{rows:[]};
  const rows=(v.rows||[]).map(r=>[`<span class=mono>${esc(r.version)}</span>`, r.runs,
    r.avg_acc_delta_pp==null?'—':r.avg_acc_delta_pp.toFixed(1)+'pp',
    r.avg_cost_delta_pct==null?'—':r.avg_cost_delta_pct.toFixed(1)+'%',
    r.regressions>0?`<span class=v-yellow>${r.regressions}</span>`:'0']);
  M().innerHTML=`<div class=ttl>TokenJam version history</div>${tbl(['tokenjam','runs','avg Δ acc','avg Δ cost','regressions'],rows)}`;
}

async function renderRegressions(){
  const g=(await getJSON('/api/regressions'))||{rows:[]};
  if(!(g.rows||[]).length){M().innerHTML='<div class=banner>&#10003; no regressions recorded.</div>';return;}
  const rows=g.rows.map(r=>[`<span class=mono>${esc(fmtTime(r.created_at))}</span>`, esc(r.benchmark),
    `<span class=mono>${esc(r.original_model)} → ${esc(r.candidate_model)}</span>`,
    `<span class=mono>${esc(r.tokenjam_version)}</span>`,
    r.accuracy_delta_pp==null?'—':r.accuracy_delta_pp.toFixed(1)+'pp', vtag(r.verdict)]);
  M().innerHTML=`<div class=ttl>Regression timeline</div>${tbl(['when','benchmark','models','tokenjam','Δ acc','verdict'],rows)}`;
}

async function renderTrends(){
  const cfg=(await getJSON('/api/configs'))||{rows:[]};
  const cfgs=cfg.rows||[];
  if(!cfgs.length){M().innerHTML='<div class=empty>No history yet.</div>';return;}
  const key=c=>c.benchmark+'|'+c.original_model+'|'+c.candidate_model;
  if(!selConfig||!cfgs.some(c=>key(c)===selConfig)) selConfig=key(cfgs[0]);
  const parts=selConfig.split('|'), bm=parts[0], orig=parts[1], cand=parts[2];
  const tr=(await getJSON(`/api/trend?benchmark=${encodeURIComponent(bm)}&original=${encodeURIComponent(orig)}&candidate=${encodeURIComponent(cand)}`))||{rows:[]};
  const sel=`<select onchange="selConfig=this.value;renderTrends()">`+
    cfgs.map(c=>`<option value="${esc(key(c))}" ${key(c)===selConfig?'selected':''}>${esc(c.benchmark)}: ${esc(c.original_model)}→${esc(c.candidate_model)}</option>`).join('')+`</select>`;
  const rows=(tr.rows||[]).map(r=>[`<span class=mono>${esc(r.tokenjam_version)}</span>`,
    pct((r.candidate_pass_rate||0)*100),
    r.accuracy_delta_pp==null?'—':r.accuracy_delta_pp.toFixed(1)+'pp',
    r.cost_delta_pct==null?'—':r.cost_delta_pct.toFixed(1)+'%',
    r.deepeval_score==null?'—':(+r.deepeval_score).toFixed(2), vtag(r.verdict)]);
  M().innerHTML=`<div class=ttl>Trend &nbsp; ${sel}</div>
    <div class=chart><div id=chartbox></div><div class=legend><span class=lg-acc>&#9632;</span> candidate accuracy
    &nbsp;&nbsp; <span class=lg-cost>&#9632;</span> cost saved</div></div>
    ${tbl(['tokenjam','cand pass-rate','Δ acc','Δ cost','deepeval','verdict'],rows)}`;
  drawChart('chartbox', (tr.rows||[]).map(r=>({a:(r.candidate_pass_rate||0)*100,c:Math.max(0,-(r.cost_delta_pct||0))})));
}

async function renderReports(){
  const runs=(await getJSON('/api/runs'))||[];
  const rows=runs.map(r=>[`<span class=mono>${esc(fmtTime(r.created_at))}</span>`, esc(r.benchmark),
    `<span class=mono>${esc(r.tokenjam_version)}</span>`, vtag(r.verdict),
    `<a class=btn href="/report/${encodeURIComponent(r.file)}" target=_blank>open report</a>`]);
  M().innerHTML=`<div class=ttl>Reports</div>${tbl(['when','benchmark','tokenjam','verdict','report'],rows)}`;
}

const RENDER={overview:renderOverview,leaderboards:renderLeaderboards,providers:renderProviders,
  versions:renderVersions,regressions:renderRegressions,trends:renderTrends,reports:renderReports};
function currentView(){
  let h=location.hash||''; if(h.indexOf('#/')===0)h=h.slice(2); else if(h.indexOf('#')===0)h=h.slice(1);
  h=h.split('?')[0]; return RENDER[h]?h:'overview';
}
function renderNav(v){
  document.getElementById('nav').innerHTML=VIEWS.map(p=>
    `<a href="#/${p[0]}" class="navlink ${p[0]===v?'active':''}">${p[1]}</a>`).join('');
}
async function route(){
  const v=currentView(); renderNav(v);
  document.getElementById('updated').textContent='updated '+new Date().toLocaleTimeString();
  try{ await RENDER[v](); }catch(e){ M().innerHTML='<div class=empty>error loading view</div>'; }
}
window.addEventListener('hashchange',route);
route(); setInterval(route,4000);
</script>
</div></body></html>"""
