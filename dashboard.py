"""Live proof dashboard — the bench's answer to TokenJam Lens.

`tjbench serve` starts a local, offline, auto-refreshing web dashboard over the
version-stamped proof artifacts in `results/`. It lists every run (accuracy,
cost delta, McNemar verdict, TokenJam version), surfaces the cross-version
regression matrix, and renders each run's full HTML report on demand.

Realtime: the page polls `/api/runs` + `/api/matrix` every few seconds, so a new
proof appears the moment its JSON lands in `results/` — no manual refresh.

Offline-first (like TokenJam Lens): one self-contained page, inline CSS/JS, no
external HTTP, stdlib `http.server` only — no new dependencies.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from matrix import build_series, series_to_dict
from report_html import render_html_from_dict


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


def serve(directory: str | Path = "results", host: str = "127.0.0.1",
          port: int = 7392) -> None:
    """Start the dashboard server (blocking until Ctrl-C)."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

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
<title>tokenjam-bench · proof dashboard</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#c9d1d9;--mut:#8b949e}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:28px 20px}
h1{font-size:20px;margin:0} .sub{color:var(--mut);margin:4px 0 18px}
.live{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2ea043;margin-right:6px;animation:pulse 1.6s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
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
</style></head><body><div class=wrap>
<h1>tokenjam-bench &middot; proof dashboard</h1>
<p class=sub><span class=live></span>live &middot; auto-refresh every 4s &middot;
<span id=updated></span></p>
<div class=tiles id=tiles></div>
<div id=banner></div>
<table><thead><tr>
<th>when</th><th>benchmark</th><th>original &rarr; candidate</th><th>tokenjam</th>
<th>n</th><th>accuracy</th><th>&Delta; cost</th><th>verdict</th><th></th>
</tr></thead><tbody id=rows><tr><td colspan=9 class=empty>loading…</td></tr></tbody></table>
<script>
const V={no_significant_regression:'v-green',quality_signals_improved:'v-green',
  regression_suspected:'v-yellow',significant_regression:'v-yellow',insufficient_evidence:'v-grey'};
function fmtTime(ts){if(!ts)return'';const d=new Date(ts*1000);return d.toLocaleString();}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function tick(){
  let runs=[],mtx={regressions_found:0};
  try{runs=await (await fetch('/api/runs')).json();}catch(e){return;}
  try{mtx=await (await fetch('/api/matrix')).json();}catch(e){}
  document.getElementById('updated').textContent='updated '+new Date().toLocaleTimeString();
  // tiles
  const real=runs.filter(r=>!r.mock).length;
  const latest=runs[0];
  document.getElementById('tiles').innerHTML=
    tile(runs.length,'proof runs')+tile(real,'live (non-mock)')+
    tile(mtx.regressions_found||0,'cross-version regressions')+
    (latest?tile((latest.cost_delta_pct>0?'+':'')+latest.cost_delta_pct+'%','latest Δ cost'):'');
  // banner
  const b=document.getElementById('banner');
  if(mtx.regressions_found>0){b.className='banner bad';
    b.innerHTML='&#9888; '+mtx.regressions_found+' cross-version regression(s) detected across TokenJam versions.';}
  else if(runs.length){b.className='banner';b.innerHTML='&#10003; no cross-version regressions.';}
  else{b.innerHTML='';}
  // rows
  const tb=document.getElementById('rows');
  if(!runs.length){tb.innerHTML='<tr><td colspan=9 class=empty>No proof artifacts yet. Run <span class=mono>tjbench run … --html</span>.</td></tr>';return;}
  tb.innerHTML=runs.map(r=>{
    const acc=`${r.original_pass_rate}% → <b>${r.candidate_pass_rate}%</b>`;
    const dc=r.cost_delta_pct, dcCls=dc<0?'neg':(dc>0?'pos':'');
    const vc=V[r.verdict]||'v-grey';
    const mock=r.mock?' <span class=tag>mock</span>':'';
    return `<tr><td class=mono>${esc(fmtTime(r.created_at))}</td>
      <td>${esc(r.benchmark)}${mock}</td>
      <td class=mono>${esc(r.original_model)} → ${esc(r.candidate_model)}</td>
      <td class=mono>${esc(r.tokenjam_version)}</td>
      <td>${r.n_tasks}</td><td>${acc}</td>
      <td class=${dcCls}>${dc>0?'+':''}${dc}%</td>
      <td class=${vc}>${esc(r.verdict).replace(/_/g,' ')}</td>
      <td><a class=btn href="/report/${encodeURIComponent(r.file)}" target=_blank>report</a></td></tr>`;
  }).join('');
}
function tile(v,l){return `<div class=tile><div class=v>${esc(v)}</div><div class=l>${esc(l)}</div></div>`;}
tick();setInterval(tick,4000);
</script>
</div></body></html>"""
