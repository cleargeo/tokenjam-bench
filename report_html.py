"""Self-contained HTML proof report.

Turns a ProofResult dict (the same dict written to the JSON artifact) into a
single, dependency-free HTML file an engineer or a manager can open and read --
no terminal, no JS frameworks, no external HTTP. Inline CSS + inline SVG bars,
mirroring TokenJam Lens's offline-first discipline.

Renders from the dict (not the dataclass) so both paths use one renderer:
`tjbench run --html` (live result.to_dict()) and `tjbench report artifact.json`
(a saved artifact).
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

# Verdict -> (accent colour, human label).
_VERDICT = {
    "no_significant_regression": ("#2ea043", "No significant regression"),
    "regression_suspected":      ("#d29922", "Regression suspected"),
    "significant_regression":    ("#f85149", "Significant regression"),
    "insufficient_evidence":     ("#8b949e", "Insufficient evidence"),
}


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _bar(label: str, value: float, vmax: float, text: str, colour: str,
         ci: list[float] | None = None) -> str:
    """One horizontal SVG bar (value scaled to vmax), with an optional CI whisker."""
    w = 360
    pct = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    bar_w = pct * w
    whisker = ""
    if ci and vmax > 0:
        x1 = max(0.0, min(1.0, ci[0] / vmax)) * w
        x2 = max(0.0, min(1.0, ci[1] / vmax)) * w
        whisker = (
            f'<line x1="{x1:.1f}" y1="11" x2="{x2:.1f}" y2="11" stroke="#c9d1d9" '
            f'stroke-width="2"/>'
            f'<line x1="{x1:.1f}" y1="6" x2="{x1:.1f}" y2="16" stroke="#c9d1d9" stroke-width="2"/>'
            f'<line x1="{x2:.1f}" y1="6" x2="{x2:.1f}" y2="16" stroke="#c9d1d9" stroke-width="2"/>'
        )
    return (
        f'<div class="barrow"><span class="barlabel">{_esc(label)}</span>'
        f'<svg width="{w}" height="22" class="bar">'
        f'<rect x="0" y="4" width="{w}" height="14" rx="3" fill="#21262d"/>'
        f'<rect x="0" y="4" width="{bar_w:.1f}" height="14" rx="3" fill="{colour}"/>'
        f'{whisker}</svg>'
        f'<span class="barval">{_esc(text)}</span></div>'
    )


def render_html_from_dict(d: dict) -> str:
    s = d.get("stats", {}) or {}
    verdict = s.get("verdict", "insufficient_evidence")
    v_colour, v_label = _VERDICT.get(verdict, ("#8b949e", verdict))
    n = d.get("n_tasks", 0)
    k = s.get("samples_per_task", 1)

    o_rate = (d.get("original_pass_rate", 0.0) or 0.0) * 100
    c_rate = (d.get("candidate_pass_rate", 0.0) or 0.0) * 100
    o_ci = s.get("original_ci_pp", [0, 0])
    c_ci = s.get("candidate_ci_pp", [0, 0])
    d_ci = s.get("delta_ci_pp", [0, 0])
    o_cost = d.get("original_cost_usd", 0.0) or 0.0
    c_cost = d.get("candidate_cost_usd", 0.0) or 0.0
    cost_max = max(o_cost, c_cost, 1e-12)

    # Pass-rate bars (0..100, with CI whiskers).
    pass_bars = (
        _bar("Original", o_rate, 100, f"{d.get('original_pass',0)}/{n} ({o_rate:.0f}%)",
             "#58a6ff", o_ci)
        + _bar("Candidate", c_rate, 100, f"{d.get('candidate_pass',0)}/{n} ({c_rate:.0f}%)",
               "#79c0ff", c_ci)
    )
    cost_bars = (
        _bar("Original", o_cost, cost_max, f"${o_cost:.6f}", "#58a6ff")
        + _bar("Candidate", c_cost, cost_max, f"${c_cost:.6f}", "#79c0ff")
    )

    # Per-task rows.
    rows = []
    for t in d.get("tasks", []):
        op, cp, sm = t.get("original_passes", 0), t.get("candidate_passes", 0), t.get("samples", 1)
        regressed = (op * 2 >= sm) and not (cp * 2 >= sm)
        flag = '<span class="reg">regressed</span>' if regressed else ""
        rows.append(
            f"<tr><td class=mono>{_esc(t.get('task_id',''))}</td>"
            f"<td>{op}/{sm}</td><td>{cp}/{sm}</td><td>{flag}</td>"
            f"<td class=mono>{_esc((t.get('candidate_detail') or '')[:90])}</td></tr>"
        )

    notes = [
        "Accuracy is the pass-rate on THIS benchmark suite, not a general "
        "&ldquo;quality preserved&rdquo; claim. Confidence is the CI + p-value, "
        "not a single &ldquo;safe %&rdquo;.",
    ]
    if verdict == "insufficient_evidence":
        notes.append(
            f"Only n={n} tasks &mdash; too few for a significance verdict. "
            "Raise the task count for a defensible result.")
    if d.get("token_inflation_flag"):
        notes.append(
            f"Candidate produced {d.get('output_token_inflation')}&times; the output "
            "tokens &mdash; measured savings already reflect this, but the per-token "
            "advantage is eroded (verbosity/retries).")
    if d.get("mock"):
        notes.append(
            "MOCK run &mdash; offline + deterministic. Numbers are illustrative, "
            "not from the real models.")
    if d.get("priced_with_defaults"):
        notes.append(
            "A model had no TokenJam rate; cost used the $0.50/$2.00 default "
            "placeholder &mdash; savings are approximate.")

    return _TEMPLATE.format(
        title=_esc(f"{d.get('benchmark','?')} proof"),
        tj=_esc(d.get("tokenjam_version", "?")),
        benchmark=_esc(d.get("benchmark", "?")),
        n=n, k=k,
        original=_esc(d.get("original_model", "?")),
        candidate=_esc(d.get("candidate_model", "?")),
        recommended_by=_esc(d.get("recommended_by", "")),
        v_colour=v_colour, v_label=_esc(v_label),
        cost_delta=f"{d.get('cost_delta_pct',0.0):+.1f}",
        acc_delta=f"{d.get('accuracy_delta_pp',0.0):+.1f}",
        d_lo=f"{d_ci[0]:+.1f}", d_hi=f"{d_ci[1]:+.1f}",
        mcnemar_p=f"{s.get('mcnemar_p_value',1.0):.3f}",
        mc_b=s.get("mcnemar_b", 0), mc_c=s.get("mcnemar_c", 0), alpha=s.get("alpha", 0.05),
        pass_bars=pass_bars, cost_bars=cost_bars,
        rows="".join(rows) or '<tr><td colspan=5 class=mono>no tasks</td></tr>',
        notes="".join(f"<li>{x}</li>" for x in notes),
        generated=_esc(time.strftime("%Y-%m-%d %H:%M", time.localtime(d.get("created_at", time.time())))),
    )


_TEMPLATE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>TokenJam proof &mdash; {title}</title>
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#c9d1d9;--mut:#8b949e}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:880px;margin:0 auto;padding:32px 20px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:14px;color:var(--mut);font-weight:600;margin:28px 0 10px;text-transform:uppercase;letter-spacing:.04em}}
.sub{{color:var(--mut);margin:0 0 20px}}
.badge{{display:inline-block;background:#1f6feb22;color:#58a6ff;border:1px solid #1f6feb55;border-radius:20px;padding:2px 10px;font-size:12px}}
.verdict{{border-left:4px solid {v_colour};background:var(--panel);border-radius:6px;padding:14px 16px;margin:0 0 22px}}
.verdict b{{color:{v_colour}}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:0 0 8px}}
.kpi{{flex:1;min-width:160px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}}
.kpi .v{{font-size:22px;font-weight:700}} .kpi .l{{color:var(--mut);font-size:12px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;margin:0 0 14px}}
.barrow{{display:flex;align-items:center;gap:10px;margin:6px 0}}
.barlabel{{width:78px;color:var(--mut)}} .barval{{font-variant-numeric:tabular-nums}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--mut);font-weight:600}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}}
.reg{{color:#f85149;font-weight:600}}
ul.notes{{color:var(--mut);font-size:12.5px;padding-left:18px}} ul.notes li{{margin:4px 0}}
.foot{{color:var(--mut);font-size:12px;margin-top:24px;border-top:1px solid var(--line);padding-top:12px}}
</style></head><body><div class=wrap>
<h1>TokenJam proof &mdash; {benchmark}</h1>
<p class=sub><span class=badge>tokenjam {tj}</span> &nbsp; n={n} tasks &middot; k={k} sample(s) &middot;
{original} &rarr; {candidate}</p>
<div class=verdict>Verdict: <b>{v_label}</b> &nbsp;&middot;&nbsp; McNemar p={mcnemar_p} (&alpha;={alpha})
&nbsp;&middot;&nbsp; candidate chosen by {recommended_by}</div>
<div class=kpis>
<div class=kpi><div class=v>{cost_delta}%</div><div class=l>cost delta (measured)</div></div>
<div class=kpi><div class=v>{acc_delta}pp</div><div class=l>pass-rate delta [95% CI {d_lo}, {d_hi}]</div></div>
<div class=kpi><div class=v>{mc_b} / {mc_c}</div><div class=l>tasks broken / fixed by the swap</div></div>
</div>
<h2>Pass rate (95% CI whiskers)</h2><div class=panel>{pass_bars}</div>
<h2>Cost (measured)</h2><div class=panel>{cost_bars}</div>
<h2>Per-task</h2><div class=panel><table>
<tr><th>task</th><th>orig</th><th>cand</th><th></th><th>candidate detail</th></tr>
{rows}</table></div>
<h2>How to read this</h2><ul class=notes>{notes}</ul>
<div class=foot>Generated {generated} &middot; tokenjam-bench &middot; executable-accuracy proof</div>
</div></body></html>"""


def write_html_report(d: dict, out_dir: str | Path) -> Path:
    """Render `d` (a ProofResult dict) to a stamped .html file; return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = int(d.get("created_at", time.time()))
    path = out / f"tjbench_{d.get('benchmark','report')}_tj{d.get('tokenjam_version','?')}_{stamp}.html"
    path.write_text(render_html_from_dict(d), encoding="utf-8")
    return path


def load_and_render(json_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """Read a saved JSON artifact and write the HTML next to it (or in out_dir)."""
    p = Path(json_path)
    d = json.loads(p.read_text())
    return write_html_report(d, out_dir or p.parent)
