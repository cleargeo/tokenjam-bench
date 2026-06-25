"""HTML report generator — self-contained, offline, faithful to the proof dict."""
from __future__ import annotations

from tjbench.pipeline import run_proof
from tjbench.report_html import render_html_from_dict, write_html_report


def _dict():
    return run_proof(
        benchmark_name="samples", original_spec="anthropic:claude-opus-4-7",
        mock=True, mock_candidate_accuracy=0.6,
    ).to_dict()


def test_html_contains_the_key_proof_facts():
    html = render_html_from_dict(_dict())
    assert "<!doctype html>" in html.lower()
    assert "anthropic:claude-opus-4-7" in html       # original
    assert "anthropic:claude-haiku-4-5" in html       # TokenJam's candidate
    assert "McNemar" in html
    assert "tokenjam 0.5.1" in html or "tokenjam 0." in html
    assert "pass-rate delta" in html


def test_html_is_offline_no_external_http():
    html = render_html_from_dict(_dict())
    # No render-time external resource loads (offline-first, like TokenJam Lens).
    for bad in ("http://", "https://", "src=", "cdn"):
        assert bad not in html


def test_html_surfaces_honest_caveats_for_mock():
    html = render_html_from_dict(_dict())
    assert "MOCK" in html
    assert "not a general" in html  # the "not a general 'quality preserved' claim" note


def test_write_html_report_creates_file(tmp_path):
    path = write_html_report(_dict(), tmp_path)
    assert path.exists() and path.suffix == ".html"
    assert path.read_text(encoding="utf-8").lstrip().lower().startswith("<!doctype html>")
