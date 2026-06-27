"""CI honesty guard (Brief C).

This repo is honesty-branded: its credibility *is* its evidence. These tests
fail CI on dishonest output so the brand discipline cannot silently regress.

  Test A — no committed artifact whose numbers are surfaced as a headline /
           dashboard figure may be priced with TokenJam's $0.50/$2.00 placeholder
           rates (`priced_with_defaults=true`).
  Test B — no banned marketing/overclaim string may appear (asserted) in the
           README, docs/, or committed artifacts. The project's *negated* honest
           idiom (e.g. "never a 'quality preserved' claim") is explicitly allowed.

Self-contained: there is no external rule package to import. The banned-string
literals live only in this file, and Test B deliberately does NOT scan tests/,
so the guard never trips on its own pattern definitions.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Test A — placeholder-priced headlines
# --------------------------------------------------------------------------- #

# Directories searched for committed proof artifacts.
ARTIFACT_ROOTS = ("docs/evidence", "results")

# Legacy pre-multipair DeepSeek runs (placeholder-priced); to be archived out of
# the headline path in the furniture pass — remove this allowlist entry once they
# move to docs/evidence/archive/.  (DeepSeek has no TokenJam rate, so these were
# priced with the $0.50/$2.00 default; they predate the honestly-priced
# 2026-06-26-multipair set and must not block the guard until Brief D relocates
# them.  Matched as EXACT parent directories so the live multipair set — which
# lives under docs/evidence/live/ too — is still fully checked.)
PRICED_WITH_DEFAULTS_ALLOWLIST = {
    "docs/evidence/live",
    "docs/evidence/live/2026-06-26-real-dashboard",
}


def _proof_artifacts() -> list[Path]:
    files: list[Path] = []
    for root in ARTIFACT_ROOTS:
        files.extend((REPO / root).rglob("*.json"))
    return sorted(files)


def _is_dashboard_surfaced(d: dict) -> bool:
    """An artifact's numbers reach a headline/dashboard figure iff the production
    dashboard would render it. That is exactly `scan_runs()`'s production filter
    (tjbench/dashboard.py): a stamped proof artifact that is neither a `--mock`
    (dev) run nor a `demo` (seeded-fixture) run."""
    if "tokenjam_version" not in d or "benchmark" not in d:
        return False  # not a proof artifact (e.g. a config/index json)
    return not d.get("mock") and not d.get("demo")


def test_no_placeholder_priced_headlines() -> None:
    offenders: list[str] = []
    for path in _proof_artifacts():
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if not isinstance(d, dict) or not _is_dashboard_surfaced(d):
            continue
        if d.get("priced_with_defaults") is not True:
            continue
        parent = path.parent.relative_to(REPO).as_posix()
        if parent in PRICED_WITH_DEFAULTS_ALLOWLIST:
            continue
        offenders.append(path.relative_to(REPO).as_posix())

    assert not offenders, (
        "Dashboard-surfaced artifacts priced with TokenJam's $0.50/$2.00 "
        "placeholder rates (priced_with_defaults=true) — a placeholder cost must "
        "never headline. Re-run with real rates, or (legacy only) archive them "
        "out of the headline path:\n  " + "\n  ".join(offenders)
    )


def test_dashboard_surfaced_predicate() -> None:
    """Test A only guards artifacts the dashboard would surface. A real
    placeholder-priced run is caught; mock/demo/non-artifact jsons are not."""
    real = {"tokenjam_version": "0.5.2", "benchmark": "gsm8k",
            "priced_with_defaults": True}
    assert _is_dashboard_surfaced(real)  # a placeholder-priced headline → guarded
    assert not _is_dashboard_surfaced({**real, "mock": True})   # dev run
    assert not _is_dashboard_surfaced({**real, "demo": True})   # seeded fixture
    assert not _is_dashboard_surfaced({"some": "config"})       # not a proof artifact


# --------------------------------------------------------------------------- #
# Test B — banned overclaim / extrapolation strings
# --------------------------------------------------------------------------- #

# Each pattern is a CLAIM that is dishonest only when ASSERTED. The repo's
# established honest idiom states the negation ("never a 'quality preserved'
# claim", "no single confidence = 95% scalar"), which must keep passing — see
# `_NEGATION` below. Allowed honest forms (CI, p-value, and the three hedged
# verdicts no_significant_regression / significant_regression /
# insufficient_evidence) match none of these patterns.
_BANNED = {
    "quality-preserved claim": r"quality\s+preserved",
    "safe-to-replace claim": r"safe\s+to\s+replace",
    "you-can-replace directive": r"you\s+can\s+replace",
    "confidence-% scalar": r"confidence\s*[=:]\s*\d+\s*%",
    "percent-confident scalar": r"\d+\s*%\s*confiden(?:t|ce)",
    "ROI extrapolation (10x)": r"\bat\s+10x\b",
    "ROI extrapolation (100x)": r"\bat\s+100x\b",
    "ROI extrapolation (annual savings)": r"annual\s+savings",
}

# Negation cues that mark an immediately-preceding disavowal (the honest idiom).
_NEGATION = re.compile(
    r"\b(?:no|not|never|without|none|avoid|avoids|avoided|deliberately|"
    r"isn't|aren't|don't|doesn't|won't|nor)\b|n't\b",
    re.IGNORECASE,
)

# tjbench/dashboard.py is intentionally NOT scanned yet: it still contains banned
# strings until Brief A's dashboard rebuild lands. Grepping it now would red-CI
# this branch. Add it to SCAN_ROOTS in a follow-up commit AFTER Brief A merges
# (see Brief C). The one-line change is documented here on purpose.
DEFERRED_UNTIL_BRIEF_A = ("tjbench/dashboard.py",)

# What to scan now: the README + everything under docs/ and results/ (markdown,
# HTML reports, JSON artifacts). tests/ is deliberately excluded so this file's
# own pattern literals never self-trigger.
SCAN_ROOTS = ("docs", "results")
SCAN_SUFFIXES = {".md", ".html", ".htm", ".json", ".txt", ".toml", ".rst"}


def _scan_files() -> list[Path]:
    files = [REPO / "README.md"]
    for root in SCAN_ROOTS:
        for p in (REPO / root).rglob("*"):
            if p.is_file() and p.suffix.lower() in SCAN_SUFFIXES:
                files.append(p)
    return sorted(f for f in files if f.exists())


def _clause_before(text: str, start: int, window: int = 110) -> str:
    """Text from the nearest preceding clause/sentence boundary up to `start`.
    Used to decide whether a banned phrase is disavowed in its own clause."""
    head = text[max(0, start - window):start]
    # cut at the last clause boundary so a negation from a *previous* sentence
    # doesn't excuse an assertion in this one
    boundary = max(head.rfind("."), head.rfind("!"), head.rfind("?"),
                   head.rfind(";"), head.rfind("—"), head.rfind("--"),
                   head.rfind("</"), head.rfind("<li"))
    return head[boundary + 1:] if boundary != -1 else head


def _banned_hits(raw: str) -> list[tuple[str, str]]:
    """Return (label, snippet) for every ASSERTED banned phrase in `raw`.

    A phrase disavowed in its own clause ("never a 'quality preserved' claim")
    is the repo's honest idiom and is not a hit. HTML entities are unescaped
    first (the report templates wrap claims in &ldquo;…&rdquo;, whose trailing
    ';' would otherwise read as a false clause boundary); whitespace is collapsed
    so a disclaimer that wraps across lines is read as one clause.
    """
    text = re.sub(r"\s+", " ", html.unescape(raw))
    hits: list[tuple[str, str]] = []
    for label, pattern in _BANNED.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if _NEGATION.search(_clause_before(text, m.start())):
                continue  # honest, negated idiom — allowed
            snippet = text[max(0, m.start() - 35):m.end() + 20].strip()
            hits.append((label, snippet))
    return hits


def test_no_banned_strings() -> None:
    violations: list[str] = []
    for path in _scan_files():
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(REPO).as_posix()
        violations.extend(f"{rel}: [{label}] …{snip}…" for label, snip in _banned_hits(raw))

    assert not violations, (
        "Banned overclaim/extrapolation strings asserted (not disavowed) in "
        "committed docs/artifacts. Use the hedged verdicts + CI + p-value "
        "instead:\n  " + "\n  ".join(violations)
    )


# --- guard self-verification: it must CATCH a regression and ALLOW honest copy --

def test_guard_catches_asserted_violations() -> None:
    """If a banned widget/string returned, the guard must fire. Each fixture is
    an ASSERTED claim with no disavowal in its clause."""
    asserted = [
        "Downsizing this model keeps quality preserved across every workload.",
        "This candidate is safe to replace the premium model.",
        "You can replace opus with haiku for all coding tasks.",
        "We report confidence = 95% for this routing decision.",
        "The judge is 92% confident the answers match.",
        "Projected annual savings exceed the subscription cost.",
        "At 10x scale the savings compound dramatically.",
        "At 100x volume this pays for itself.",
    ]
    for claim in asserted:
        assert _banned_hits(claim), f"guard missed an asserted violation: {claim!r}"


def test_guard_allows_honest_negated_idiom() -> None:
    """The repo's honest forms must keep passing: disavowals of the banned
    claims, plus CI + p-value + the three hedged verdicts."""
    honest = [
        'Accuracy is the pass-rate on this suite, not a general "quality '
        'preserved" claim.',
        "There is deliberately no single `confidence = 95%` scalar — the honest "
        "expression is the CI + p-value.",
        "This is not a 'safe to replace' verdict; use the hedged result.",
        "Confidence is the CI + p-value, not a single 'safe %'.",
        "We do not extrapolate to annual savings or quote figures at 10x.",
        "Pass rate 48/50 [95% CI 87-99%], McNemar p=0.180 → "
        "no_significant_regression.",
        "Verdicts are significant_regression / insufficient_evidence only.",
    ]
    for line in honest:
        assert not _banned_hits(line), f"guard false-positived on honest copy: {line!r}"
