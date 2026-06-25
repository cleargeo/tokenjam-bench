"""Cross-version regression matrix.

TokenJam changes daily. Every proof artifact is already stamped with the exact
`tokenjam_version` that produced it, so detecting "did the new release move the
numbers" is a comparison across stamped artifacts — no need to install multiple
TokenJams at once.

`build_series` groups artifacts by (benchmark, original_model), orders them by
TokenJam version, and for each consecutive pair flags three regressions the
tech-lead review asked for:
  - accuracy regression  — the candidate's pass-rate dropped,
  - cost regression      — the savings shrank (cost delta got less negative),
  - recommendation change — TokenJam now downgrades to a different candidate.

The workflow: run a proof, `make update-tokenjam`, run again; artifacts pile up
in results/ stamped by version; `tjbench matrix` shows the trend and the day a
release regressed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Thresholds (percentage points). Surfaced in output so the judgment is
# inspectable rather than hidden.
ACCURACY_REGRESSION_PP = 2.0   # candidate pass-rate dropped this much vs prev version
COST_REGRESSION_PP = 5.0       # savings shrank this much (cost delta got less negative)


@dataclass
class VersionPoint:
    tokenjam_version: str
    created_at: float
    candidate_model: str
    n_tasks: int
    candidate_pass_rate: float    # 0..1
    accuracy_delta_pp: float      # candidate vs original, this version
    cost_delta_pct: float         # candidate vs original, this version (neg = cheaper)
    verdict: str
    # filled relative to the previous version in the series:
    pass_rate_change_pp: float | None = None
    cost_delta_change_pp: float | None = None
    candidate_changed: bool = False
    regressions: list[str] = field(default_factory=list)


@dataclass
class ConfigSeries:
    benchmark: str
    original_model: str
    points: list[VersionPoint]

    @property
    def regression_count(self) -> int:
        return sum(len(p.regressions) for p in self.points)


def _version_key(v: str) -> tuple:
    """Sort key for 'X.Y.Z' style versions; non-numeric parts fall back to string."""
    parts: list[Any] = []
    for chunk in str(v).split("."):
        num = "".join(ch for ch in chunk if ch.isdigit())
        parts.append((0, int(num)) if num and num == chunk else (1, chunk))
    return tuple(parts)


def load_artifacts(directory: str | Path) -> list[dict]:
    """Read every *.json proof artifact in a directory (non-recursive)."""
    out: list[dict] = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "tokenjam_version" in d and "benchmark" in d:
            out.append(d)
    return out


def build_series(artifacts: list[dict]) -> list[ConfigSeries]:
    """Group artifacts into per-(benchmark, original) series ordered by version."""
    groups: dict[tuple[str, str], dict[str, dict]] = {}
    for d in artifacts:
        key = (d.get("benchmark", "?"), d.get("original_model", "?"))
        ver = str(d.get("tokenjam_version", "?"))
        bucket = groups.setdefault(key, {})
        # If multiple artifacts share a version, keep the most recent.
        prev = bucket.get(ver)
        if prev is None or d.get("created_at", 0) >= prev.get("created_at", 0):
            bucket[ver] = d

    series: list[ConfigSeries] = []
    for (benchmark, original), by_ver in sorted(groups.items()):
        versions = sorted(by_ver.keys(), key=_version_key)
        points: list[VersionPoint] = []
        for ver in versions:
            d = by_ver[ver]
            stats = d.get("stats", {}) or {}
            points.append(VersionPoint(
                tokenjam_version=ver,
                created_at=float(d.get("created_at", 0.0)),
                candidate_model=d.get("candidate_model", "?"),
                n_tasks=int(d.get("n_tasks", 0)),
                candidate_pass_rate=float(d.get("candidate_pass_rate", 0.0)),
                accuracy_delta_pp=float(d.get("accuracy_delta_pp", 0.0)),
                cost_delta_pct=float(d.get("cost_delta_pct", 0.0)),
                verdict=stats.get("verdict", "?"),
            ))
        _annotate_deltas(points)
        series.append(ConfigSeries(benchmark=benchmark, original_model=original, points=points))
    return series


def _annotate_deltas(points: list[VersionPoint]) -> None:
    """Fill cross-version deltas + regression flags for each point after the first."""
    for i in range(1, len(points)):
        prev, cur = points[i - 1], points[i]
        cur.pass_rate_change_pp = round((cur.candidate_pass_rate - prev.candidate_pass_rate) * 100, 2)
        cur.cost_delta_change_pp = round(cur.cost_delta_pct - prev.cost_delta_pct, 2)
        cur.candidate_changed = cur.candidate_model != prev.candidate_model

        if cur.pass_rate_change_pp <= -ACCURACY_REGRESSION_PP:
            cur.regressions.append(
                f"accuracy −{abs(cur.pass_rate_change_pp):.1f}pp vs {prev.tokenjam_version}")
        if cur.cost_delta_change_pp >= COST_REGRESSION_PP:
            cur.regressions.append(
                f"savings shrank +{cur.cost_delta_change_pp:.1f}pp vs {prev.tokenjam_version}")
        if cur.candidate_changed:
            cur.regressions.append(
                f"recommendation changed {prev.candidate_model} → {cur.candidate_model}")


def total_regressions(series: list[ConfigSeries]) -> int:
    return sum(s.regression_count for s in series)


def series_to_dict(series: list[ConfigSeries]) -> dict:
    """JSON-serialisable view of the matrix."""
    return {
        "regressions_found": total_regressions(series),
        "thresholds": {
            "accuracy_regression_pp": ACCURACY_REGRESSION_PP,
            "cost_regression_pp": COST_REGRESSION_PP,
        },
        "series": [
            {
                "benchmark": s.benchmark,
                "original_model": s.original_model,
                "points": [
                    {
                        "tokenjam_version": p.tokenjam_version,
                        "candidate_model": p.candidate_model,
                        "n_tasks": p.n_tasks,
                        "candidate_pass_rate": round(p.candidate_pass_rate, 4),
                        "accuracy_delta_pp": p.accuracy_delta_pp,
                        "cost_delta_pct": p.cost_delta_pct,
                        "verdict": p.verdict,
                        "pass_rate_change_pp": p.pass_rate_change_pp,
                        "cost_delta_change_pp": p.cost_delta_change_pp,
                        "candidate_changed": p.candidate_changed,
                        "regressions": p.regressions,
                    }
                    for p in s.points
                ],
            }
            for s in series
        ],
    }
