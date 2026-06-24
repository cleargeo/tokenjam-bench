"""Resolve and stamp the TokenJam version under test.

This is the anchor for the "TokenJam changes every day" requirement: every
proof artifact records the EXACT tokenjam build it was produced against, so a
savings/accuracy number is always attributable to a version and the same
benchmark can be re-run across versions to catch regressions.

We read the *installed package* metadata — not a repo checkout — because the
bench is a black-box consumer of the published artifact users actually get.
"""
from __future__ import annotations

import importlib.metadata as _md
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenjamBuild:
    version: str
    location: str | None  # site-packages path, for provenance

    def to_dict(self) -> dict[str, str | None]:
        return {"tokenjam_version": self.version, "tokenjam_location": self.location}


def resolve_tokenjam_build() -> TokenjamBuild:
    """Return the installed tokenjam version, or raise a clear error if absent."""
    try:
        version = _md.version("tokenjam")
    except _md.PackageNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "tokenjam is not installed in this environment. Install it with "
            "`pip install -U tokenjam` (the bench consumes it as a package)."
        ) from exc

    location: str | None = None
    try:
        import tokenjam  # noqa: F401  (import only to read __file__)

        location = getattr(tokenjam, "__file__", None)
    except Exception:  # pragma: no cover - provenance is best-effort
        location = None
    return TokenjamBuild(version=version, location=location)
