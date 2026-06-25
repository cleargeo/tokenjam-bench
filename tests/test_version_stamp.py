"""The version stamp is the anchor for 'TokenJam changes every day'."""
from __future__ import annotations

from tjbench.version import resolve_tokenjam_build


def test_resolves_installed_tokenjam_version():
    build = resolve_tokenjam_build()
    # A real semver-ish string from the installed package, not empty.
    assert build.version
    assert build.version[0].isdigit()
    d = build.to_dict()
    assert d["tokenjam_version"] == build.version
