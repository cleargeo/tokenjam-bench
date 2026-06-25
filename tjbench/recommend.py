"""The CANDIDATE comes from TokenJam, not from us.

This is what makes the bench a *proof of TokenJam* rather than a generic
model-comparison tool: the cheaper model we replay against is the one TokenJam's
own downsize analyzer would route to (its DOWNGRADE_CANDIDATES map). If a new
TokenJam release changes that map, the bench tests the new recommendation
automatically — and the version stamp records which mapping was in effect.
"""
from __future__ import annotations

import re

from tokenjam.core.optimize import DOWNGRADE_CANDIDATES


def candidate_for(provider: str, model: str) -> str | None:
    """TokenJam's recommended cheaper model for (provider, model), or None.

    Tolerates a trailing -YYYYMMDD date suffix the same way TokenJam's analyzer
    does, so dated model names still resolve.
    """
    mapping = DOWNGRADE_CANDIDATES.get(provider, {})
    if model in mapping:
        return mapping[model]
    m = re.match(r"^(.*)-(\d{8})$", model)
    if m and m.group(1) in mapping:
        return mapping[m.group(1)]
    return None
