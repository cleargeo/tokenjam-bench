"""Cost = priced by TokenJam's own pricing table.

We intentionally do NOT carry our own price list. `tokenjam.core.pricing` is the
single source of truth so the dollars in a proof report match what TokenJam
itself would report — and any pricing change in a new TokenJam release flows
straight into the bench (and is captured by the version stamp).
"""
from __future__ import annotations

from tokenjam.core.pricing import get_rates

from tjbench.models.base import Completion


def price_completion(provider: str, model: str, c: Completion) -> float:
    """USD cost of one completion, using TokenJam's rates.

    Falls back to TokenJam's own default-rate behavior for unknown models
    (get_rates returns None → we apply tokenjam's documented $0.50/$2.00
    per-MTok default). We surface that the price was a default via
    `priced_with_defaults` on the dict variant below.
    """
    rates = get_rates(provider, model)
    if rates is None:
        in_rate, out_rate, cache_rate = 0.50, 2.00, 0.0
    else:
        in_rate = rates.input_per_mtok
        out_rate = rates.output_per_mtok
        cache_rate = rates.cache_read_per_mtok
    return round(
        (c.input_tokens / 1_000_000) * in_rate
        + (c.output_tokens / 1_000_000) * out_rate
        + (c.cache_tokens / 1_000_000) * cache_rate,
        8,
    )


def is_priced_with_defaults(provider: str, model: str) -> bool:
    """True when TokenJam has no rate for this model (cost used the flat default).

    Surfaced in the report so a 'savings' number derived from placeholder
    pricing is never presented as if it came from a real rate.
    """
    return get_rates(provider, model) is None
