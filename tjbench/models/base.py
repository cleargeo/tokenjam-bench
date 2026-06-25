"""Provider-agnostic model interface.

A `ModelClient` turns a prompt into a `Completion` carrying the text plus token
usage. Cost is deliberately NOT computed here — it's priced downstream by
TokenJam's pricing table (see cost.py) so the bench reports the
same dollars TokenJam would.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cache_tokens: int = 0


@runtime_checkable
class ModelClient(Protocol):
    provider: str
    model: str

    def complete(
        self, prompt: str, system: str | None = ..., max_tokens: int = ...,
        temperature: float = ...,
    ) -> Completion: ...
