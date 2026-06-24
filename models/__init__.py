"""Model adapters: provider-agnostic ModelClient + live clients + offline mock."""
from __future__ import annotations

from models.base import Completion, ModelClient
from models.registry import get_client, parse_spec

__all__ = ["Completion", "ModelClient", "get_client", "parse_spec"]
