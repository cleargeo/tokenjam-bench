"""Resolve a `provider:model` spec to a ModelClient.

Spec form: "anthropic:claude-opus-4-7", "openai:gpt-4o", "google:gemini-2-5-pro",
or "mock:anything". Live clients are constructed lazily so importing this module
never requires a provider SDK or a key.
"""
from __future__ import annotations

from models.base import ModelClient
from models.mock_client import MockClient

_LIVE_PROVIDERS = {"anthropic", "openai", "google"}


def parse_spec(spec: str) -> tuple[str, str]:
    """Split 'provider:model' → (provider, model)."""
    if ":" not in spec:
        raise ValueError(
            f"Model spec '{spec}' must be 'provider:model' "
            f"(e.g. anthropic:claude-opus-4-7)."
        )
    provider, model = spec.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise ValueError(f"Model spec '{spec}' is missing a provider or model.")
    return provider, model


def get_client(spec: str, *, mock: bool = False, mock_accuracy: float = 1.0) -> ModelClient:
    """Build a client for the spec. `mock=True` forces the offline client
    (preserving the real provider/model labels for honest reporting)."""
    provider, model = parse_spec(spec)
    if mock or provider == "mock":
        return MockClient(model=model, provider=provider, accuracy=mock_accuracy)
    if provider == "anthropic":
        from models.anthropic_client import AnthropicClient
        return AnthropicClient(model)
    if provider == "openai":
        from models.openai_client import OpenAIClient
        return OpenAIClient(model)
    if provider == "google":
        from models.google_client import GoogleClient
        return GoogleClient(model)
    raise ValueError(
        f"Unknown provider '{provider}'. Supported: {sorted(_LIVE_PROVIDERS)} (or 'mock')."
    )
