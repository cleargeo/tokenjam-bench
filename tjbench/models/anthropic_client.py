"""Live Anthropic client. Lazy-imports the `anthropic` SDK so the package
imports without it; install with `tokenjam-bench[providers]`. Reads
ANTHROPIC_API_KEY from the environment.
"""
from __future__ import annotations

from tjbench.models.base import Completion


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "The anthropic SDK is not installed. Run "
                "`pip install 'tokenjam-bench[providers]'`."
            ) from exc
        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> Completion:
        import anthropic
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        # `temperature` is rejected by some newer models (e.g. claude-opus-4-7,
        # which deprecated it). Send it, but transparently retry without it if
        # the API reports it unsupported, so the harness prices the real run.
        try:
            resp = self._client.messages.create(temperature=temperature, **kwargs)
        except anthropic.BadRequestError as exc:
            if "temperature" not in str(exc):
                raise
            resp = self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
        usage = resp.usage
        return Completion(
            text=text,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
