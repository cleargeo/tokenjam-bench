"""Live OpenAI client. Lazy-imports the `openai` SDK; reads OPENAI_API_KEY."""
from __future__ import annotations

from models.base import Completion


class OpenAIClient:
    provider = "openai"

    def __init__(self, model: str) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The openai SDK is not installed. Run "
                "`pip install 'tokenjam-bench[providers]'`."
            ) from exc
        self.model = model
        self._client = openai.OpenAI()

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return Completion(
            text=text,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cache_tokens=cached,
        )
