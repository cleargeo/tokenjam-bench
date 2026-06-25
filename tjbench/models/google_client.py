"""Live Google Gemini client. Lazy-imports `google-genai`; reads
GEMINI_API_KEY / GOOGLE_API_KEY per the SDK's own resolution.
"""
from __future__ import annotations

from tjbench.models.base import Completion


class GoogleClient:
    provider = "google"

    def __init__(self, model: str) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The google-genai SDK is not installed. Run "
                "`pip install 'tokenjam-bench[providers]'`."
            ) from exc
        self.model = model
        self._client = genai.Client()

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> Completion:
        from google.genai import types

        contents = prompt if not system else f"{system}\n\n{prompt}"
        resp = self._client.models.generate_content(
            model=self.model, contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature, max_output_tokens=max_tokens,
            ),
        )
        text = getattr(resp, "text", "") or ""
        meta = getattr(resp, "usage_metadata", None)
        return Completion(
            text=text,
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            cache_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
        )
