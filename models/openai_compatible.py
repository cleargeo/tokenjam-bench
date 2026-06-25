"""Provider-agnostic OpenAI-compatible clients.

OpenAI, DeepSeek, and any future OpenAI-compatible endpoint share ONE abstraction
here: a provider entry (base_url + which env var holds the key + default model),
and clients that talk the OpenAI Chat Completions shape. Adding a provider is a
one-line entry in `PROVIDERS` — no new client class.

Secret handling: the API key is read from the environment on each call and never
stored on the instance, logged, printed, or written anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from models.base import Completion
from models.tool_calling import AssistantTurn, ToolCall


@dataclass(frozen=True)
class OpenAICompatProvider:
    name: str
    base_url: str | None     # None → OpenAI's default endpoint
    api_key_env: str         # env var that holds the key (never hard-coded)
    default_model: str


# The provider registry. Future OpenAI-compatible providers: add an entry.
PROVIDERS: dict[str, OpenAICompatProvider] = {
    "openai": OpenAICompatProvider("openai", None, "OPENAI_API_KEY", "gpt-4o"),
    "deepseek": OpenAICompatProvider(
        "deepseek", "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat"),
}


def is_openai_compatible(provider: str) -> bool:
    return provider in PROVIDERS


def _make_openai_client(provider: OpenAICompatProvider):
    """Build an OpenAI SDK client for a provider; key read from env, never stored."""
    try:
        import openai
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The openai SDK is not installed. Run `pip install 'tokenjam-bench[providers]'`."
        ) from exc
    key = os.environ.get(provider.api_key_env)
    if not key:
        raise RuntimeError(
            f"{provider.api_key_env} is not set. Export it (it is read from the "
            f"environment only and never persisted)."
        )
    return openai.OpenAI(api_key=key, base_url=provider.base_url)


def _usage_tokens(usage) -> tuple[int, int, int]:
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return (getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0, cached)


class OpenAICompatibleClient:
    """Single-shot completion client for any OpenAI-compatible provider."""

    def __init__(self, model: str, provider_name: str = "openai") -> None:
        if provider_name not in PROVIDERS:
            raise ValueError(f"'{provider_name}' is not an OpenAI-compatible provider.")
        self.provider = provider_name
        self.model = model
        self._prov = PROVIDERS[provider_name]

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> Completion:
        client = _make_openai_client(self._prov)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens,
            temperature=temperature,
        )
        in_tok, out_tok, cached = _usage_tokens(resp.usage)
        return Completion(
            text=resp.choices[0].message.content or "",
            input_tokens=in_tok, output_tokens=out_tok, cache_tokens=cached,
        )


class OpenAICompatibleAgentClient:
    """Tool-calling client for any OpenAI-compatible provider (agent benchmarks)."""

    def __init__(self, model: str, provider_name: str = "openai") -> None:
        if provider_name not in PROVIDERS:
            raise ValueError(f"'{provider_name}' is not an OpenAI-compatible provider.")
        self.provider = provider_name
        self.model = model
        self._prov = PROVIDERS[provider_name]

    @staticmethod
    def _to_openai_messages(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "tool":
                out.append({"role": "tool", "tool_call_id": m.get("tool_call_id"),
                            "content": m.get("content", "")})
            elif role == "assistant" and m.get("tool_calls"):
                import json
                out.append({"role": "assistant", "content": m.get("content") or None,
                            "tool_calls": [
                                {"id": tc.id, "type": "function",
                                 "function": {"name": tc.name,
                                              "arguments": json.dumps(tc.arguments)}}
                                for tc in m["tool_calls"]]})
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return out

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("parameters", {"type": "object", "properties": {}})}}
            for t in tools]

    def chat(self, messages: list[dict], tools: list[dict],
             temperature: float = 0.0, max_tokens: int = 1024) -> AssistantTurn:
        import json
        client = _make_openai_client(self._prov)
        resp = client.chat.completions.create(
            model=self.model, messages=self._to_openai_messages(messages),
            tools=self._to_openai_tools(tools) or None,
            temperature=temperature, max_tokens=max_tokens,
        )
        choice = resp.choices[0].message
        calls = []
        for tc in (choice.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        in_tok, out_tok, cached = _usage_tokens(resp.usage)
        return AssistantTurn(
            text=choice.content or "", tool_calls=calls,
            input_tokens=in_tok, output_tokens=out_tok, cache_tokens=cached,
        )
