"""Live Anthropic tool-calling client.

Translates the provider-neutral message list (see models/tool_calling.py) into
Anthropic's Messages API tool-use shape and back. This is the primary live agent
path. It is exercised only with a real ANTHROPIC_API_KEY — there's no offline
test for it (the MockAgentClient is the tested path), so it's kept deliberately
small and close to the documented API.
"""
from __future__ import annotations

import json
from typing import Any

from models.tool_calling import AssistantTurn, ToolCall


class AnthropicAgentClient:
    provider = "anthropic"

    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The anthropic SDK is not installed. Run "
                "`pip install 'tokenjam-bench[providers]'`."
            ) from exc
        self.model = model
        self._client = anthropic.Anthropic()

    @staticmethod
    def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"name": t["name"], "description": t.get("description", ""),
             "input_schema": t.get("parameters", {"type": "object", "properties": {}})}
            for t in tools
        ]

    @staticmethod
    def _to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        from models.openai_compatible import strip_mock_directives
        for m in messages:
            role = m.get("role")
            if role == "user":
                # Strip mock-only scaffolding so live models never see the plan.
                out.append({"role": "user", "content": strip_mock_directives(m.get("content", ""))})
            elif role == "assistant":
                content: list[dict[str, Any]] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content.append({
                        "type": "tool_use", "id": tc.id, "name": tc.name,
                        "input": tc.arguments,
                    })
                out.append({"role": "assistant", "content": content})
            elif role == "tool":
                out.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id"),
                    "content": m.get("content", ""),
                }]})
        return out

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
             temperature: float = 0.0, max_tokens: int = 1024) -> AssistantTurn:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=self._to_anthropic_tools(tools),
            messages=self._to_anthropic_messages(messages),
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                args = block.input if isinstance(block.input, dict) else json.loads(block.input)
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=args))
        usage = resp.usage
        return AssistantTurn(
            text="".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )


def get_tool_calling_client(spec: str, *, mock: bool = False, behavior: str = "ok"):
    """Resolve 'provider:model' → a ToolCallingClient.

    Offline (`mock=True` or provider 'mock') → MockAgentClient. Live → the
    provider's agent client (Anthropic implemented; others raise until built).
    """
    from models.mock_agent_client import MockAgentClient
    from models.registry import parse_spec

    provider, model = parse_spec(spec)
    if mock or provider == "mock":
        return MockAgentClient(model=model, provider=provider, behavior=behavior)
    if provider == "anthropic":
        return AnthropicAgentClient(model)
    from models.openai_compatible import PROVIDERS, OpenAICompatibleAgentClient
    if provider in PROVIDERS:  # openai, deepseek, future compatible providers
        return OpenAICompatibleAgentClient(model, provider_name=provider)
    raise NotImplementedError(
        f"Live agent runs for provider '{provider}' aren't implemented yet. "
        f"Supported: anthropic, {sorted(PROVIDERS)}. Or use --mock."
    )
