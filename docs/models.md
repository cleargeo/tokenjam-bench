# Models

## Overview

The `models/` package abstracts provider-specific APIs into two protocols:

1. [`ModelClient`](#modelclient) — Single-shot completion: `complete(prompt) → Completion`
2. [`ToolCallingClient`](#toolcallingclient) — Multi-turn chat with tool use: `chat(messages, tools) → AssistantTurn`

All live clients lazy-import their SDKs so the package imports without them. Mock clients provide deterministic offline behavior for CI/testing.

## ModelClient Protocol

Defined in [`models/base.py`](../models/base.py).

```python
class ModelClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> Completion: ...
```

### Completion Dataclass

```python
@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cache_tokens: int = 0
```

## ToolCallingClient Protocol

Defined in [`models/tool_calling.py`](../models/tool_calling.py).

```python
class ToolCallingClient(Protocol):
    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AssistantTurn: ...
```

### AssistantTurn Dataclass

```python
@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0
```

### ToolCall Dataclass

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
```

## Live Clients

### AnthropicClient

[`models/anthropic_client.py`](../models/anthropic_client.py)

- Lazy-imports `anthropic`
- Requires `ANTHROPIC_API_KEY`
- Supports `claude-*` models
- Handles system prompts, max_tokens, temperature

### OpenAIClient

[`models/openai_client.py`](../models/openai_client.py)

- Lazy-imports `openai`
- Requires `OPENAI_API_KEY`
- Supports `gpt-*` models and OpenAI-compatible providers

### GoogleClient

[`models/google_client.py`](../models/google_client.py)

- Lazy-imports `google.generativeai`
- Requires `GEMINI_API_KEY`
- Supports `gemini-*` models

### AnthropicAgentClient

[`models/anthropic_agent_client.py`](../models/anthropic_agent_client.py)

- Tool-calling variant for Anthropic
- Returns `AssistantTurn` with `ToolCall` objects
- Handles tool result messages in the conversation loop

## Mock Clients

### MockClient

[`models/mock_client.py`](../models/mock_client.py)

Deterministic offline client for single-shot benchmarks.

- Reads `# task_key:` markers from prompts
- Returns predetermined responses based on the task key
- No SDKs, no keys, no network
- `mock_candidate_accuracy` controls pass rate

### MockAgentClient

[`models/mock_agent_client.py`](../models/mock_agent_client.py)

Deterministic offline client for agent benchmarks.

- Simulates tool-calling behavior
- `candidate_behavior` modes: `ok`, `wrong`, `unsafe`
- Returns `AssistantTurn` with mock `ToolCall` objects

## Registry

[`models/registry.py`](../models/registry.py)

### `parse_spec(spec: str) → tuple[str, str]`

Parses `"provider:model"` strings.

```python
>>> parse_spec("anthropic:claude-opus-4-7")
("anthropic", "claude-opus-4-7")
```

### `get_client(spec: str, *, mock: bool = False, mock_accuracy: float = 1.0) → ModelClient`

Factory for single-shot clients.

```python
client = get_client("anthropic:claude-opus-4-7", mock=True)
completion = client.complete("What is 2+2?")
```

### `get_tool_calling_client(spec: str, *, mock: bool = False, behavior: str = "ok") → ToolCallingClient`

Factory for tool-calling clients.

```python
client = get_tool_calling_client("anthropic:claude-opus-4-7", mock=True, behavior="ok")
turn = client.chat(messages, tools)
```

## Provider Spec Format

```
provider:model
```

Examples:
- `anthropic:claude-opus-4-7`
- `anthropic:claude-sonnet-4-20250514`
- `openai:gpt-4o`
- `openai:gpt-4o-mini`
- `google:gemini-2-5-pro`

The registry tolerates dated model suffixes (e.g., `-20250514`) by stripping them during client construction.

## Related Documentation

- [Pipelines](pipelines.md) — How clients are used in proof pipelines
- [Benchmarks](benchmarks.md) — How benchmarks invoke clients
- [Agents](agents.md) — How AgentRunner uses ToolCallingClient
- [CLI Reference](cli-reference.md) — `--original`, `--candidate`, `--mock` flags
- [TokenJam's Provider Patches](https://github.com/HoomanDigital/tokenjam/blob/main/docs/python-sdk.md) — How TokenJam intercepts live provider calls
