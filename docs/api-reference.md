# API Reference

## Module-Level Reference

### `pipeline.py`

#### `run_proof()`

```python
def run_proof(
    benchmark: Benchmark,
    original_spec: str,
    candidate_spec: str | None = None,
    *,
    mock: bool = False,
    mock_candidate_accuracy: float = 1.0,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    samples: int = 1,
    limit: int | None = None,
) -> ProofResult:
```

Main entry point for single-shot proof. See [Pipelines](pipelines.md#single-shot-pipeline).

#### `assemble_proof()`

```python
def assemble_proof(
    outcomes: list[TaskOutcome],
    original_spec: str,
    candidate_spec: str,
    *,
    benchmark_name: str,
    mock: bool = False,
    tokenjam_version: str = "",
    tokenjam_location: str = "",
) -> ProofResult:
```

Shared assembler for both pipelines. Computes statistics, CIs, verdict. See [Pipelines](pipelines.md#assemble_proof).

---

### `agent_pipeline.py`

#### `run_agent_proof()`

```python
def run_agent_proof(
    benchmark: AgentBenchmark,
    original_spec: str,
    candidate_spec: str | None = None,
    *,
    mock: bool = False,
    candidate_behavior: str = "ok",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    max_turns: int = 10,
    samples: int = 1,
    limit: int | None = None,
) -> ProofResult:
```

Main entry point for agent proof. See [Pipelines](pipelines.md#agent-proof-pipeline).

---

### `report.py`

#### `TaskOutcome`

```python
@dataclass
class TaskOutcome:
    task_id: str
    original_passed: bool
    candidate_passed: bool
    original_cost: float
    candidate_cost: float
    original_tokens: int
    candidate_tokens: int
    detail: str
```

Per-task result collected during pipeline runs.

#### `ProofStats`

```python
@dataclass
class ProofStats:
    original_ci: tuple[float, float]
    candidate_ci: tuple[float, float]
    mcnemar_p: float
    mcnemar_significant: bool
    delta_ci: tuple[float, float]
    verdict: str
    pass_at_1: float
    pass_at_k: float | None
```

Statistical summary of a proof.

#### `ProofResult`

```python
@dataclass
class ProofResult:
    benchmark_name: str
    original_spec: str
    candidate_spec: str
    n_tasks: int
    n_samples: int
    original_pass_rate: float
    candidate_pass_rate: float
    original_cost: float
    candidate_cost: float
    cost_delta: float
    cost_delta_percent: float
    token_inflation: bool
    stats: ProofStats
    tasks: list[TaskOutcome]
    regressions: list[str]
    tokenjam_version: str
    tokenjam_location: str
    bench_version: str
    timestamp: str
    mock: bool

    def to_dict(self) -> dict: ...
    def write(self, out_dir: str) -> Path: ...
    def headline(self) -> str: ...
```

The full aggregate result. `write()` produces a version-stamped JSON file in `out_dir/`.

---

### `stats.py`

#### `wilson_interval()`

```python
def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
```

Wilson score interval for binomial proportion. See [Statistics](statistics.md#wilson-interval).

#### `mcnemar_exact()`

```python
def mcnemar_exact(b: int, c: int) -> tuple[float, bool]:
```

Exact two-sided McNemar test. See [Statistics](statistics.md#mcnemar-exact-test).

#### `paired_delta_ci()`

```python
def paired_delta_ci(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float]:
```

CI on paired pass-rate delta. See [Statistics](statistics.md#paired-delta-ci).

#### `pass_at_k()`

```python
def pass_at_k(n: int, c: int, k: int) -> float:
```

Unbiased pass@k estimator. See [Statistics](statistics.md#passk-estimator).

---

### `cost.py`

#### `price_completion()`

```python
def price_completion(
    provider: str,
    model: str,
    completion: Completion,
) -> float:
```

Compute USD cost using TokenJam's pricing table. See [Cost & Pricing](cost-pricing.md).

---

### `recommend.py`

#### `resolve_candidate()`

```python
def resolve_candidate(original_spec: str) -> str | None:
```

Look up the cheaper model TokenJam recommends. Returns `None` if TokenJam has no recommendation. See [TokenJam Integration](tokenjam-integration.md#candidate-recommendation).

---

### `version.py`

#### `resolve_tokenjam_build()`

```python
def resolve_tokenjam_build() -> tuple[str, str]:
```

Returns `(version_string, location_string)` for the installed `tokenjam` package. See [TokenJam Integration](tokenjam-integration.md#version-stamp).

---

### `exec_sandbox.py`

#### `run_python()`

```python
def run_python(code: str, timeout: float = 5.0) -> tuple[bool, str]:
```

Run Python code in a subprocess with timeout. Returns `(success, output_or_error)`. See [Benchmarks](benchmarks.md#execution-sandbox).

---

### `models/base.py`

#### `ModelClient` (Protocol)

```python
class ModelClient(Protocol):
    def complete(self, prompt: str, *, system=None, max_tokens=None, temperature=0.0) -> Completion: ...
```

See [Models](models.md#modelclient).

#### `Completion`

```python
@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    cache_tokens: int = 0
```

---

### `models/tool_calling.py`

#### `ToolCallingClient` (Protocol)

```python
class ToolCallingClient(Protocol):
    def chat(self, messages, tools, *, temperature=0.0, max_tokens=None) -> AssistantTurn: ...
```

See [Models](models.md#toolcallingclient).

#### `AssistantTurn`

```python
@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
```

#### `ToolCall`

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
```

---

### `models/registry.py`

#### `parse_spec()`

```python
def parse_spec(spec: str) -> tuple[str, str]:
```

Parse `"provider:model"` into `(provider, model)`.

#### `get_client()`

```python
def get_client(spec: str, *, mock=False, mock_accuracy=1.0) -> ModelClient:
```

Factory for single-shot clients. See [Models](models.md#registry).

#### `get_tool_calling_client()`

```python
def get_tool_calling_client(spec: str, *, mock=False, behavior="ok") -> ToolCallingClient:
```

Factory for tool-calling clients. See [Models](models.md#registry).

---

### `benchmarks/base.py`

#### `Benchmark` (Protocol)

```python
class Benchmark(Protocol):
    def tasks(self, limit=None) -> list[Task]: ...
    def score(self, task, text) -> ScoreResult: ...
```

See [Benchmarks](benchmarks.md#benchmark-protocol).

#### `Task`

```python
@dataclass
class Task:
    task_id: str
    prompt: str
    kind: str
    test_program_template: str | None
    expected: str | None
    metadata: dict
```

#### `ScoreResult`

```python
@dataclass
class ScoreResult:
    passed: bool
    detail: str
```

---

### `benchmarks/agent_base.py`

#### `AgentBenchmark` (Protocol)

```python
class AgentBenchmark(Protocol):
    def tools(self) -> ToolRegistry: ...
    def tasks(self, limit=None) -> list[AgentTask]: ...
    def score(self, task, trace) -> ScoreResult: ...
```

See [Benchmarks](benchmarks.md#agentbenchmark-protocol).

#### `AgentTask`

```python
@dataclass
class AgentTask:
    task_id: str
    prompt: str
    expected_answer: str
    expected_tools: list[str]
    forbidden_tools: list[str]
    expected_order: list[str] | None
```

---

### `agents/runner.py`

#### `AgentRunner`

```python
class AgentRunner:
    def __init__(self, client: ToolCallingClient, registry: ToolRegistry, max_turns=10): ...
    def run(self, task_id: str, prompt: str) -> AgentTrace: ...
```

See [Agents](agents.md#agentrunner).

---

### `agents/tools.py`

#### `ToolRegistry`

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None: ...
    def execute(self, name: str, arguments: dict) -> ToolResult: ...
    def specs(self) -> list[dict]: ...
```

See [Agents](agents.md#toolregistry).

#### `Tool`

```python
@dataclass
class Tool:
    name: str
    schema: dict
    dangerous: bool
    run: Callable[[dict], str]
```

---

### `agents/trace.py`

#### `AgentTrace`

```python
@dataclass
class AgentTrace:
    task_id: str
    turns: list[TurnRecord]
    final_text: str
    stopped_reason: str
```

See [Agents](agents.md#agenttrace).

---

### `agents/validation.py`

#### `validate_tools()`

```python
def validate_tools(trace, registry, expected_tools, forbidden_tools, expected_order=None) -> ToolValidation:
```

See [Agents](agents.md#tool-validation).

#### `ToolValidation`

```python
@dataclass
class ToolValidation:
    called_tools: list[str]
    expected_tools_called: bool
    forbidden_called: bool
    ordering_ok: bool
    safe: bool
```

---

## Related Documentation

- [Architecture](architecture.md) — System design
- [Pipelines](pipelines.md) — How the API is used
- [Models](models.md) — Model protocols
- [Benchmarks](benchmarks.md) — Benchmark protocols
- [Agents](agents.md) — Agent execution
- [Statistics](statistics.md) — Statistical methods
- [Cost & Pricing](cost-pricing.md) — Cost computation
- [TokenJam Integration](tokenjam-integration.md) — TokenJam APIs
