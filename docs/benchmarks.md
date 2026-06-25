# Benchmarks

## Overview

The `benchmarks/` package defines executable benchmarks with objective ground truth. Each benchmark implements either the [`Benchmark`](#benchmark) protocol (single-shot) or the [`AgentBenchmark`](#agentbenchmark) protocol (multi-turn).

## Available Benchmarks

| Name | Type | Ground Truth | Needs | Description |
|------|------|-------------|-------|-------------|
| `samples` | Single-shot | Code execution + exact match | Nothing (offline) | 5 tiny tasks (3 code + 2 math) |
| `humaneval` | Single-shot | Unit-test pass/fail | `[datasets]` | OpenAI HumanEval code problems |
| `gsm8k` | Single-shot | Numeric exact match | `[datasets]` | Grade school math problems |
| `sample-agent` | Agent | Tool validation + answer | Nothing (offline) | 3 tool-use tasks with safety gate |
| `swe-bench-lite` | Agent | Tool-use + test execution | `[datasets]` | Real GitHub issue bug fixes (300 tasks) |

## Single-Shot Benchmarks

### Benchmark Protocol

Defined in [`benchmarks/base.py`](../benchmarks/base.py).

```python
class Benchmark(Protocol):
    def tasks(self, limit: int | None = None) -> list[Task]: ...
    def score(self, task: Task, text: str) -> ScoreResult: ...
```

### Task Dataclass

```python
@dataclass
class Task:
    task_id: str
    prompt: str
    kind: str          # "code" or "math"
    test_program_template: str | None
    expected: str | None
    metadata: dict[str, Any]
```

### ScoreResult Dataclass

```python
@dataclass
class ScoreResult:
    passed: bool
    detail: str
```

### Samples Benchmark

[`benchmarks/samples.py`](../benchmarks/samples.py)

5 built-in offline tasks:

1. **hello_world** — Write a function that returns `"hello world"`
2. **add_two** — Write a function that adds two numbers
3. **factorial** — Write a factorial function
4. **two_plus_two** — What is 2+2?
5. **square_nine** — What is 9 squared?

No external dependencies. Runs entirely offline.

### HumanEval Benchmark

[`benchmarks/humaneval.py`](../benchmarks/humaneval.py)

Loads `openai_humaneval` via the `datasets` library. Each task:
- Provides a function signature and docstring
- Model must complete the function body
- Scored by running hidden tests in [`exec_sandbox.py`](../exec_sandbox.py)

Requires: `pip install -e ".[datasets]"`

### GSM8K Benchmark

[`benchmarks/gsm8k.py`](../benchmarks/gsm8k.py)

Loads `gsm8k` via the `datasets` library. Each task:
- Provides a grade-school math word problem
- Model must produce the final numeric answer
- Scored by [`score_exact_match`](#score_exact_match) against the ground truth

Requires: `pip install -e ".[datasets]"`

## Agent Benchmarks

### AgentBenchmark Protocol

Defined in [`benchmarks/agent_base.py`](../benchmarks/agent_base.py).

```python
class AgentBenchmark(Protocol):
    def tools(self) -> ToolRegistry: ...
    def tasks(self, limit: int | None = None) -> list[AgentTask]: ...
    def score(self, task: AgentTask, trace: AgentTrace) -> ScoreResult: ...
```

### AgentTask Dataclass

```python
@dataclass
class AgentTask:
    task_id: str
    prompt: str
    expected_answer: str
    expected_tools: list[str]
    forbidden_tools: list[str]
    expected_order: list[str] | None = None
```

### SampleAgent Benchmark

[`benchmarks/sample_agent.py`](../benchmarks/sample_agent.py)

3 offline tool-use tasks:

1. **lookup_and_compute** — Look up a value, then compute with it
2. **multi_step** — Call tools A then B in order
3. **safety_check** — Task that tries to trick the agent into calling a dangerous tool

Includes a `delete_records` tool marked as **dangerous** to exercise the safety gate.

### SWE-Bench Lite Benchmark

[`benchmarks/swe_bench_lite.py`](../benchmarks/swe_bench_lite.py)

Loads real GitHub issues from the [SWE-Bench Lite dataset](https://github.com/princeton-nlp/SWE-bench) (300 tasks). Each task:
- Provides a real bug report (problem statement from GitHub)
- Includes the repository name and affected files
- The agent must explore, edit, and test to fix the bug

**Agent tools provided:**
- `view` — Read file contents with line numbers
- `view_range` — Read specific line range
- `str_replace` — Exact-match string replacement (must match exactly once)
- `create` — Create new file
- `insert` — Insert text after a specific line
- `bash` — Run shell commands (tests, git, etc.)

**Scoring (mock mode):**
- Checks if agent used `view`, `str_replace`, and `bash` tools
- Full implementation would run FAIL_TO_PASS and PASS_TO_PASS tests

**Scoring (live mode):**
- Would clone the repository at the base commit
- Apply the agent's edits
- Run FAIL_TO_PASS tests (must pass = bug fixed)
- Run PASS_TO_PASS tests (must pass = no regressions)

Requires: `pip install -e ".[datasets]"`

**Usage:**
```bash
# Offline mock mode (no keys needed)
tjbench agent --benchmark swe-bench-lite --original anthropic:claude-opus-4-7 --mock --limit 5

# Live mode (requires API key + dataset)
tjbench agent --benchmark swe-bench-lite --original anthropic:claude-opus-4-7 --limit 10
```

See [SWE-Bench Tools](agents.md#swe-bench-tools) for the tool implementations.

---

### `score_code()`

[`benchmarks/scoring.py`](../benchmarks/scoring.py)

```python
def score_code(task: Task, text: str) -> ScoreResult:
```

1. Extracts code from markdown fences (```python ... ```)
2. Injects it into the test template
3. Runs in [`exec_sandbox.py`](../exec_sandbox.py) (subprocess with timeout)
4. Returns pass/fail + detail

### `score_exact_match()`

```python
def score_exact_match(task: Task, text: str) -> ScoreResult:
```

1. Extracts the final number from the response
2. Prefers `####` marker (e.g., `#### 42`)
3. Falls back to last number in text
4. Normalizes (strips commas, spaces)
5. Compares to expected answer

### `extract_code()`

```python
def extract_code(text: str) -> str:
```

Extracts code from markdown fences. Handles:
- ````python` fences
- ```` ```` plain fences
- No fences (returns entire text)

## Execution Sandbox

[`exec_sandbox.py`](../exec_sandbox.py)

```python
def run_python(code: str, timeout: float = 5.0) -> tuple[bool, str]:
```

- Runs code in a subprocess with a timeout
- Creates a temp directory for isolation
- Returns (success, output_or_error)
- Safe for untrusted code (subprocess boundary)

## Registry

[`benchmarks/__init__.py`](../benchmarks/__init__.py)

```python
def get_benchmark(name: str) -> Benchmark:
    """Get a single-shot benchmark by name."""

def get_agent_benchmark(name: str) -> AgentBenchmark:
    """Get an agent benchmark by name."""
```

Supported names:
- `get_benchmark`: `samples`, `humaneval`, `gsm8k`
- `get_agent_benchmark`: `sample-agent`, `swe-bench-lite`

## Related Documentation

- [Pipelines](pipelines.md) — How benchmarks are used in proof pipelines
- [Agents](agents.md) — AgentRunner and tool validation
- [Models](models.md) — How clients generate responses for benchmarks
- [Statistics](statistics.md) — How pass rates are analyzed
- [CLI Reference](cli-reference.md) — `--benchmark` flag
