# Architecture

## System Design

tokenjam-bench is a **flat-layout Python package** (no inner package directory). The top-level `.py` files and subpackages live directly at the repo root.

```
tokenjam-bench/
├── pyproject.toml          # Build config (hatchling), deps, scripts, pytest, ruff
├── Makefile                # install, update-tokenjam, test, lint, bench-smoke
├── README.md               # Project overview, quickstart, benchmark table
├── .gitignore
├── results/.gitkeep          # Output directory for JSON artifacts
│
├── cli.py                  # Click CLI: tjbench version | recommend | run | agent
├── pipeline.py             # Single-shot proof pipeline (run_proof, assemble_proof)
├── agent_pipeline.py       # Agent proof pipeline (run_agent_proof)
├── report.py               # ProofResult, ProofStats, TaskOutcome dataclasses
├── stats.py                # Wilson interval, McNemar exact, paired delta CI, pass@k
├── cost.py                 # Cost pricing via tokenjam.core.pricing.get_rates
├── recommend.py            # Resolves downgrade candidate from tokenjam.core.optimize
├── version.py              # Resolves installed tokenjam version for stamping
├── exec_sandbox.py         # Subprocess sandbox for executing model-generated code
├── bench_meta.py           # Bench package version (0.1.0)
│
├── models/                 # Model client adapters
│   ├── __init__.py
│   ├── base.py             # Completion dataclass, ModelClient protocol
│   ├── registry.py         # parse_spec, get_client (live + mock)
│   ├── anthropic_client.py # Live Anthropic single-shot client
│   ├── openai_client.py    # Live OpenAI single-shot client
│   ├── google_client.py    # Live Google Gemini single-shot client
│   ├── mock_client.py      # Offline deterministic single-shot client
│   ├── anthropic_agent_client.py # Live Anthropic tool-calling client
│   ├── mock_agent_client.py      # Offline deterministic tool-calling client
│   └── tool_calling.py     # ToolCall, AssistantTurn, ToolCallingClient protocol
│
├── benchmarks/             # Benchmark definitions + scoring
│   ├── __init__.py         # Registry: get_benchmark, get_agent_benchmark
│   ├── base.py             # Task, ScoreResult, Benchmark protocol
│   ├── scoring.py          # extract_code, score_code, score_exact_match
│   ├── samples.py          # Built-in offline sample benchmark (code + math)
│   ├── humaneval.py        # HumanEval code benchmark (datasets extra)
│   ├── gsm8k.py            # GSM8K math benchmark (datasets extra)
│   ├── agent_base.py       # AgentTask, AgentBenchmark protocol
│   └── sample_agent.py     # Offline sample agent benchmark (tool use + safety)
│
├── agents/                 # Multi-turn agent execution
│   ├── __init__.py
│   ├── runner.py           # AgentRunner: the multi-turn loop
│   ├── tools.py            # Tool, ToolResult, ToolRegistry
│   ├── trace.py            # AgentTrace, TurnRecord, ToolCallRecord
│   └── validation.py       # validate_tools, ToolValidation (safety gate)
│
└── tests/                  # Test suite (pytest, offline, no keys)
    ├── test_pipeline_offline.py
    ├── test_agent_pipeline_offline.py
    ├── test_agent_runner.py
    ├── test_agent_validation.py
    ├── test_scoring.py
    ├── test_stats.py
    ├── test_report.py
    └── test_version_stamp.py
```

## Data Flow

### Single-Shot Proof Pipeline

```
CLI: tjbench run --benchmark samples --original anthropic:claude-opus-4-7 --mock
    │
    ▼
resolve_candidate("anthropic:claude-opus-4-7")
    → query [tokenjam.core.optimize.DOWNGRADE_CANDIDATES](https://github.com/HoomanDigital/tokenjam/blob/main/tokenjam/core/optimize/analyzers/model_downgrade.py)
    → "anthropic:claude-haiku-4-5"
    │
    ▼
get_client(original_spec, mock=...)  → [ModelClient](models.md#modelclient)
get_client(candidate_spec, mock=...)  → [ModelClient](models.md#modelclient)
    │
    ▼
get_benchmark("samples") → [SampleBenchmark](benchmarks.md#samplebenchmark)
benchmark.tasks(limit=...) → list[Task]
    │
    ▼  (for each task)
_run_samples(original, ...)  → (passes, output_tokens, cost, detail)
_run_samples(candidate, ...)   → (passes, output_tokens, cost, detail)
    │
    ▼  (score each sample)
benchmark.score(task, completion_text) → [ScoreResult](benchmarks.md#scoreresult)
    │
    ▼  (price each aggregated completion)
[price_completion](cost-pricing.md)(provider, model, Completion) → USD
    │
    ▼
Collect per-task [TaskOutcome](api-reference.md#taskoutcome) objects
    │
    ▼
[assemble_proof](pipelines.md#assemble_proof)(outcomes, ...)
    │
    ├── resolve_tokenjam_build() → version stamp
    ├── [wilson_interval](statistics.md#wilson_interval)(orig_pass, n) → original CI
    ├── [wilson_interval](statistics.md#wilson_interval)(cand_pass, n) → candidate CI
    ├── [mcnemar_exact](statistics.md#mcnemar_exact)(b, c) → p-value, significance
    ├── [paired_delta_ci](statistics.md#paired_delta_ci)(b, c, n) → delta CI
    ├── _verdict(n, significant, delta_pp) → verdict string
    │
    ▼
[ProofResult](api-reference.md#proofresult)
    ├── to_dict() → JSON-serializable dict
    ├── write(out_dir) → version-stamped JSON file
    └── headline() → human-readable summary
    │
    ▼
CLI renders Rich table (or --json)
```

### Agent Proof Pipeline

```
CLI: tjbench agent --benchmark sample-agent --original anthropic:claude-opus-4-7 --mock
    │
    ▼
Same candidate resolution via [recommend.py](api-reference.md#recommendpy)
    │
    ▼
get_tool_calling_client(...) → [ToolCallingClient](models.md#toolcallingclient) (live or mock)
    │
    ▼
get_agent_benchmark("sample-agent") → [SampleAgentBenchmark](benchmarks.md#sampleagentbenchmark)
benchmark.tools() → [ToolRegistry](agents.md#toolregistry)
benchmark.tasks(limit=...) → list[AgentTask]
    │
    ▼  (for each task)
_run_agent_samples(original, ...)
    │
    ├── [AgentRunner](agents.md#agentrunner)(client, registry, max_turns).run(task_id, prompt)
    │       │
    │       ├── loop up to max_turns:
    │       │   client.chat(messages, tools) → [AssistantTurn](models.md#assistantturn)
    │       │   if turn.wants_tools: execute tools, feed results back
    │       │   else: final answer, stop
    │       │
    │       └── returns [AgentTrace](agents.md#agenttrace) (turns, final_text, token sums)
    │
    ├── benchmark.score(task, trace) → [ScoreResult](benchmarks.md#scoreresult)
    │       └── [validate_tools](agents.md#validate_tools)(trace, registry, expected_tools, forbidden_tools, expected_order)
    │           → safety gate, ordering, tool errors
    │
    └── [price_completion](cost-pricing.md)(provider, model, trace.as_completion()) → USD
    │
    ▼
Same [assemble_proof](pipelines.md#assemble_proof) as single-shot → [ProofResult](api-reference.md#proofresult)
```

## Module Relationships

```
models/ ──► protocols used by pipeline, agent_pipeline, benchmarks, agents
    │
    ├── base.py ──► [ModelClient](models.md#modelclient) protocol
    │   ├── registry.py ──► client factory
    │   ├── anthropic_client.py ──► live Anthropic
    │   ├── openai_client.py ──► live OpenAI
    │   ├── google_client.py ──► live Google
    │   └── mock_client.py ──► offline deterministic
    │
    └── tool_calling.py ──► [ToolCallingClient](models.md#toolcallingclient) protocol
        ├── anthropic_agent_client.py ──► live tool-calling
        └── mock_agent_client.py ──► offline tool-calling

benchmarks/ ──► task definitions and scoring
    │
    ├── base.py ──► [Benchmark](benchmarks.md#benchmark) protocol
    │   ├── samples.py ──► offline smoke benchmark
    │   ├── humaneval.py ──► HumanEval loader
    │   └── gsm8k.py ──► GSM8K loader
    │
    ├── agent_base.py ──► [AgentBenchmark](benchmarks.md#agentbenchmark) protocol
    │   └── sample_agent.py ──► offline agent benchmark
    │
    └── scoring.py ──► [score_code](benchmarks.md#score_code), [score_exact_match](benchmarks.md#score_exact_match)

agents/ ──► multi-turn execution
    │
    ├── runner.py ──► [AgentRunner](agents.md#agentrunner) (keystone loop)
    ├── tools.py ──► [ToolRegistry](agents.md#toolregistry)
    ├── trace.py ──► [AgentTrace](agents.md#agenttrace)
    └── validation.py ──► [validate_tools](agents.md#validate_tools) (safety gate)

pipeline.py ──► [run_proof](pipelines.md#run_proof), [assemble_proof](pipelines.md#assemble_proof)
agent_pipeline.py ──► [run_agent_proof](pipelines.md#run_agent_proof)
report.py ──► [ProofResult](api-reference.md#proofresult), [ProofStats](api-reference.md#proofstats)
stats.py ──► [wilson_interval](statistics.md), [mcnemar_exact](statistics.md), [pass_at_k](statistics.md)
cost.py ──► [price_completion](cost-pricing.md)
recommend.py ──► [resolve_candidate](tokenjam-integration.md#resolve_candidate)
version.py ──► [resolve_tokenjam_build](tokenjam-integration.md#version-stamping)
exec_sandbox.py ──► [run_python](benchmarks.md#run_python) (subprocess sandbox)
cli.py ──► Click CLI entry point
```

## Integration with TokenJam

The bench is a **black-box consumer** of the published `tokenjam` package. Three integration points:

1. **[Candidate Recommendation](tokenjam-integration.md#candidate-recommendation)** — [`recommend.py`](api-reference.md#recommendpy) imports [`tokenjam.core.optimize.DOWNGRADE_CANDIDATES`](https://github.com/HoomanDigital/tokenjam/blob/main/tokenjam/core/optimize/analyzers/model_downgrade.py)
2. **[Cost Pricing](tokenjam-integration.md#cost-pricing)** — [`cost.py`](api-reference.md#costpy) imports [`tokenjam.core.pricing.get_rates`](https://github.com/HoomanDigital/tokenjam/blob/main/tokenjam/core/pricing.py)
3. **[Version Stamp](tokenjam-integration.md#version-stamping)** — [`version.py`](api-reference.md#versionpy) reads [`importlib.metadata.version("tokenjam")`](https://github.com/HoomanDigital/tokenjam/blob/main/tokenjam/core/models.py)

See [TokenJam Integration](tokenjam-integration.md) for the full deep dive.

## Related Documentation

- [Pipelines](pipelines.md) — Detailed pipeline documentation
- [Models](models.md) — Model client adapters
- [Benchmarks](benchmarks.md) — Benchmark definitions
- [Agents](agents.md) — Multi-turn agent execution
- [Statistics](statistics.md) — Statistical methods
- [TokenJam Integration](tokenjam-integration.md) — How we consume TokenJam
- [TokenJam Architecture](https://github.com/HoomanDigital/tokenjam/blob/main/docs/architecture.md) — Main project architecture
