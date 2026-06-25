# Changelog

## 0.1.0 (2025-06-24)

### Added

#### Agent Evaluation Framework
- **AgentRunner** — Multi-turn agent execution loop with max-turns guard
- **ToolRegistry** — Register and execute tools with JSON-schema advertisement
- **AgentTrace** — Observable record of every turn, tool call, and result
- **Safety Gate** — `validate_tools()` catches forbidden tools even with correct answers
- **ToolValidation** — Reports expected tools, ordering, safety, and error rate

#### Agent Benchmarks
- **sample-agent** — 3 offline tool-use tasks with safety gate validation
- **swe-bench-lite** — 300 real GitHub issue bug fixes from SWE-Bench Lite dataset

#### SWE-Bench Tools
- **view** — Read file contents with line numbers
- **view_range** — Read specific line range
- **str_replace** — Exact-match string replacement (must match exactly once)
- **create** — Create new file
- **insert** — Insert text after specific line
- **bash** — Run shell commands with timeout
- **Path traversal blocking** — Prevents escaping workspace
- **Exact-match enforcement** — Prevents accidental mass-replace

#### Model Clients
- **AnthropicAgentClient** — Live tool-calling client for Anthropic
- **MockAgentClient** — Deterministic offline tool-calling client
- **ToolCallingClient** protocol — Multi-turn chat with tool use

#### Pipelines
- **Agent Proof Pipeline** — `run_agent_proof()` for multi-turn agent evaluation
- **Token summation** — Aggregates token usage across all turns for pricing
- **Tool validation scoring** — Safety gate + ordering + expected tools

#### Documentation (14 files)
- docs/README.md — Master documentation index
- docs/overview.md — Project overview and design principles
- docs/architecture.md — System design, data flow, module relationships
- docs/quickstart.md — 5-minute quickstart guide
- docs/cli-reference.md — Complete `tjbench` command reference
- docs/pipelines.md — Single-shot and agent proof pipeline deep dive
- docs/models.md — Model client adapters and protocols
- docs/benchmarks.md — Available benchmarks and scoring
- docs/agents.md — Multi-turn agent execution framework
- docs/statistics.md — Statistical methods used for proof
- docs/cost-pricing.md — How costs are computed
- docs/tokenjam-integration.md — How we consume TokenJam
- docs/development.md — Contributing, testing, extending
- docs/api-reference.md — Module-level API documentation
- docs/swe-bench-lite.md — SWE-Bench Lite integration guide
- docs/tests.md — Complete test suite inventory

#### Tests
- 55 total tests (20 new for SWE-Bench Lite)
- Mock scoring tests for SWE-Bench Lite
- Tool operation tests (view, replace, create, insert, bash)
- Path traversal safety tests
- Patch parsing tests

### Design Principles

- **Black-box consumer** of TokenJam — imports as pip dependency, never vendored
- **Offline-first** — All tests run without API keys using mock clients
- **Objective ground truth** — Code execution and exact-match scoring, not LLM-as-judge
- **Statistical honesty** — Wilson CIs, McNemar exact tests, never claim significance on small samples
- **Safety-first** — Agent benchmarks include safety gate for dangerous tool calls

### Integration Points

| Feature | TokenJam API | Module |
|---------|-------------|--------|
| Candidate recommendation | `tokenjam.core.optimize.DOWNGRADE_CANDIDATES` | `recommend.py` |
| Cost pricing | `tokenjam.core.pricing.get_rates` | `cost.py` |
| Version stamp | `importlib.metadata.version("tokenjam")` | `version.py` |

---

*This changelog documents the initial release of tokenjam-bench as an agent evaluation framework.*
