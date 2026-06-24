# Quickstart

## Prerequisites

- Python 3.10+
- Git

## Install

```bash
cd tokenjam-bench
pip install -e ".[dev]"
```

For live provider support (optional):
```bash
pip install -e ".[providers,datasets]"
```

## Verify Installation

```bash
tjbench version
```

This shows both the bench version and the installed TokenJam version that proofs will be stamped with.

## Run Your First Proof (Offline, No Keys)

The fastest way to see the pipeline in action is with `--mock`, which runs deterministically with no provider SDKs, no keys, and no spend:

```bash
tjbench run --benchmark samples --original anthropic:claude-opus-4-7 --mock
```

Output:
```
┌─────────────┬──────────┬──────────┬──────────┬──────────┐
│ Metric      │ Original │ Candidate│ Delta    │ Verdict  │
├─────────────┼──────────┼──────────┼──────────┼──────────┤
│ Pass rate   │ 100%     │ 100%     │ 0pp      │ preserved│
│ Cost        │ $0.42    │ $0.08    │ -$0.34   │ cheaper  │
│ Tokens out  │ 1,024    │ 1,024    │ 0        │ ok       │
└─────────────┴──────────┴──────────┴──────────┴──────────┘
```

> **Note**: `--mock` numbers are illustrative — for plumbing verification, not actual proofs. See [Honesty](#honesty) below.

## Check What TokenJam Recommends

```bash
tjbench recommend --original anthropic:claude-opus-4-7
```

This queries [`tokenjam.core.optimize.DOWNGRADE_CANDIDATES`](https://github.com/HoomanDigital/tokenjam/blob/main/tokenjam/core/optimize/analyzers/model_downgrade.py) and shows the cheaper model TokenJam would route to.

## Run a Real Proof (Live, Requires API Key)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
tjbench run --benchmark humaneval --original anthropic:claude-opus-4-7 --limit 50
```

This runs HumanEval on Opus and on TokenJam's recommended downgrade (Haiku), executes both against hidden tests, and reports the cost saved and the pass-rate delta — stamped to the installed TokenJam version.

## Run an Agent Proof (Multi-Turn, Tool-Use)

```bash
tjbench agent --benchmark sample-agent --original anthropic:claude-opus-4-7 --mock
```

This exercises the [agent pipeline](pipelines.md#agent-proof-pipeline) with tool-calling, safety validation, and multi-turn scoring.

## Update TokenJam and Re-Run

```bash
make update-tokenjam     # pip install -U tokenjam
tjbench version          # confirm new version
tjbench run --benchmark samples --original anthropic:claude-opus-4-7 --mock
```

Compare `results/` artifacts across versions to see how TokenJam changes affect recommendations.

## Available Benchmarks

| name | ground truth | needs | docs |
|------|-------------|-------|------|
| `samples` | tiny built-in code + math tasks | nothing (offline) | [Benchmarks](benchmarks.md#samples) |
| `humaneval` | unit-test pass/fail | `[datasets]` | [Benchmarks](benchmarks.md#humaneval) |
| `gsm8k` | numeric exact-match | `[datasets]` | [Benchmarks](benchmarks.md#gsm8k) |
| `sample-agent` | tool-use + safety validation | nothing (offline) | [Benchmarks](benchmarks.md#sample-agent) |

## Honesty

- **Mock runs** (`--mock`) are flagged as illustrative in reports
- **Small samples** (`--limit < 30`) cannot reach statistical significance — reports will say `insufficient_evidence`
- **Cost fallback**: If TokenJam's pricing table lacks a model, we use `$0.50/$2.00` per MTok defaults and flag it
- **Accuracy is suite-specific**: Pass-rate on HumanEval ≠ general coding ability

## Next Steps

- [CLI Reference](cli-reference.md) — Full command documentation
- [Architecture](architecture.md) — System design
- [Pipelines](pipelines.md) — How proofs work under the hood
- [Development Guide](development.md) — Contributing and extending

## Related TokenJam Docs

- [TokenJam Installation](https://github.com/HoomanDigital/tokenjam/blob/main/docs/installation.md)
- [TokenJam Python SDK](https://github.com/HoomanDigital/tokenjam/blob/main/docs/python-sdk.md)
- [TokenJam Quickstart](https://github.com/HoomanDigital/tokenjam/blob/main/README.md)
