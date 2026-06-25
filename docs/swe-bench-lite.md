# SWE-Bench Lite Integration

## Overview

The SWE-Bench Lite integration is the **first real agent benchmark** that exercises the [AgentRunner](agents.md#agentrunner) on actual capability (bug fixing) rather than just plumbing. It transforms tokenjam-bench from an "agent evaluation framework proof-of-concept" into an "agent benchmark platform."

## What is SWE-Bench Lite?

[SWE-Bench](https://github.com/princeton-nlp/SWE-bench) is a benchmark for evaluating large language models on real-world software engineering tasks. **SWE-Bench Lite** is a curated subset of 300 tasks from 23 popular Python repositories.

Each task consists of:
- A **real GitHub issue** with a bug report
- A **repository** (e.g., `astropy/astropy`, `django/django`)
- A **base commit** (the state before the fix)
- A **patch** (the actual fix that was merged)
- **Test patches** (tests that verify the fix)
- `FAIL_TO_PASS` tests (must pass after fix = bug is fixed)
- `PASS_TO_PASS` tests (must still pass after fix = no regressions)

## How It Works in tokenjam-bench

### Architecture

```
SWE-Bench Lite Dataset (300 tasks)
    │
    ├──▶ Problem Statement → Agent Prompt
    │
    ├──▶ File Paths from Patch → Context for Agent
    │
    └──▶ FAIL_TO_PASS / PASS_TO_PASS → Scoring Criteria
    │
    ▼
AgentRunner + SWE-Bench Tools
    │
    ├──▶ view → Read files to understand bug
    ├──▶ str_replace → Edit code to fix bug
    ├──▶ create → Create new files if needed
    ├──▶ insert → Insert code at specific lines
    └──▶ bash → Run tests to verify fix
    │
    ▼
Scoring
    │
    ├──▶ Did agent view files? (exploration)
    ├──▶ Did agent make edits? (action)
    ├──▶ Did agent run tests? (verification)
    └──▶ (Full impl: FAIL_TO_PASS pass + PASS_TO_PASS pass)
```

### Mock Mode

In `--mock` mode, the benchmark uses deterministic scoring without downloading the dataset or cloning repos:

```bash
tjbench agent --benchmark swe-bench-lite --original anthropic:claude-opus-4-7 --mock --limit 5
```

**Mock scoring checks:**
1. Did the agent call `view`? (explored the codebase)
2. Did the agent call `str_replace`, `create`, or `insert`? (made edits)
3. Did the agent call `bash`? (ran tests)

All three must be true for a passing mock score. This tests whether the agent follows the expected developer workflow.

### Live Mode (Full Implementation)

In live mode, the benchmark would:

1. **Clone the repository** at the base commit
2. **Set up the environment** (install dependencies)
3. **Run the agent** with SWE-Bench tools operating on the real repo
4. **Apply agent's edits** to the working directory
5. **Run FAIL_TO_PASS tests** — must pass (bug is fixed)
6. **Run PASS_TO_PASS tests** — must pass (no regressions)
7. **Score** based on test results

```bash
tjbench agent --benchmark swe-bench-lite --original anthropic:claude-opus-4-7 --limit 10
```

## SWE-Bench Tools

See [Agents - SWE-Bench Tools](agents.md#swe-bench-tools) for detailed documentation of each tool.

### Tool Summary

| Tool | Purpose | Dangerous |
|------|---------|-----------|
| `view` | Read file with line numbers | No |
| `view_range` | Read specific line range | No |
| `str_replace` | Exact-match string replacement | No |
| `create` | Create new file | No |
| `insert` | Insert after specific line | No |
| `bash` | Run shell commands | Yes |

### Safety Features

- **Path traversal blocked**: `../../../etc/passwd` → error
- **Exact-match enforcement**: Multiple occurrences of `old_str` → error
- **Timeout protection**: Bash commands timeout after configurable seconds
- **Dangerous flag**: `bash` marked for safety gate validation

## Benchmark Implementation

### `SWEBenchLiteBenchmark`

[`benchmarks/swe_bench_lite.py`](../benchmarks/swe_bench_lite.py)

```python
class SWEBenchLiteBenchmark(AgentBenchmark):
    def __init__(self, limit=None, mock=False): ...
    def tools(self) -> ToolRegistry: ...
    def tasks(self, limit=None) -> list[SWEBenchTask]: ...
    def score(self, task: SWEBenchTask, trace: AgentTrace) -> ScoreResult: ...
```

### `SWEBenchTask`

Extends `AgentTask` with SWE-Bench-specific fields:

```python
@dataclass
class SWEBenchTask(AgentTask):
    repo: str                    # e.g., "astropy/astropy"
    base_commit: str             # Git commit hash
    test_patch: str              # Test diff
    fail_to_pass: list[str]     # Test names that must pass
    pass_to_pass: list[str]      # Test names that must still pass
    problem_statement: str       # GitHub issue description
    hints_text: str              # Additional hints
```

### Prompt Construction

The prompt is built from the problem statement and includes:
- Repository name
- Issue ID
- Problem description
- Files that may need changes (extracted from patch)
- Instructions to explore, fix, and test

## Scoring

### Mock Scoring

Deterministic scoring based on tool usage:

| Condition | Result |
|-----------|--------|
| Used `view` + edit tool + `bash` | PASS |
| Missing `view` | FAIL (didn't explore) |
| Missing edit tool | FAIL (didn't fix) |
| Missing `bash` | FAIL (didn't verify) |

### Live Scoring (Target)

```python
def score(task: SWEBenchTask, trace: AgentTrace) -> ScoreResult:
    # 1. Apply agent's edits to repo
    # 2. Run FAIL_TO_PASS tests
    # 3. Run PASS_TO_PASS tests
    # 4. Return pass/fail based on test results
```

| Condition | Result |
|-----------|--------|
| All FAIL_TO_PASS pass + All PASS_TO_PASS pass | PASS |
| Any FAIL_TO_PASS fails | FAIL (bug not fixed) |
| Any PASS_TO_PASS fails | FAIL (regression) |

## Why This Matters

### Before SWE-Bench Lite

```text
Agent Benchmarks:
├── sample-agent (3 toy tasks)
│   └── lookup_and_compute
│   └── multi_step
│   └── safety_check
└── (nothing else)
```

The framework could evaluate tool-use, but only on artificial tasks.

### After SWE-Bench Lite

```text
Agent Benchmarks:
├── sample-agent (3 toy tasks — plumbing verification)
└── swe-bench-lite (300 real tasks — capability evaluation)
    └── Real GitHub issues
    └── Real repositories
    └── Real bug fixes
    └── Real test verification
```

Now the framework can evaluate whether a cheaper model can **actually fix real bugs** in **real codebases**.

## Roadmap: Full SWE-Bench Integration

### Current (Lite)
- Mock scoring based on tool usage
- Dataset loading for prompt construction
- Tool implementations for file editing

### Next Steps
1. **Repository cloning** — Clone repos at base commit
2. **Environment setup** — Install dependencies
3. **Test execution** — Run FAIL_TO_PASS and PASS_TO_PASS tests
4. **Patch application** — Apply agent's edits as a git patch
5. **Result reporting** — Full test output in ProofResult

## Related Documentation

- [Agents](agents.md) — AgentRunner, ToolRegistry, safety gate
- [Benchmarks](benchmarks.md) — Benchmark protocols and registry
- [Pipelines](pipelines.md) — Agent proof pipeline
- [CLI Reference](cli-reference.md) — `tjbench agent` command
- [SWE-Bench Official Repo](https://github.com/princeton-nlp/SWE-bench) — Original benchmark
