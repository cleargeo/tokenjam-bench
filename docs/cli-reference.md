# CLI Reference

The CLI is `tjbench` (defined in `pyproject.toml` as `cli:cli`).

## Global Options

| Option | Description |
|--------|-------------|
| `--json` | Output raw JSON instead of Rich tables |
| `--out DIR` | Write result artifact to directory (default: `results/`) |

---

## `tjbench version`

Shows bench version + resolved TokenJam version under test.

```bash
tjbench version
```

Output:
```
tokenjam-bench 0.1.0
tokenjam 0.4.2  (/Users/.../site-packages/tokenjam)
```

The TokenJam version and location are stamped on every proof artifact.

---

## `tjbench recommend`

Shows what TokenJam would downsize the model to.

```bash
tjbench recommend --original SPEC
```

| Option | Required | Description |
|--------|----------|-------------|
| `--original` | Yes | Provider:model spec, e.g. `anthropic:claude-opus-4-7` |

Example:
```bash
tjbench recommend --original anthropic:claude-opus-4-7
```

This queries [`tokenjam.core.optimize.DOWNGRADE_CANDIDATES`](https://github.com/HoomanDigital/tokenjam/blob/main/tokenjam/core/optimize/analyzers/model_downgrade.py) and prints the recommended cheaper model.

---

## `tjbench run`

Single-shot proof: run original vs candidate, score, price, report.

```bash
tjbench run [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--benchmark` | `samples` | Benchmark to run: `samples`, `humaneval`, `gsm8k` |
| `--original` | (required) | Original model spec, e.g. `anthropic:claude-opus-4-7` |
| `--candidate` | (TokenJam) | Override candidate model (bypasses TokenJam recommendation) |
| `--limit` | all | Limit number of tasks |
| `--samples` | 1 | Run each task N times (for pass@k) |
| `--temperature` | 0.0 | Sampling temperature |
| `--mock` | false | Offline deterministic run (no SDKs, no keys, no spend) |
| `--mock-candidate-accuracy` | 1.0 | Mock candidate accuracy (0.0-1.0) |
| `--max-tokens` | model default | Max output tokens |
| `--out` | `results/` | Output directory for JSON artifacts |
| `--json` | false | Print JSON instead of Rich table |

### Examples

**Offline smoke test:**
```bash
tjbench run --benchmark samples --original anthropic:claude-opus-4-7 --mock
```

**Live HumanEval with 50 tasks:**
```bash
tjbench run --benchmark humaneval --original anthropic:claude-opus-4-7 --limit 50
```

**Force a specific candidate:**
```bash
tjbench run --benchmark gsm8k --original anthropic:claude-opus-4-7 --candidate anthropic:claude-haiku-4-5
```

**Multi-sample for pass@k:**
```bash
tjbench run --benchmark samples --original anthropic:claude-opus-4-7 --mock --samples 5
```

---

## `tjbench agent`

Multi-turn agent proof: tool use + safety validation.

```bash
tjbench agent [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--benchmark` | `sample-agent` | Agent benchmark: `sample-agent`, `swe-bench-lite` |
| `--original` | (required) | Original model spec |
| `--candidate` | (TokenJam) | Override candidate model |
| `--limit` | all | Limit number of tasks |
| `--samples` | 1 | Run each task N times |
| `--temperature` | 0.0 | Sampling temperature |
| `--max-turns` | 10 | Max agent turns before stopping |
| `--max-tokens` | model default | Max output tokens per turn |
| `--mock` | false | Offline deterministic run |
| `--candidate-behavior` | `ok` | Mock behavior: `ok`, `wrong`, `unsafe` |
| `--out` | `results/` | Output directory |
| `--json` | false | Print JSON instead of Rich table |

### Candidate Behavior Modes (Mock Only)

| Mode | Effect |
|------|--------|
| `ok` | Correct answer, correct tools |
| `wrong` | Wrong answer (tests accuracy regression) |
| `unsafe` | Calls dangerous tool (tests safety gate) |

### Examples

**Offline agent smoke test:**
```bash
tjbench agent --benchmark sample-agent --original anthropic:claude-opus-4-7 --mock
```

**Test safety gate with mock:**
```bash
tjbench agent --benchmark sample-agent --original anthropic:claude-opus-4-7 --mock --candidate-behavior unsafe
```

**SWE-Bench Lite agent proof:**
```bash
tjbench agent --benchmark swe-bench-lite --original anthropic:claude-opus-4-7 --mock --limit 5
```

---

## Key Flags Explained

### `--mock`

Runs the entire pipeline deterministically with no provider SDKs, no API keys, and no spend. Mock clients read `# task_key:` markers embedded in prompts and return predetermined responses.

- **Use for**: CI, testing, plumbing verification
- **Not for**: Actual proofs (numbers are illustrative)
- **Flagged in reports**: Every mock run is marked `mock: true`

### `--candidate-behavior` (Agent Only)

Simulates different candidate behaviors in mock mode:
- `ok`: Correct answer and correct tool calls
- `wrong`: Wrong final answer (exercises accuracy regression detection)
- `unsafe`: Calls a forbidden/dangerous tool (exercises safety gate)

### `--samples`

Runs each task N times. Used for:
- Pass@k estimation (how many of k attempts pass)
- Variance reduction on small benchmarks
- Statistical power

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (proof completed, report generated) |
| 1 | General error (bad args, missing keys, scoring failure) |
| 2 | No candidate found (TokenJam has no recommendation for this model) |

---

## Related Documentation

- [Pipelines](pipelines.md) — How `run` and `agent` work under the hood
- [Benchmarks](benchmarks.md) — Available benchmarks
- [Agents](agents.md) — Multi-turn agent execution
- [Statistics](statistics.md) — How proof stats are computed
- [TokenJam CLI Reference](https://github.com/HoomanDigital/tokenjam/blob/main/docs/cli-reference.md) — The main `tj` CLI
