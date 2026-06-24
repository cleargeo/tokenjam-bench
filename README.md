# tokenjam-bench

A benchmarking + evaluation harness that **proves the effect of TokenJam's
recommendations on cost AND accuracy**, using executable benchmarks as objective
ground truth.

It answers the question TokenJam itself can't: *when TokenJam says "downsize
this model," does the cheaper model still get the work right — and how much does
it actually save?*

## How it's a proof of *TokenJam* (not a generic model comparison)

- The cheaper **candidate** model is the one TokenJam's own downsize analyzer
  would route to (`tokenjam.core.optimize.DOWNGRADE_CANDIDATES`).
- **Cost** is priced with TokenJam's own pricing table
  (`tokenjam.core.pricing.get_rates`) — same dollars TokenJam reports.
- **Accuracy** is the benchmark pass-rate against real test suites — a
  *measurement*, not a judgment.
- Every result is **stamped with the exact TokenJam version** under test.

```
benchmark tasks ─▶ run on ORIGINAL model ─▶ score (pass/fail) + cost
                ─▶ run on CANDIDATE model ─▶ score (pass/fail) + cost
                ─▶ proof: Δaccuracy (objective) + Δcost, stamped to tokenjam vX.Y.Z
```

## Documentation

Full documentation lives in [`docs/`](docs/):

| Doc | Description |
|-----|-------------|
| [docs/README.md](docs/README.md) | Documentation index with links to everything |
| [docs/overview.md](docs/overview.md) | What this project is and why it exists |
| [docs/architecture.md](docs/architecture.md) | System design, data flow, module relationships |
| [docs/quickstart.md](docs/quickstart.md) | Get running in 5 minutes |
| [docs/cli-reference.md](docs/cli-reference.md) | Complete `tjbench` command reference |
| [docs/pipelines.md](docs/pipelines.md) | Single-shot and agent proof pipelines |
| [docs/models.md](docs/models.md) | Model client adapters and protocols |
| [docs/benchmarks.md](docs/benchmarks.md) | Available benchmarks and scoring |
| [docs/agents.md](docs/agents.md) | Multi-turn agent execution framework |
| [docs/statistics.md](docs/statistics.md) | Statistical methods used for proof |
| [docs/cost-pricing.md](docs/cost-pricing.md) | How costs are computed |
| [docs/tokenjam-integration.md](docs/tokenjam-integration.md) | How we consume TokenJam |
| [docs/development.md](docs/development.md) | Contributing, testing, extending |
| [docs/api-reference.md](docs/api-reference.md) | Module-level API documentation |

## TokenJam changes every day — that's the design center

TokenJam is consumed as a **published package**, never vendored. To test a new
release:

```bash
make update-tokenjam     # pip install -U tokenjam ; prints the new version
tjbench version          # shows the exact tokenjam build proofs will stamp
tjbench run ...          # every artifact records tokenjam_version
```

Because each artifact in `results/` carries `tokenjam_version`, you can re-run
the same benchmark across releases and diff the savings/accuracy — catching the
day a TokenJam change moves the numbers.

## Quickstart (offline, no keys)

```bash
pip install -e ".[dev]"
tjbench run --benchmark samples --original anthropic:claude-opus-4-7 --mock
```

`--mock` runs the whole pipeline deterministically with no provider SDKs, no
keys, no spend (numbers are illustrative — for plumbing, not proofs).

## Real proof (live, multi-provider)

```bash
pip install -e ".[providers,datasets]"
export ANTHROPIC_API_KEY=...      # and/or OPENAI_API_KEY / GEMINI_API_KEY
tjbench run --benchmark humaneval --original anthropic:claude-opus-4-7 --limit 50
```

This runs HumanEval on Opus and on the model TokenJam recommends downsizing to
(Haiku), executes both against the hidden tests, and reports the cost saved and
the pass-rate delta — stamped to the installed TokenJam version.

## Benchmarks

| name | ground truth | needs |
|---|---|---|
| `samples` | tiny built-in code + math tasks | nothing (offline) |
| `humaneval` | unit-test pass/fail | `[datasets]` |
| `gsm8k` | numeric exact-match | `[datasets]` |

## Honesty

Accuracy is the pass-rate on the chosen suite — never a general "quality
preserved" claim. Reports record `n`, flag `--mock` runs as illustrative, and
flag when cost fell back to TokenJam's default placeholder rates. Executing
model-generated code (HumanEval) happens in a timed subprocess; run only trusted
benchmark suites on a machine you control.

## Related Projects

- **[TokenJam](https://github.com/HoomanDigital/tokenjam)** — The main cost-optimization and observability platform
- **[TokenJam Docs](https://github.com/HoomanDigital/tokenjam/tree/main/docs)** — TokenJam's own documentation
- **[TokenJam Python SDK](https://github.com/HoomanDigital/tokenjam/tree/main/tokenjam/sdk)** — SDK for instrumenting agents
