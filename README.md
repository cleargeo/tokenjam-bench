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
