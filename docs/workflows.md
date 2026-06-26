# Production Workflow Benchmark Suite

Beyond academic datasets, TokenJam Bench can validate **real enterprise AI
workflows** — the things companies actually run in production — and answer the
only question that matters before a model swap:

> *If TokenJam recommends replacing my expensive model with a cheaper one, will
> my real production workflow still work?*

Production Workflows are **another benchmark family**, not a new pipeline. Each
suite is a dataset-driven, judge-scored `Benchmark` that flows through the exact
same machinery as HumanEval/GSM8K/judged:

- accuracy = **measured pass-rate** (DeepEval judge for text workloads;
  trace-scoring via `AgentRunner` for agentic ones),
- **Wilson CI + exact McNemar + pass@k** for the verdict (`assemble_proof`),
- **measured token cost** via TokenJam's cost engine,
- the same **report**, **dashboard**, and **historical DB** surfaces.

So a workflow proof is as rigorous — and as hedged — as every other benchmark:
it reports `no_significant_regression` / `significant_regression` /
`insufficient_evidence`, never a bare "safe".

## Suites

| Suite | Kind | Real evidence |
|---|---|---|
| `customer-support` | text · judge-scored (grounded support tickets) | ✅ real (DeepSeek reasoner→chat) |
| `enterprise-rag` | text · judge-scored (KB grounding / citations) | ✅ real (DeepSeek reasoner→chat) |
| `email-assistant` | text · judge-scored (reply/summarize/triage/extract) | ✅ real (DeepSeek reasoner→chat) |
| `research-assistant` | text · judge-scored (multi-doc synthesis) | ✅ real (DeepSeek reasoner→chat) |
| `n8n` | agentic · `AgentRunner` trace (tool order + safety gate) | ⏳ code shipped; real run needs a tool-capable provider¹ |
| `coding-workflow` | agentic · `AgentRunner` trace (edit/test/PR, safety) | ⏳ code shipped; real run needs a tool-capable provider¹ |

¹ The agentic suites drive a multi-turn tool loop, so **both** the original and
candidate must support function calling. `deepseek-reasoner` (R1) does not, so
the agentic suites can't be benchmarked `reasoner→chat` with a DeepSeek-only
key — they remain mock/agent-tested until a tool-capable provider key is
supplied (the pipeline itself is unchanged). Until then they carry **no** real
evidence and **do not appear on the production dashboard**.

The agentic suites reuse the Scenario Library's `AgentRunner` + safety gate; the
text suites reuse the `judged` DeepEval seam. Adding a suite is **pure data**:
drop a folder under `datasets/<suite>/` and register it in
`tjbench/workflows/__init__.py`.

## Real benchmarks only — no synthetic data on the dashboard

The production dashboard renders **measured runs only**. `scan_runs()` skips any
artifact marked `mock` (a `--mock`/dev run) or `demo` (a seeded fixture), so
every number on the Overview, Benchmarks, and Business Impact pages traces back
to a real provider-API execution and is reproducible by re-running the same
config. Reproduce the current dashboard with:

```bash
export DEEPSEEK_API_KEY=sk-...          # env only; never committed
./scripts/run_real_benchmarks.sh        # gsm8k, humaneval, judged + 4 workflows
tjbench serve                           # dashboard reads the fresh artifacts
```

`demo/seed_demo.py` still exists for **local UI development**, but its artifacts
are stamped `demo: true` and are invisible to the production dashboard. Mock
mode (`--mock`) likewise never reaches it.

## CLI

```bash
tjbench workflow customer-support --original anthropic:claude-opus-4-7 --mock --html
```

Options mirror `tjbench run`: `--original`, `--candidate`, `--limit`,
`--samples`, `--temperature`, `--mock`, `--max-tokens`, `--out`, `--html`,
`--json`. The judge backend is chosen via `TJBENCH_JUDGE` (offline `MockJudge`
by default; `TJBENCH_JUDGE=deepseek` for a real DeepEval judge):

```bash
TJBENCH_JUDGE=deepseek TJBENCH_JUDGE_METRIC=correctness \
tjbench workflow customer-support \
  --original deepseek:deepseek-reasoner --candidate deepseek:deepseek-chat --limit 16 --html
```

## Datasets

```
datasets/
  customer_support/
    tickets.json     # cases: question, expected_intent, expected_response,
    README.md        #        knowledge_context, difficulty, category, priority
```

Datasets are resolved from the repo root (or `TJBENCH_DATASETS_DIR`). See the
[Customer Support dataset README](../datasets/customer_support/README.md) for
the schema and the 16-ticket coverage (including safety-sensitive cases where
the reference reply deliberately refuses unsafe actions).

## Dashboard

Production Workflow suites appear as their own **Production Workflows** group on
the Benchmarks page (and in Leaderboards / Provider Comparison / Trends /
Regression Center), with the same evidence-rich cards as every benchmark:
pass-rate, cost saved, Wilson CI, McNemar, difficulty, coverage, latency saved,
and top failure modes.
