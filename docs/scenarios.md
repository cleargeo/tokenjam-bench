# Real Scenario Library (P1)

Benchmarks like HumanEval prove a model's raw *capability*. The Scenario Library
provides **production-shaped agent workloads** — the kind of multi-step,
tool-using work TokenJam's users actually run — so you can benchmark a downsize
on representative tasks even before you have telemetry to replay (P0).

Each suite is its own `AgentBenchmark`, scored on the whole trace: right tools,
right order, right answer, **and no catastrophic action** (every suite ships a
destructive tool that trips the safety gate).

## Suites

| Suite | Style | Tools (excerpt) | Dangerous (safety gate) |
|---|---|---|---|
| `coding-assistant` | Claude Code / coding assistant | read_file, search_code, edit_file, run_tests, git_commit | force_push, delete_branch |
| `rag-support` | RAG + customer support | search_kb, get_article, check_order | issue_refund, cancel_order |
| `research-agent` | LangGraph / research | web_search, fetch_page, summarize, save_notes | post_publicly, send_email |
| `browser-agent` | browser automation | navigate, read_page, click, fill_form, extract_data | submit_payment, confirm_purchase |

List them:

```bash
python3 run.py scenarios          # or: tjbench scenarios
```

## Run a suite

```bash
# offline (no keys/spend) — demonstrates the framework + safety gate
python3 run.py agent --benchmark coding-assistant \
  --original anthropic:claude-opus-4-7 --mock --html

# live downsize proof
python3 run.py agent --benchmark rag-support \
  --original deepseek:deepseek-reasoner --candidate deepseek:deepseek-chat --html
```

Each run produces the same `ProofResult` artifact as every other benchmark, so it
appears in the **dashboard** (`run.py serve`), renders an **HTML report**, and
participates in **`tjbench matrix`** — no extra wiring.

## Add your own suite (extensibility)

A suite is data. In `benchmarks/scenario_suites.py`:

```python
SuiteSpec(
    name="my-suite",
    tools={
        "do_thing":   ("Does the thing.", "did it"),
        "danger":     ("Destructive.", "boom", True),   # dangerous=True
    },
    scenarios=[
        Scenario("my/task", "Do the thing safely.",
                 plan=["do_thing"], answer="Done.", expected_answer="Done",
                 unsafe_tool="danger"),
    ],
).build()
```

Then register the name in `benchmarks/__init__.py` (`_SCENARIO_SUITE_NAMES`).
Offline runs work immediately via the plan-driven mock; live runs strip the
`# plan:` scaffolding so the model never sees it.

## Honesty

Scenario scores are pass/fail on the suite's own criteria (tools + order + answer
+ safety), never a general "quality" claim. The mock suites are deterministic
fixtures for the framework + safety gate; a *real* downsize proof uses live
models judged on these same criteria.
