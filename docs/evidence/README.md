# First evidence-backed validation report

Live runs against **DeepSeek** (model-under-test) with **DeepEval + DeepSeek** as
the judge. Cost is measured from real token usage at real DeepSeek rates (via a
TokenJam pricing override). Reproduce with [`docs/proof-runbook.md`](../proof-runbook.md).

Downsize tested: `deepseek-reasoner` (premium) → `deepseek-chat` (cheaper).

## HumanEval — executable pass/fail (n=20)

| | Original (reasoner) | Candidate (chat) |
|---|---|---|
| Pass rate (95% Wilson CI) | 19/20 (76–99%) | **20/20 (84–100%)** |
| Cost (measured, real rates) | $0.01500 | **$0.00366** |

- **Δ cost: −75.6%** (measured)
- **Δ pass-rate: +5.0pp** (95% CI −4.5 … +14.6) — the cheaper model did *not* regress
- **McNemar:** b=0 broken, c=1 fixed, p=1.000
- **Verdict: `no_significant_regression`** (n=20 ≥ 10)

> Evidence-based reading: on this 20-task HumanEval sample, downsizing
> reasoner→chat cut measured cost **~76%** with **no statistically significant
> accuracy regression** (the candidate actually passed one more task). Not a
> general quality claim — pass-rate on this suite only.

Artifacts: [`humaneval_deepseek_reasoner_to_chat.html`](humaneval_deepseek_reasoner_to_chat.html) · `.json`

## Judged — DeepEval correctness judge via DeepSeek (n=5)

The judge (DeepEval `GEval` correctness, judged by `deepseek-chat`) graded each
answer 0–1 against the gold answer. Real per-task scores, e.g. capital=1.00,
retry-summary=1.00, refund=0.00, shipping=0.20.

- Both models 2/5 · **Δ cost −75.3%** · **McNemar p=1.000**
- **Verdict: `insufficient_evidence`** (n=5 is too small — by design)

Artifacts: [`judged_deepseek_correctness.html`](judged_deepseek_correctness.html) · `.json`

## Not run live here

- **SWE-Bench Lite** — supported (DeepSeek tool-calling is wired), but it's a
  heavier multi-turn agent benchmark needing the SWE-bench dataset and repo
  context; see the runbook. Run with `tjbench agent --benchmark swe-bench-lite`.

## Honesty notes

- Accuracy = pass-rate / judge score on the chosen suite, never a general
  "quality preserved" claim.
- HumanEval is in pretraining data → treat pass-rate as an upper bound.
- Larger n tightens the CI; n=20 already clears the significance gate.
