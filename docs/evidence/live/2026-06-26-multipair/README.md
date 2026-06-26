# Broad evidence — TokenJam downsize pairs across Anthropic + OpenAI (2026-06-26)

Real, honestly-priced proof of TokenJam's **own** downsize recommendations on
both **cost** and **accuracy**. Every number here traces to a measured API run;
there is no mock, seeded, or synthetic data (`scan_runs()` skips any artifact
marked `mock`/`demo`, and every artifact below is `mock=false`,
`priced_with_defaults=false`).

- **TokenJam version:** `0.5.2` (stamped on every artifact)
- **Pairs:** TokenJam's real `DOWNGRADE_CANDIDATES` map, Anthropic + OpenAI
- **Pricing:** real list rates from `tokenjam.core.pricing.get_rates` for every
  model used — verified non-placeholder before running (the `$0.50/$2.00`
  default never appears; `priced_with_defaults` is `false` everywhere)
- **Objective suites:** `gsm8k` and `humaneval` at **n=50** — executable,
  pass/fail against ground truth. **This is the load-bearing evidence.**
- **Secondary suites:** `judged` + text workflows (`customer-support`,
  `enterprise-rag`, `email-assistant`, `research-assistant`), GEval correctness
  judged by `gpt-4o`. **Judge-scored, NOT objective** — directional only.

## What "pass-rate" means here (read this first)

Accuracy is **pass-rate on the specific suite**, never a general "quality
preserved" claim. A pair that holds on `gsm8k` but drops on `humaneval` has
*not* been shown to be safe in general — it has been shown to regress on code.
Confidence is the **Wilson CI + McNemar p-value**, not a single "safe %".

`HumanEval` and `GSM8K` are public datasets present in pretraining corpora, so
absolute pass-rates may be **contamination-inflated — treat them as an upper
bound**. The *delta* between two models on the same fixed task set (what
regression detection actually uses) is far more robust to this than the
absolute numbers.

## Objective evidence — executable pass/fail, n=50 per cell

Pairs are exactly what TokenJam's downgrade analyzer routes to (premium →
`claude-haiku-4-5` for Anthropic; the mapped mini/`o4-mini` for OpenAI).

| Pair (orig → cand) | Suite | Pass (orig → cand) | Δ cost | Δ pass-rate | 95% CI (pp) | McNemar p | Verdict |
|---|---|---|---|---|---|---|---|
| `claude-opus-4-7` → `claude-haiku-4-5` | gsm8k | 49/50 → 48/50 | **−59.2%** | −2.0pp | [−5.9, +1.9] | 1.000 | no_significant_regression |
| `claude-opus-4-7` → `claude-haiku-4-5` | humaneval | 45/50 → 28/50 | **−81.6%** | −34.0pp | [−47.1, −20.9] | **0.000** | **significant_regression** |
| `claude-sonnet-4-6` → `claude-haiku-4-5` | gsm8k | 49/50 → 48/50 | **−68.2%** | −2.0pp | [−5.9, +1.9] | 1.000 | no_significant_regression |
| `claude-sonnet-4-6` → `claude-haiku-4-5` | humaneval | 47/50 → 28/50 | **−71.6%** | −38.0pp | [−51.5, −24.6] | **0.000** | **significant_regression** |
| `gpt-4o` → `gpt-4o-mini` | gsm8k | 48/50 → 47/50 | **−92.9%** | −2.0pp | [−5.9, +1.9] | 1.000 | no_significant_regression |
| `gpt-4o` → `gpt-4o-mini` | humaneval | 45/50 → 40/50 | **−94.9%** | −10.0pp | [−21.4, +1.4] | 0.180 | no_significant_regression |
| `o3` → `o4-mini` | gsm8k | 49/50 → 50/50 | **−87.5%** | +2.0pp | [−1.9, +5.9] | 1.000 | no_significant_regression |
| `o3` → `o4-mini` | humaneval | 44/50 → 42/50 | **−87.6%** | −4.0pp | [−13.5, +5.5] | 0.688 | no_significant_regression |

### Honest reading of the objective table

- **Cost savings are large and real** (−59% … −95%), measured from actual token
  usage at real list rates — not estimated.
- **The downsize is suite-dependent, and the harness proves it.** On `gsm8k`
  (grade-school math) every pair holds with no significant regression. On
  `humaneval` (code), the Anthropic **`→ haiku-4-5`** downsize **significantly
  regresses** (McNemar p≈0.000, CI excludes 0) — 45→28 and 47→28 passing. The
  OpenAI downsizes (`gpt-4o-mini`, `o4-mini`) hold up on code: their deltas are
  small and their CIs cross zero (no significant regression at n=50).
- **This is the point of the harness.** TokenJam's recommendation is a good
  default for math-style workloads but, for the Anthropic→haiku pairs, would
  trade a real measured accuracy drop on code for the cost saving. A buyer
  should keep the premium model for code-heavy work and downsize the rest.
- Verdicts are `no_significant_regression` / `significant_regression` /
  `insufficient_evidence` — never "SAFE".

## Secondary evidence — judge-scored, NOT objective

> ⚠️ These suites are scored by an LLM judge (DeepEval `GEval` correctness,
> `gpt-4o`, threshold 0.5), not by executing ground truth. They are a
> **directional signal only** and must not be read as objective pass/fail.
> Note the judge (`gpt-4o`) shares a family with the OpenAI candidate
> (`gpt-4o-mini`); the per-task *delta* is still the meaningful quantity, but
> absolute judge scores carry that caveat. Sample sizes are small by design
> (`judged` n=5; workflows n=12), so most verdicts are `insufficient_evidence`
> or hinge on a couple of tasks.

| Pair (orig → cand) | Suite | Judge-pass (orig → cand) | Δ cost | Δ judge-pass | 95% CI (pp) | McNemar p | Verdict |
|---|---|---|---|---|---|---|---|
| `claude-opus-4-7` → `claude-haiku-4-5` | judged | 5/5 → 5/5 | −88.8% | 0.0pp | [0, 0] | 1.000 | insufficient_evidence |
| `claude-opus-4-7` → `claude-haiku-4-5` | customer-support | 12/12 → 11/12 | −92.0% | −8.3pp | [−24.0, +7.3] | 1.000 | no_significant_regression |
| `claude-opus-4-7` → `claude-haiku-4-5` | email-assistant | 12/12 → 12/12 | −84.9% | 0.0pp | [0, 0] | 1.000 | no_significant_regression |
| `claude-opus-4-7` → `claude-haiku-4-5` | enterprise-rag | 5/12 → 12/12 | −90.6% | +58.3pp | [+30.4, +86.2] | 0.016 | no_significant_regression |
| `claude-opus-4-7` → `claude-haiku-4-5` | research-assistant | 9/12 → 7/12 | −91.7% | −16.7pp | [−37.8, +4.4] | 0.500 | no_significant_regression |
| `gpt-4o` → `gpt-4o-mini` | judged | 2/5 → 3/5 | −93.0% | +20.0pp | [−15.1, +55.1] | 1.000 | insufficient_evidence |
| `gpt-4o` → `gpt-4o-mini` | customer-support | 7/12 → 8/12 | −94.1% | +8.3pp | [−27.9, +44.6] | 1.000 | no_significant_regression |
| `gpt-4o` → `gpt-4o-mini` | email-assistant | 12/12 → 11/12 | −94.2% | −8.3pp | [−24.0, +7.3] | 1.000 | no_significant_regression |
| `gpt-4o` → `gpt-4o-mini` | enterprise-rag | 1/12 → 2/12 | −91.1% | +8.3pp | [−7.3, +24.0] | 1.000 | no_significant_regression |
| `gpt-4o` → `gpt-4o-mini` | research-assistant | 1/12 → 1/12 | −94.1% | 0.0pp | [0, 0] | 1.000 | no_significant_regression |

Why these are *secondary*, concretely: `enterprise-rag` shows the candidate
**out-scoring** the original (opus→haiku 5/12 → 12/12; gpt-4o→mini 1/12 → 2/12),
and several workflows sit at very low absolute scores (e.g. `research-assistant`
1/12 for the OpenAI pair). Under strict GEval correctness, small wording/format
differences flip a judgment, and n=5–12 is too small to separate signal from
judge variance. Read these as "no obvious regression jumped out," not as proof.

## Reproduce

```bash
pip install -e ".[dev,providers,judge,datasets]"
export ANTHROPIC_API_KEY=...   # env only; never committed
export OPENAI_API_KEY=...

# Objective (executable) suites, n=50, real pricing:
GROUP=anthropic ./scripts/run_multipair_evidence.sh
GROUP=openai    ./scripts/run_multipair_evidence.sh
# Secondary judged/workflow suites (gpt-4o judge):
GROUP=judged    ./scripts/run_multipair_evidence.sh

tjbench serve --dir docs/evidence/live/2026-06-26-multipair   # live dashboard
```

HumanEval executes model-generated code in a timed subprocess
(`exec_sandbox.py`) — run only on a machine you control.

## Run totals (this evidence set)

- **Artifacts:** 18 (8 objective + 10 secondary), all `mock=false`,
  `priced_with_defaults=false`, `tokenjam_version=0.5.2`
- **Tasks scored:** 506
- **Measured spend:** **$3.19 → $0.46 (−85.5%)** at real list rates
- **Model pairs:** 4 (`opus-4-7→haiku-4-5`, `sonnet-4-6→haiku-4-5`,
  `gpt-4o→gpt-4o-mini`, `o3→o4-mini`) spanning Anthropic + OpenAI

### Pricing provenance (real rates only)

`tokenjam.core.pricing.get_rates` returned non-placeholder rates for every
model used (input/output $/Mtok):

| Model | input | output |
|---|---|---|
| `claude-opus-4-7` | 5.00 | 25.00 |
| `claude-sonnet-4-6` | 3.00 | 15.00 |
| `claude-haiku-4-5` | 0.80 | 4.00 |
| `gpt-4o` | 2.50 | 10.00 |
| `gpt-4o-mini` | 0.15 | 0.60 |
| `o3` | 10.00 | 40.00 |
| `o4-mini` | 1.10 | 4.40 |

> `claude-sonnet-4-5` is in TokenJam's downgrade map but `get_rates` returns
> **no real rate** for it (would fall back to the `$0.50/$2.00` placeholder), so
> that pair was **deliberately excluded** rather than ship a
> `priced_with_defaults` headline.

