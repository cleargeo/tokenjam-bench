# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tokenjam-bench` (CLI: `tjbench`) is an evidence-based benchmarking harness that **proves** the effect of TokenJam's model-downsize recommendations on cost *and* accuracy. It runs benchmark tasks on an "original" model and the cheaper "candidate" model TokenJam would route to, scores both against objective ground truth (executable tests / exact-match / LLM judge), and emits a version-stamped proof artifact with full statistics (Wilson CI, McNemar exact, paired delta CI). Accuracy is always *pass-rate on the chosen suite* — never a general "quality preserved" claim.

## Shared conventions (hygiene)

> **Shared across all TokenJam repos. Canonical source: `tokenjam/CLAUDE.md`.
> If you change this section, propagate it to the other repos (bench, engine,
> website, ai, cloud). Repo-specific rules live elsewhere in this file.**

### Concurrent agents — one worktree per task
When more than one agent edits this repo in parallel, each agent MUST work in its
own git worktree. A single working dir shares one `HEAD`, so two `git commit`s
from different agents land on whichever branch was checked out last, leaking
commits into the wrong PR. Before starting:
```bash
git worktree add -b <type>/<task> ../<repo>-<task> main
cd ../<repo>-<task>
```
Prune when the PR merges: `git worktree remove ../<repo>-<task>`. Symptom of a
missed worktree: `git log` shows a commit on a branch you didn't intend. Don't
force-push to fix it — rebase the stray commit off your branch first.

### Branch + PR naming
- Branches are slash-separated, kebab-case, type-prefixed: `fix/<area>`,
  `feat/<area>`, `docs/<area>`, `chore/<area>`, `release/<X.Y.Z>`.
- PR titles lead with the verb/type and reference issues by number when
  applicable: `Fix #N: …`, `[feature] … (#N)`, `docs: …`, `Bump version to X.Y.Z`.
- Use `Closes #N` in the PR body (one per line — not `Closes #1, #2`; GitHub
  only catches the first) so issues auto-close on merge.

### Commit messages
- Subject (≤72 chars): one-line summary, active voice, reference issues with `#N`.
- Body (after a blank line): explain *why*, not *what* (the diff shows what).
- Trailers (after another blank line, at the end): always
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (or the
  appropriate model identifier). When fixing an externally-reported bug, also
  credit the reporter: `Co-Authored-By: <handle> <noreply@github.com>`.
- Use a HEREDOC for multi-line messages to preserve formatting.

### PR body structure
A framing sentence, then `## Summary` (high-level bullets), per-issue/feature
detail, `## Tests / Verification` (what you ran + results), and
`## What's NOT in this PR` when scope was deliberately limited (load-bearing —
it tells the reviewer what you chose to defer).

### Self-review checklist (before requesting review)
1. Tests pass locally.
2. Lint + type-check clean for files you touched.
3. CI green on the branch (at least the fast jobs).
4. Acceptance criteria from the issue met, one by one.
5. No accidental files in the diff (local config, test data, debug artifacts).
6. PR body explains the WHY (symptom + root cause + fix).
7. Honesty preserved (below) — no user-facing claim silently strengthened.

### Scope discipline
Do what the brief/issue says, no more. Notice an adjacent issue? File it
separately rather than expanding the PR. Exception: a change functionally
required to make the primary fix work (note it under "What's also in this PR").
When unsure about scope, ask before expanding.

### Worker vs master
Worker agents open PRs and request review; they do NOT merge their own PRs, file
follow-up issues unprompted, or bump versions. The master + the human handle
merges, follow-ups, and releases.

### Honesty discipline (the brand)
Never overclaim. Use "candidate," "measured," "estimated," "looks like,"
"review before…" — never "safe," "certified," "guaranteed," "saves you,"
"quality preserved." No fabricated customer logos, testimonials, or compliance
badges we don't hold. Forward-looking work is labeled as such (early-access /
design-partner / roadmap), never as shipped/GA. If you touch a user-facing
claim, verify it matches existing caveat language; never silently strengthen it.
(Each repo keeps its own specific instantiation of this rule.)

### Writing style (all prose: PRs, issues, docs, marketing, comments)
Write like a person, not an LLM. Top tells to strip while drafting: em-dashes
(≤3 per piece), "X but Y" / "not just X" contrast-pair cadence (≤2), and default
tricolons ("A, B, and C" — vary the list lengths). Also: cut filler qualifiers
(typically / generally / it's worth noting), no section-transition recaps, no
"let me explain" / "in other words," vary sentence length, cut adjective stacks.

## Commands

```bash
make install            # pip install -e ".[dev]"
make test               # pytest -q  (full suite, offline, ~5s)
make lint               # ruff check .
make bench-smoke        # offline end-to-end smoke (no keys, no spend)
make update-tokenjam    # pip install -U tokenjam; print resolved version
make serve              # live proof dashboard at http://127.0.0.1:7392/

pytest -q tests/test_stats.py                     # single file
pytest -q tests/test_stats.py::test_wilson_basic  # single test
```

Run the CLI from a checkout via `python3 run.py <cmd>` (the installed `tjbench` console script works identically). Commands: `version | recommend | run | workflow | agent | report | scenarios | replay | matrix | serve`.

Model specs are always `provider:model` (e.g. `anthropic:claude-opus-4-7`, `deepseek:deepseek-reasoner`). `--mock` runs the entire pipeline deterministically with zero provider SDKs/keys/spend (numbers illustrative, plumbing real). Omitting `--candidate` makes the bench resolve it live from TokenJam's downgrade analyzer.

## Architecture

**Flat layout** — top-level `.py` files and the `tjbench/` subpackages live at the package root; there is no inner `tokenjam_bench/` directory. The CLI entry point is `tjbench/cli.py` → `cli()`.

The two proof pipelines share one statistical backbone:

- **Single-shot** (`pipeline.py`): for each benchmark task, run original + candidate, score with `benchmarks/`, price with `cost.py`, collect `TaskOutcome`s, then `assemble_proof()` → `ProofResult`.
- **Agent** (`agent_pipeline.py`): same shape, but `agents/runner.py:AgentRunner` drives a multi-turn tool-calling loop producing an `AgentTrace`, scored through `agents/validation.py:validate_tools` (safety gate, tool ordering, forbidden tools).

Both end in `assemble_proof()` (in `pipeline.py`), which stamps the TokenJam version and computes the stats block.

**Protocols, not inheritance.** `ModelClient` (`models/base.py`), `ToolCallingClient` (`models/tool_calling.py`), `Benchmark` (`benchmarks/base.py`), and `AgentBenchmark` (`benchmarks/agent_base.py`) are protocols. Clients are produced by `models/registry.py:get_client`; benchmarks by `benchmarks/__init__.py:get_benchmark` / `get_agent_benchmark`.

**TokenJam is a black-box published dependency** — consumed via `pip`, never vendored or pinned to a checkout. Three integration points, all importing from the installed package:
1. `recommend.py` → candidate from `tokenjam.core.optimize.DOWNGRADE_CANDIDATES`
2. `cost.py` → pricing from `tokenjam.core.pricing.get_rates`
3. `version.py` → `importlib.metadata.version("tokenjam")`, stamped onto every artifact

Every artifact in `results/` carries `tokenjam_version`, so `tjbench matrix` can diff proofs across releases and catch the day a TokenJam change moves the numbers (it exits non-zero on regression — usable as a CI guard).

**Statistics are zero-dependency** (`stats.py`) — Wilson interval, McNemar exact, paired delta CI, pass@k implemented from first principles (no `scipy`/`numpy`). There is deliberately **no single `confidence = 95%` scalar**; the honest output is CI + p-value. Verdicts are `no_significant_regression` / `significant_regression` / `insufficient_evidence` — never `SAFE`.

## Conventions

- **Offline-first.** The `samples` and `sample-agent` benchmarks and the entire test suite run with no provider SDKs, no keys, no network. Live providers (`[providers]`), datasets (`[datasets]`), and the DeepEval judge (`[judge]`) are optional extras. Live clients **lazy-import** their SDKs so the offline path never needs them.
- **Tests are 100% offline and deterministic** via `MockClient`/`MockAgentClient`; mock behavior is keyed by `# task_key:` markers in prompts. Live-provider tests skip cleanly when extras aren't installed. Tests also assert honesty properties (e.g. `n=5` cannot yield a significant McNemar result).
- **Pure domain logic**: `models/`, `benchmarks/`, `agents/` import no CLI or HTTP code.
- **No unicode bullets in CLI output** — `rich` handles formatting.
- Ruff: line length 100, target py310.

## Adding things

- **Benchmark**: new file in `benchmarks/` implementing `Benchmark`/`AgentBenchmark`, register in `benchmarks/__init__.py`, add tests.
- **Model client**: new file in `models/` implementing `ModelClient`/`ToolCallingClient`, register in `models/registry.py`, add tests.

## Honesty constraints (these are load-bearing, not decoration)

When touching reports, stats, or pipelines: `--mock` runs must be flagged illustrative; cost must flag when it fell back to TokenJam's `$0.50/$2.00` placeholder rates; `n` is always reported; production dashboards show **only real measured runs**, never synthetic data. HumanEval executes model-generated code in a timed subprocess (`exec_sandbox.py`) — run only trusted suites on a machine you control.
