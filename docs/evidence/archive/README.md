# Archived evidence (historical record — not headline)

These are the earliest TokenJam Bench proof runs, kept for provenance. They are
**not** surfaced by the dashboard and are **not** headline evidence:
`scan_runs()` skips anything under `archive/`, and the CI honesty guard treats
this directory as non-headline (so its placeholder-priced costs never count as a
claim).

Why archived: these pre-date the real-priced `2026-06-26-multipair` run and were
priced with TokenJam's `$0.50/$2.00` **default placeholder rates**
(`priced_with_defaults=true`) — DeepSeek/early pairs had no real rate at the
time, so the dollar figures here are illustrative, not measured.

The single source of headline evidence is
[`docs/evidence/live/2026-06-26-multipair/`](../live/2026-06-26-multipair/) —
real measured rates across HumanEval, GSM8K, judged QA, and the production
workflow suites.

Contents:
- `2026-06-26-real-dashboard/` — the first multi-suite dashboard run (tj 0.5.1).
- `tjbench_{gsm8k,humaneval,judged}_tj0.5.1_*` — the first single-suite runs.
