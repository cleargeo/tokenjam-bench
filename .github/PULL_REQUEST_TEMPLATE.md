<!-- Thanks for contributing to TokenJam Bench. -->

## What & why

<!-- What does this change do, and why? Link any related issue. -->

## Checklist

- [ ] `ruff check .` passes
- [ ] `pytest` passes locally (includes the honesty guard)
- [ ] Docs updated if behavior or CLI changed
- [ ] No placeholder-priced run is surfaced as headline/dashboard evidence
      (`priced_with_defaults=true` stays under `docs/evidence/archive/`)
- [ ] No banned overclaim strings added (use Wilson CI + McNemar p + the hedged
      verdicts, never "quality preserved" / a single `confidence = NN%` scalar /
      ROI extrapolation)

## Evidence (if numbers changed)

<!-- Paste the relevant artifact filename(s) under docs/evidence/ or results/,
     or the headline line, so reviewers can reproduce. -->
