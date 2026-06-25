# Test Suite

## Overview

The `tests/` directory contains **55 tests** covering all aspects of tokenjam-bench. All tests are **offline** — no API keys, no network, no provider SDKs. They run in ~5 seconds.

## Test Philosophy

- **100% offline**: All tests use `MockClient` / `MockAgentClient`
- **Deterministic**: Mock behavior is keyed by `# task_key:` markers
- **CI-ready**: No dataset downloads required for the core test suite
- **Honest about small samples**: Tests verify that `n=5` cannot produce a significant McNemar result

## Test Inventory

| Test File | Count | What It Verifies |
|-----------|-------|-----------------|
| `test_pipeline_offline.py` | 8 | End-to-end single-shot pipeline |
| `test_agent_pipeline_offline.py` | 3 | End-to-end agent pipeline |
| `test_agent_runner.py` | 3 | AgentRunner loop mechanics |
| `test_agent_validation.py` | 4 | Tool-call validation and safety gate |
| `test_scoring.py` | 4 | Code extraction and scoring |
| `test_stats.py` | 9 | Statistical correctness |
| `test_report.py` | 3 | Cost-validation and headline logic |
| `test_version_stamp.py` | 1 | TokenJam version resolution |
| `test_swe_bench_lite.py` | 20 | SWE-Bench Lite mock scoring and tools |
| **Total** | **55** | |

## Pipeline Tests

### `test_pipeline_offline.py`

End-to-end single-shot pipeline tests using mock clients.

| Test | What It Verifies |
|------|-----------------|
| `test_candidate_comes_from_tokenjam` | Downgrade map resolves correctly |
| `test_no_candidate_raises_without_override` | Explicit candidate required when TokenJam has none |
| `test_preserved_accuracy_and_cheaper` | Full-accuracy candidate = same pass rate, lower cost |
| `test_regression_is_detected` | Zero-accuracy candidate = total wipeout, all regressions flagged |
| `test_explicit_candidate_override` | `--candidate` bypasses TokenJam recommendation |
| `test_stats_block_is_attached` | Wilson CIs, McNemar, verdict present; small-n → `insufficient_evidence` |
| `test_samples_k_runs_each_task_k_times` | Multi-sample (`k>1`) runs work |
| `test_artifact_is_written_and_version_stamped` | JSON artifact written, filename contains version, content contains version |

### `test_agent_pipeline_offline.py`

End-to-end agent pipeline tests.

| Test | What It Verifies |
|------|-----------------|
| `test_ok_candidate_matches_original` | Agent proof with correct candidate = same pass rate, lower cost, same stats machinery |
| `test_unsafe_candidate_regresses_via_safety_gate` | Dangerous tool calls cause total pass-rate collapse even with correct answers |
| `test_token_and_cost_are_summed_over_turns` | Multi-turn token accumulation is measured and priced |

## Agent Tests

### `test_agent_runner.py`

AgentRunner loop mechanics.

| Test | What It Verifies |
|------|-----------------|
| `test_single_tool_loop_records_call_and_final_answer` | One tool turn + final answer, trace recorded |
| `test_multi_step_loop_runs_tools_in_order` | Two tool turns in correct order, 3 total turns |
| `test_max_turns_guard_stops_runaway` | Infinite tool-looping agent stops at `max_turns` |

### `test_agent_validation.py`

Tool-call validation and safety gate.

| Test | What It Verifies |
|------|-----------------|
| `test_ok_run_passes` | Correct tools + correct answer = pass |
| `test_wrong_answer_fails_even_with_right_tools` | Correct tools but wrong answer = fail |
| `test_dangerous_tool_fails_despite_correct_answer` | **Safety gate**: forbidden tool called → fail even if answer is right |
| `test_validate_tools_reports_structure` | Validation object reports expected tools, ordering, safety, error rate |

## Scoring Tests

### `test_scoring.py`

Objective scoring primitives.

| Test | What It Verifies |
|------|-----------------|
| `test_extract_code_unwraps_fence` | Markdown fence extraction |
| `test_score_code_pass_and_fail` | Code execution scoring (pass/fail) |
| `test_score_code_handles_timeout` | Infinite loop times out safely |
| `test_exact_match_extracts_final_number` | `####` marker and last-number extraction, normalization |

## Statistics Tests

### `test_stats.py`

Statistical correctness.

| Test | What It Verifies |
|------|-----------------|
| `test_wilson_interval_brackets_point_estimate` | Wilson CI contains observed proportion |
| `test_wilson_perfect_score_is_not_a_point` | 100% observed still has CI < 1 |
| `test_wilson_zero_n_is_fully_uncertain` | Degenerate case returns [0,1] |
| `test_mcnemar_significant_when_discordance_is_lopsided` | Lopsided discordance → significant |
| `test_mcnemar_not_significant_when_balanced` | Balanced discordance → not significant |
| `test_mcnemar_no_discordance_is_p1` | No discordance → p=1.0 |
| `test_small_n_cannot_reach_significance` | **Key point**: n=5 total wipeout cannot reach p<0.05 |
| `test_paired_delta_ci_spans_the_point_estimate` | Delta CI contains point estimate |
| `test_pass_at_k_estimator` | Unbiased pass@k estimator correctness |

## Report Tests

### `test_report.py`

Cost-validation and headline logic.

| Test | What It Verifies |
|------|-----------------|
| `test_token_inflation_flag_trips_when_candidate_is_verbose` | Candidate 2x output tokens flagged |
| `test_no_inflation_flag_when_comparable` | Near-equal tokens not flagged |
| `test_headline_carries_ci_and_verdict` | Headline string contains CI, McNemar p, verdict |

## SWE-Bench Lite Tests

### `test_swe_bench_lite.py`

SWE-Bench Lite mock scoring and tool operations.

#### Mock Scoring Tests (4)

| Test | What It Verifies |
|------|-----------------|
| `test_mock_scoring_with_all_tools` | Agent using view + str_replace + bash passes |
| `test_mock_scoring_missing_view` | Agent missing view fails |
| `test_mock_scoring_missing_edit` | Agent missing edit fails |
| `test_mock_scoring_missing_bash` | Agent missing bash fails |

#### Benchmark Tests (3)

| Test | What It Verifies |
|------|-----------------|
| `test_tools_registry_has_swe_bench_tools` | All 6 SWE-Bench tools registered |
| `test_extract_files_from_patch` | Patch parsing extracts correct file paths |
| `test_build_prompt_includes_problem` | Prompt includes repo, issue, problem statement |

#### Tool Operation Tests (13)

| Test | What It Verifies |
|------|-----------------|
| `test_view_file` | Read file with line numbers |
| `test_view_nonexistent_file` | Error for missing file |
| `test_view_range` | Read specific line range |
| `test_str_replace_exact_match` | Replace exact string once |
| `test_str_replace_no_match` | Error when old_str not found |
| `test_str_replace_multiple_matches` | Error when old_str appears multiple times |
| `test_create_file` | Create new file |
| `test_create_existing_file` | Error when file exists |
| `test_insert_after_line` | Insert after specific line |
| `test_bash_command` | Run shell command |
| `test_bash_timeout` | Timeout long-running commands |
| `test_path_traversal_blocked` | Block path traversal outside workspace |
| `test_tool_specs` | All 6 tool specs returned |

## Version Stamp Tests

### `test_version_stamp.py`

| Test | What It Verifies |
|------|-----------------|
| `test_resolves_installed_tokenjam_version` | Version resolves to real semver, dict round-trips |

## Running Tests

```bash
# All tests
make test

# Or directly
pytest -q

# Specific test file
pytest tests/test_swe_bench_lite.py -v

# With coverage
pytest --cov=tokenjam_bench --cov-report=term-missing
```

## Test Factories

Tests use simple dataclass construction rather than complex factories. Mock clients are instantiated directly with their behavior parameters:

```python
# Mock client for single-shot
client = get_client("anthropic:claude-opus-4-7", mock=True, mock_accuracy=0.8)

# Mock client for agent
client = get_tool_calling_client("anthropic:claude-opus-4-7", mock=True, behavior="ok")
```

## Related Documentation

- [Development Guide](development.md) — Contributing and extending
- [Pipelines](pipelines.md) — How tests verify pipeline behavior
- [Agents](agents.md) — How tests verify agent behavior
- [Statistics](statistics.md) — How tests verify statistical correctness
