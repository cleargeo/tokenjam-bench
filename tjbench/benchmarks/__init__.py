"""Benchmark registry. `samples` is offline + dependency-free; the others load
real datasets lazily (tokenjam-bench[datasets])."""
from __future__ import annotations

from tjbench.benchmarks.base import Benchmark, ScoreResult, Task

BENCHMARK_NAMES = ["samples", "humaneval", "gsm8k", "judged"]
# Scenario-library suites (P1) are each their own AgentBenchmark.
_SCENARIO_SUITE_NAMES = ["coding-assistant", "rag-support", "research-agent", "browser-agent"]
AGENT_BENCHMARK_NAMES = (
    ["sample-agent", "swe-bench-lite", "real-scenarios"] + _SCENARIO_SUITE_NAMES
)


def get_benchmark(name: str) -> Benchmark:
    if name == "samples":
        from tjbench.benchmarks.samples import SampleBenchmark
        return SampleBenchmark()
    if name == "humaneval":
        from tjbench.benchmarks.humaneval import HumanEvalBenchmark
        return HumanEvalBenchmark()
    if name == "gsm8k":
        from tjbench.benchmarks.gsm8k import GSM8KBenchmark
        return GSM8KBenchmark()
    if name == "judged":
        from tjbench.benchmarks.judged import JudgedBenchmark
        return JudgedBenchmark()
    raise ValueError(f"Unknown benchmark '{name}'. Available: {BENCHMARK_NAMES}")


def get_agent_benchmark(name: str):
    """Agent benchmarks score an AgentTrace (final answer + tool usage)."""
    if name == "sample-agent":
        from tjbench.benchmarks.sample_agent import SampleAgentBenchmark
        return SampleAgentBenchmark()
    if name == "swe-bench-lite":
        from tjbench.benchmarks.swe_bench_lite import SWEBenchLiteBenchmark
        return SWEBenchLiteBenchmark()
    if name == "real-scenarios":
        from tjbench.benchmarks.real_scenarios import RealScenariosBenchmark
        return RealScenariosBenchmark()
    if name in _SCENARIO_SUITE_NAMES:  # P1 Real Scenario Library
        from tjbench.benchmarks.scenario_suites import get_scenario_suite
        return get_scenario_suite(name)
    raise ValueError(
        f"Unknown agent benchmark '{name}'. Available: {AGENT_BENCHMARK_NAMES}"
    )


__all__ = [
    "Benchmark", "ScoreResult", "Task", "get_benchmark", "BENCHMARK_NAMES",
    "get_agent_benchmark", "AGENT_BENCHMARK_NAMES",
]
