"""Benchmark registry. `samples` is offline + dependency-free; the others load
real datasets lazily (tokenjam-bench[datasets])."""
from __future__ import annotations

from benchmarks.base import Benchmark, ScoreResult, Task

BENCHMARK_NAMES = ["samples", "humaneval", "gsm8k"]
AGENT_BENCHMARK_NAMES = ["sample-agent", "swe-bench-lite", "real-scenarios"]


def get_benchmark(name: str) -> Benchmark:
    if name == "samples":
        from benchmarks.samples import SampleBenchmark
        return SampleBenchmark()
    if name == "humaneval":
        from benchmarks.humaneval import HumanEvalBenchmark
        return HumanEvalBenchmark()
    if name == "gsm8k":
        from benchmarks.gsm8k import GSM8KBenchmark
        return GSM8KBenchmark()
    raise ValueError(f"Unknown benchmark '{name}'. Available: {BENCHMARK_NAMES}")


def get_agent_benchmark(name: str):
    """Agent benchmarks score an AgentTrace (final answer + tool usage)."""
    if name == "sample-agent":
        from benchmarks.sample_agent import SampleAgentBenchmark
        return SampleAgentBenchmark()
    if name == "swe-bench-lite":
        from benchmarks.swe_bench_lite import SWEBenchLiteBenchmark
        return SWEBenchLiteBenchmark()
    if name == "real-scenarios":
        from benchmarks.real_scenarios import RealScenariosBenchmark
        return RealScenariosBenchmark()
    raise ValueError(
        f"Unknown agent benchmark '{name}'. Available: {AGENT_BENCHMARK_NAMES}"
    )


__all__ = [
    "Benchmark", "ScoreResult", "Task", "get_benchmark", "BENCHMARK_NAMES",
    "get_agent_benchmark", "AGENT_BENCHMARK_NAMES",
]
