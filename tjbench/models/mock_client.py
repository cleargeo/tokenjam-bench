"""Offline, deterministic model client.

Lets the whole pipeline — and CI — run end-to-end with zero provider SDKs, zero
keys, and zero spend. It is NOT a quality proxy: it returns a canned solution
for known sample tasks so the plumbing (scoring, cost, reporting, version stamp)
can be exercised. Real proofs use the live provider clients.

`accuracy` lets a test simulate a weaker candidate: with accuracy < 1.0 the
client deterministically "fails" a fraction of tasks (returns a wrong answer),
so before/after accuracy deltas are observable offline.
"""
from __future__ import annotations

import hashlib

from tjbench.models.base import Completion

# Canned correct answers for the built-in sample tasks (see benchmarks/samples.py).
_CANNED: dict[str, str] = {
    "samples/add": "def add(a, b):\n    return a + b\n",
    "samples/is_even": "def is_even(n):\n    return n % 2 == 0\n",
    "samples/factorial": (
        "def factorial(n):\n"
        "    result = 1\n"
        "    for i in range(2, n + 1):\n"
        "        result *= i\n"
        "    return result\n"
    ),
    "samples/gsm-apples": "#### 8",
    "samples/gsm-double": "#### 14",
}


class MockClient:
    def __init__(self, model: str = "mock-model", provider: str = "mock",
                 accuracy: float = 1.0) -> None:
        self.provider = provider
        self.model = model
        self.accuracy = accuracy

    def _fails_this_task(self, task_key: str) -> bool:
        if self.accuracy >= 1.0:
            return False
        # Deterministic per (model, task): hash to [0,1), fail when above accuracy.
        h = hashlib.sha256(f"{self.model}:{task_key}".encode()).hexdigest()
        frac = int(h[:8], 16) / 0xFFFFFFFF
        return frac > self.accuracy

    def complete(self, prompt: str, system: str | None = None,
                 max_tokens: int = 1024, temperature: float = 0.0) -> Completion:
        # Deterministic by design — temperature is accepted for interface parity
        # but ignored. Multi-sample variance only appears on live models.
        _ = temperature
        # The prompt embeds a `# task_key:` marker, and optionally a `# echo:`
        # directive carrying the canned correct output (judged benchmarks use
        # this so any task drives the mock without hardcoding answers here).
        task_key = ""
        echo = ""
        for line in prompt.splitlines():
            if line.startswith("# task_key:"):
                task_key = line.split(":", 1)[1].strip()
            elif line.startswith("# echo:"):
                echo = line.split(":", 1)[1].strip()
        answer = echo or _CANNED.get(task_key, "def _unsolved():\n    pass\n")
        if task_key and self._fails_this_task(task_key):
            answer = "totally unrelated wrong output #### -1"
        # Token counts: cheap, deterministic, length-based (good enough for the
        # offline cost path — real numbers come from live clients).
        return Completion(
            text=answer,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(answer) // 4),
        )
