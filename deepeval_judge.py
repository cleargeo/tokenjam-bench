"""DeepEval-backed judge, provider-agnostic over OpenAI-compatible endpoints.

DeepEval supplies the metrics (correctness / answer-relevancy / faithfulness /
task-completion); the judge MODEL behind them can be OpenAI or DeepSeek (or any
future OpenAI-compatible provider) — all via the single abstraction in
models/openai_compatible.py. DeepSeek is wired as a custom `DeepEvalBaseLLM` so
metrics call DeepSeek's `/chat/completions`.

Key handling: the API key is read from the provider's env var (DEEPSEEK_API_KEY,
OPENAI_API_KEY, …) at call time and never stored, logged, or persisted.
Everything here is lazy + key-gated — importing this module needs neither
DeepEval nor a key. The MockJudge (judge.py) is the offline-tested path; this
adapter is verified the moment a valid key is exported.

  pip install -e ".[judge,providers]"
  export DEEPSEEK_API_KEY=...
  TJBENCH_JUDGE=deepseek TJBENCH_JUDGE_METRIC=correctness \
      tjbench run --benchmark judged --original deepseek:deepseek-chat
"""
from __future__ import annotations

from judge import JUDGE_METRICS, JudgeCase, JudgeResult


def _make_eval_model(provider_name: str, model: str):
    """Build a DeepEval custom model backed by an OpenAI-compatible provider.

    Defined inside a factory so subclassing DeepEvalBaseLLM (and importing
    DeepEval at all) only happens when a real evaluation runs.
    """
    from deepeval.models import DeepEvalBaseLLM  # lazy

    from models.openai_compatible import PROVIDERS, _make_openai_client

    if provider_name not in PROVIDERS:
        raise ValueError(f"'{provider_name}' is not an OpenAI-compatible provider.")
    prov = PROVIDERS[provider_name]

    class _OpenAICompatibleEvalModel(DeepEvalBaseLLM):
        def __init__(self) -> None:
            self._model = model

        def load_model(self):
            return self

        def generate(self, prompt: str, schema=None):
            client = _make_openai_client(prov)  # key from env, never stored
            if schema is not None:
                resp = client.chat.completions.create(
                    model=self._model, temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                return schema.model_validate_json(content)
            resp = client.chat.completions.create(
                model=self._model, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return f"{provider_name}:{self._model}"

    return _OpenAICompatibleEvalModel()


class DeepEvalJudge:
    name = "deepeval"

    def __init__(self, metric: str = "correctness", threshold: float = 0.5,
                 model: str | None = None, provider: str = "openai") -> None:
        if metric not in JUDGE_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Available: {JUDGE_METRICS}")
        from models.openai_compatible import PROVIDERS
        if provider not in PROVIDERS:
            raise ValueError(f"'{provider}' is not an OpenAI-compatible provider.")
        self.metric = metric
        self.threshold = threshold
        self.provider = provider
        self.model = model or PROVIDERS[provider].default_model

    def _build_metric(self):
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                GEval,
            )
            from deepeval.test_case import LLMTestCaseParams
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "DeepEval is not installed. Run `pip install -e '.[judge]'` and "
                "export a judge-model API key (e.g. DEEPSEEK_API_KEY)."
            ) from exc

        eval_model = _make_eval_model(self.provider, self.model)
        if self.metric == "answer-relevancy":
            return AnswerRelevancyMetric(threshold=self.threshold, model=eval_model)
        if self.metric == "faithfulness":
            return FaithfulnessMetric(threshold=self.threshold, model=eval_model)
        if self.metric == "task-completion":
            return GEval(
                name="TaskCompletion",
                criteria="Whether the actual output fully completes the task in the input.",
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
                threshold=self.threshold, model=eval_model)
        return GEval(  # correctness
            name="Correctness",
            criteria="Whether the actual output is factually correct and "
                     "semantically equivalent to the expected output.",
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=self.threshold, model=eval_model)

    def evaluate(self, case: JudgeCase) -> JudgeResult:  # pragma: no cover - needs a key
        from deepeval.test_case import LLMTestCase

        metric = self._build_metric()
        tc = LLMTestCase(
            input=case.input, actual_output=case.actual_output,
            expected_output=case.expected_output, retrieval_context=case.context,
        )
        metric.measure(tc)
        score = float(metric.score or 0.0)
        try:
            passed = bool(metric.is_successful())
        except Exception:
            passed = score >= self.threshold
        return JudgeResult(
            metric=f"{self.metric}@{self.provider}:{self.model}",
            score=round(score, 4), threshold=self.threshold, passed=passed,
            reason=str(getattr(metric, "reason", "") or ""),
        )
