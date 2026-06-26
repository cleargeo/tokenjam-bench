#!/usr/bin/env bash
# Run TokenJam Bench against REAL provider APIs and write versioned proof
# artifacts. Every production-dashboard number traces back to these runs —
# there is no seeded or synthetic data in production (scan_runs() skips any
# artifact marked mock/demo).
#
# Requires a provider key in the environment (read from env only, never
# committed):
#   export DEEPSEEK_API_KEY=sk-...
#
# DeepSeek-only by default (reasoner -> chat). Supply ORIG/CAND (and the
# matching provider key) to benchmark Anthropic/OpenAI/Gemini pairs — the
# pipeline is identical:
#   ORIG=openai:gpt-4o CAND=openai:gpt-4o-mini OPENAI_API_KEY=... ./scripts/run_real_benchmarks.sh
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   ./scripts/run_real_benchmarks.sh
#   tjbench serve   # dashboard now reads these real artifacts
set -uo pipefail
cd "$(dirname "$0")/.."

ORIG="${ORIG:-deepseek:deepseek-reasoner}"
CAND="${CAND:-deepseek:deepseek-chat}"
LIMIT="${LIMIT:-12}"
OUT="${OUT:-results}"
PY="${PY:-python}"
export TJBENCH_JUDGE="${TJBENCH_JUDGE:-deepseek}"
export TJBENCH_JUDGE_METRIC="${TJBENCH_JUDGE_METRIC:-correctness}"
export HF_HUB_DISABLE_PROGRESS_BARS=1 DEEPEVAL_TELEMETRY_OPT_OUT=YES

echo "Real benchmark run: ${ORIG} -> ${CAND} (limit ${LIMIT})  ->  ${OUT}/"
run(){ echo; echo "===== $* ====="; "$PY" run.py "$@" --original "$ORIG" --candidate "$CAND" --html --out "$OUT"; }

# Executable benchmarks (public datasets; code/answer execution — no judge):
run run --benchmark gsm8k     --limit "$LIMIT"
run run --benchmark humaneval --limit "$LIMIT"

# LLM-judged QA (DeepEval judge selected by TJBENCH_JUDGE):
run run --benchmark judged

# Production text workflows (judge-scored against grounded references):
for s in customer-support enterprise-rag email-assistant research-assistant; do
  run workflow "$s" --limit "$LIMIT"
done

echo
echo "Done. Versioned JSON + HTML artifacts in ${OUT}/."
echo "Restart 'tjbench serve' to refresh the dashboard from these real runs."

# Not benchmarkable with a DeepSeek-only key (documented limitation):
#   - mbpp: no real loader yet (add an MBPPBenchmark like humaneval).
#   - n8n / coding-workflow / scenario suites: multi-turn tool-calling; the
#     deepseek-reasoner (R1) endpoint does not support function calling, and
#     both sides of an agent benchmark need it. Use a tool-capable provider.
#   - swe-bench-lite / replay: need a repo-execution harness / captured
#     telemetry input respectively.
