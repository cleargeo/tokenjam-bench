#!/usr/bin/env bash
# Broad, honestly-priced evidence for TokenJam's REAL downsize recommendations.
#
# Pairs are TokenJam's own downgrade map (tokenjam.core.optimize.DOWNGRADE_CANDIDATES);
# pricing is real (tokenjam.core.pricing.get_rates returns non-placeholder rates for
# every model used — verified before running). Keys are read from the environment
# only (ANTHROPIC_API_KEY / OPENAI_API_KEY) and never committed.
#
# Usage:
#   GROUP=anthropic ./scripts/run_multipair_evidence.sh   # opus-4-7, sonnet-4-6 -> haiku-4-5
#   GROUP=openai    ./scripts/run_multipair_evidence.sh   # gpt-4o, o3 -> 4o-mini/o4-mini
#   GROUP=judged    ./scripts/run_multipair_evidence.sh   # secondary judged/workflow suites
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-docs/evidence/live/2026-06-26-multipair}"
PY="${PY:-python3}"
N="${N:-50}"
mkdir -p "$OUT"
export HF_HUB_DISABLE_PROGRESS_BARS=1 DEEPEVAL_TELEMETRY_OPT_OUT=YES TOKENIZERS_PARALLELISM=false

run() { echo; echo "===== $(date +%H:%M:%S) $* ====="; "$PY" run.py "$@" --html --out "$OUT"; }

# Objective, executable suites at n=50 — the load-bearing evidence.
# o3/o4-mini are reasoning models: reasoning tokens count against the output
# budget, so give them headroom to avoid truncation-as-false-failure.
obj() { # obj <orig> <cand> <max_tokens>
  local orig="$1" cand="$2" mt="$3"
  run run --benchmark gsm8k     --original "$orig" --candidate "$cand" --limit "$N" --max-tokens "$mt"
  run run --benchmark humaneval --original "$orig" --candidate "$cand" --limit "$N" --max-tokens "$mt"
}

case "${GROUP:?set GROUP=anthropic|openai|judged}" in
  anthropic)
    obj anthropic:claude-opus-4-7   anthropic:claude-haiku-4-5 2048
    obj anthropic:claude-sonnet-4-6 anthropic:claude-haiku-4-5 2048
    ;;
  openai)
    obj openai:gpt-4o openai:gpt-4o-mini 2048
    obj openai:o3     openai:o4-mini     8000
    ;;
  judged)
    # SECONDARY signal only — judge-scored (GEval correctness via gpt-4o),
    # NOT objective pass/fail. Run for one flagship pair per provider.
    export TJBENCH_JUDGE=openai TJBENCH_JUDGE_METRIC=correctness
    JN="${JN:-12}"
    for pair in "anthropic:claude-opus-4-7 anthropic:claude-haiku-4-5" \
                "openai:gpt-4o openai:gpt-4o-mini"; do
      set -- $pair; orig="$1"; cand="$2"
      run run --benchmark judged --original "$orig" --candidate "$cand"
      for s in customer-support enterprise-rag email-assistant research-assistant; do
        run workflow "$s" --original "$orig" --candidate "$cand" --limit "$JN"
      done
    done
    ;;
  *) echo "unknown GROUP"; exit 2 ;;
esac
echo; echo "Done ($GROUP) at $(date +%H:%M:%S)."
