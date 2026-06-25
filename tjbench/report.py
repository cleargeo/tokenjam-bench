"""Proof artifacts: per-task records, the statistical block, and the aggregate.

The aggregate answers the boss's question for one (benchmark, original→candidate)
pair, stamped to a TokenJam build AND carrying its own error bars:

    "On {benchmark} (n={n}, k={samples}), TokenJam {version} recommends
     {orig}→{cand}. Δcost {cost%} (measured), Δpass-rate {pp}pp
     [95% CI ...], McNemar p={p} → {verdict}."

Honesty discipline (mirrors TokenJam's Rule 14):
  - accuracy = benchmark pass-rate, never a general 'quality preserved' claim;
  - the verdict is 'no_significant_regression' / 'significant_regression' /
    'insufficient_evidence' — NEVER 'SAFE', which would assert equivalence;
  - there is deliberately NO single fabricated 'confidence = 95%' scalar; the
    honest expression of confidence is the CI + p-value (a scalar 'safe %' would
    recreate the hardcoded-confidence flaw we criticized in TokenJam itself);
  - cost is from MEASURED token usage, with a flag when the candidate inflated
    token count (savings already reflect it, but the per-token edge is eroded).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Verdicts. Hedged on purpose — a benchmark can falsify "preserved", never
# certify "safe".
NO_SIGNIFICANT_REGRESSION = "no_significant_regression"
SIGNIFICANT_REGRESSION = "significant_regression"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"

# Candidate output-token inflation above this multiple is flagged: the measured
# savings still hold, but verbosity/retries are eroding the per-token advantage.
TOKEN_INFLATION_FLAG_RATIO = 1.25


@dataclass
class TaskOutcome:
    task_id: str
    samples: int
    original_passes: int             # passing samples out of `samples`
    candidate_passes: int
    original_cost_usd: float          # measured, summed over samples
    candidate_cost_usd: float
    original_output_tokens: int       # summed over samples
    candidate_output_tokens: int
    original_detail: str = ""
    candidate_detail: str = ""

    @property
    def original_passed(self) -> bool:
        """Majority-vote pass over the k samples (tie → pass)."""
        return self.original_passes * 2 >= self.samples

    @property
    def candidate_passed(self) -> bool:
        return self.candidate_passes * 2 >= self.samples

    @property
    def regressed(self) -> bool:
        return self.original_passed and not self.candidate_passed

    @property
    def recovered(self) -> bool:
        return self.candidate_passed and not self.original_passed


@dataclass
class ProofStats:
    samples_per_task: int
    original_ci_pp: list[float]       # [low, high] Wilson CI on original pass rate
    candidate_ci_pp: list[float]
    delta_ci_pp: list[float]          # [low, high] paired CI on the pass-rate delta
    mcnemar_b: int                    # original PASS, candidate FAIL
    mcnemar_c: int                    # original FAIL, candidate PASS
    mcnemar_p_value: float
    significant: bool                 # p < alpha
    alpha: float
    verdict: str
    original_pass_at_1: float         # mean per-sample pass rate
    candidate_pass_at_1: float


@dataclass
class ProofResult:
    tokenjam_version: str
    tokenjam_location: str | None
    benchmark: str
    original_model: str
    candidate_model: str
    recommended_by: str
    n_tasks: int
    original_pass: int                # tasks passed (majority vote)
    candidate_pass: int
    original_cost_usd: float          # measured
    candidate_cost_usd: float
    original_output_tokens: int
    candidate_output_tokens: int
    mock: bool
    priced_with_defaults: bool
    stats: ProofStats
    created_at: float = field(default_factory=time.time)
    tasks: list[TaskOutcome] = field(default_factory=list)

    # -- derived --
    @property
    def original_pass_rate(self) -> float:
        return self.original_pass / self.n_tasks if self.n_tasks else 0.0

    @property
    def candidate_pass_rate(self) -> float:
        return self.candidate_pass / self.n_tasks if self.n_tasks else 0.0

    @property
    def accuracy_delta_pp(self) -> float:
        return round((self.candidate_pass_rate - self.original_pass_rate) * 100, 2)

    @property
    def cost_delta_pct(self) -> float:
        if self.original_cost_usd <= 0:
            return 0.0
        return round(
            (self.candidate_cost_usd - self.original_cost_usd) / self.original_cost_usd * 100, 2
        )

    @property
    def regressions(self) -> int:
        return sum(1 for t in self.tasks if t.regressed)

    @property
    def output_token_inflation(self) -> float:
        """Candidate output tokens ÷ original. >1 means the cheaper model is
        more verbose — eroding the per-token savings."""
        if self.original_output_tokens <= 0:
            return 0.0
        return round(self.candidate_output_tokens / self.original_output_tokens, 3)

    @property
    def token_inflation_flag(self) -> bool:
        return self.output_token_inflation > TOKEN_INFLATION_FLAG_RATIO

    def headline(self) -> str:
        prefix = "[MOCK — illustrative] " if self.mock else ""
        d = self.stats.delta_ci_pp
        return (
            f"{prefix}{self.benchmark} (n={self.n_tasks}, k={self.stats.samples_per_task}) · "
            f"tokenjam {self.tokenjam_version}: {self.original_model} → {self.candidate_model} · "
            f"Δcost {self.cost_delta_pct:+.1f}% (measured) · "
            f"Δpass-rate {self.accuracy_delta_pp:+.1f}pp [95% CI {d[0]:+.1f}, {d[1]:+.1f}] · "
            f"McNemar p={self.stats.mcnemar_p_value:.3f} → {self.stats.verdict}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["original_pass_rate"] = round(self.original_pass_rate, 4)
        d["candidate_pass_rate"] = round(self.candidate_pass_rate, 4)
        d["accuracy_delta_pp"] = self.accuracy_delta_pp
        d["cost_delta_pct"] = self.cost_delta_pct
        d["regressions"] = self.regressions
        d["output_token_inflation"] = self.output_token_inflation
        d["token_inflation_flag"] = self.token_inflation_flag
        d["headline"] = self.headline()
        return d

    def write(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = int(self.created_at)
        path = out / f"tjbench_{self.benchmark}_tj{self.tokenjam_version}_{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
