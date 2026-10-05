"""ESD router: risk score -> RETURN (low risk) or VERIFY (high risk) -> PASS / REPAIR (bounded retries) / UNCERTAIN."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence


class Outcome(str, Enum):
    RETURNED = "returned"          # low risk, returned without verification
    VERIFIED_PASS = "verified_pass"
    REPAIRED = "repaired"          # a regenerated candidate passed verification
    UNCERTAIN = "uncertain"        # could not establish correctness within the retry limit -> abstain


@dataclass
class RouteResult:
    outcome: Outcome
    chosen_idx: int | None         # index into [greedy, *regenerations]; None when abstaining
    n_verifications: int = 0
    n_regenerations: int = 0
    trace: list[str] = field(default_factory=list)


class ESDRouter:
    """Pure control logic; candidates and the verifier are injected (dependency injection) so the same loop is used in
    simulation and in a live deployment."""

    def __init__(self, tau_risk: float, verify: Callable[[int], bool], max_retries: int = 2):
        self.tau, self.verify, self.max_retries = tau_risk, verify, max_retries

    def route(self, risk: float, n_candidates: int) -> RouteResult:
        if risk < self.tau:
            return RouteResult(Outcome.RETURNED, 0, 0, 0, ["low risk"])
        res = RouteResult(Outcome.UNCERTAIN, None)
        for k in range(min(self.max_retries, n_candidates - 1) + 1):  # k=0 is the original answer
            res.n_regenerations = k  # k-th candidate required k regenerations
            res.n_verifications += 1
            res.trace.append(f"verify#{k}")
            if self.verify(k):
                res.outcome = Outcome.VERIFIED_PASS if k == 0 else Outcome.REPAIRED
                res.chosen_idx = k
                return res
        return res
