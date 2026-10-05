"""Offline simulation of the verification/repair loop on pre-generated candidates.

Regeneration k uses the k-th stored temperature sample (an i.i.d. draw), so no new model calls are needed and results are
exactly reproducible. Ground truth is used ONLY after the loop has decided, to score outcomes — never inside it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from harness.routing.router import ESDRouter, Outcome


@dataclass
class Candidates:
    correct: np.ndarray       # (n, K) bool, K = 1 + max_retries; col 0 = original greedy answer
    verify_pass: np.ndarray   # (n, K) bool, verifier decision per candidate (label-free)


def simulate(risk: np.ndarray, budget: float | None, cands: Candidates, max_retries: int = 2,
             tau: float | None = None, return_items: bool = False):
    """Flag the top-``budget`` fraction by risk (or risk >= tau) for verification, then run the bounded repair loop."""
    n, K = cands.correct.shape
    if tau is None:
        tau = np.inf if budget is not None and budget <= 0 else (
            -np.inf if budget is not None and budget >= 1 else np.quantile(risk, 1 - budget))
    final_correct = np.zeros(n, dtype=bool)
    returned = np.zeros(n, dtype=bool)
    verifs = regens = 0
    outcomes: list[Outcome] = []
    for i in range(n):
        router = ESDRouter(tau, lambda k, i=i: bool(cands.verify_pass[i, k]), min(max_retries, K - 1))
        r = router.route(float(risk[i]), K)
        verifs += r.n_verifications
        regens += r.n_regenerations
        outcomes.append(r.outcome)
        if r.chosen_idx is not None:
            returned[i] = True
            final_correct[i] = cands.correct[i, r.chosen_idx]
    flagged = np.array([o is not Outcome.RETURNED for o in outcomes])
    init_wrong, init_right = ~cands.correct[:, 0], cands.correct[:, 0]
    acc_err = float((~final_correct[returned]).mean()) if returned.any() else float("nan")
    out = {
        "flag_rate": float(flagged.mean()),
        "coverage": float(returned.mean()),
        "abstain_rate": float(1 - returned.mean()),
        "initial_error_rate": float(init_wrong.mean()),
        "accepted_error_rate": acc_err,
        "error_leakage": float((returned & ~final_correct).mean()),  # wrong answers that reached the user, per request
        "repair_success_rate": float(final_correct[flagged & init_wrong].mean()) if (flagged & init_wrong).any() else float("nan"),
        "harm_rate": float((~final_correct[flagged & init_right]).mean()) if (flagged & init_right).any() else float("nan"),
        "correct_answers_lost": float((init_right & ~final_correct).mean()),  # cost of over-rejection / replacement
        "verifications_per_item": verifs / n,
        "regenerations_per_item": regens / n,
        "n": n,
    }
    if return_items:
        return out, {"returned": returned, "final_correct": final_correct, "flagged": flagged}
    return out
