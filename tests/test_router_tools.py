import numpy as np

from harness.routing.router import ESDRouter, Outcome
from harness.routing.simulate import Candidates, simulate
from harness.verification.tools import Verdict, arithmetic_tool, ordering_tool


def test_router_low_risk_returns_without_verification():
    r = ESDRouter(0.5, lambda k: False).route(0.1, 3)
    assert r.outcome is Outcome.RETURNED and r.n_verifications == 0


def test_router_repair_and_retry_limit_and_abstain():
    r = ESDRouter(0.5, lambda k: k == 2, max_retries=2).route(0.9, 3)
    assert r.outcome is Outcome.REPAIRED and r.chosen_idx == 2 and r.n_regenerations == 2
    r = ESDRouter(0.5, lambda k: k == 3, max_retries=2).route(0.9, 4)  # would pass at k=3 but limit is 2
    assert r.outcome is Outcome.UNCERTAIN and r.chosen_idx is None and r.n_verifications == 3


def test_simulation_reduces_accepted_errors_with_informative_risk():
    n = 200
    rng = np.random.default_rng(0)
    wrong = rng.random(n) < 0.5
    correct = np.stack([~wrong, rng.random(n) < 0.5, rng.random(n) < 0.5], 1)
    vp = correct.copy()  # perfect verifier
    risk = wrong + 0.1 * rng.random(n)
    base = simulate(risk, 0.0, Candidates(correct, vp))
    routed = simulate(risk, 0.5, Candidates(correct, vp))
    assert routed["accepted_error_rate"] < base["accepted_error_rate"]
    assert base["coverage"] == 1.0 and routed["coverage"] <= 1.0


def test_tools():
    assert arithmetic_tool("3 + 4 = 8.").verdict is Verdict.FAIL
    assert arithmetic_tool("3 + 4 = 7.").verdict is Verdict.UNKNOWN
    prompt = "Statements: Alice is taller than Bob. Bob is taller than Carol. Question: who?"
    assert ordering_tool(prompt, "Carol is taller than Alice.").verdict is Verdict.FAIL
    assert ordering_tool(prompt, "Alice is taller than Carol.").verdict is Verdict.UNKNOWN
