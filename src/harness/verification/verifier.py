"""Second-stage verifier: independent judge + deterministic tools -> PASS / FAIL decision."""
from __future__ import annotations

from dataclasses import dataclass

from harness.core.types import Example
from harness.verification.tools import ToolResult, Verdict, run_tools


@dataclass
class VerifierDecision:
    passed: bool
    reason: str
    tool: ToolResult


class Verifier:
    """Conservative combination: a failing tool vetoes; a passing complete tool accepts; otherwise the independent judge
    must reach P(yes) >= tau_judge (tau chosen on VALIDATION). Absent positive evidence the answer is NOT passed."""

    def __init__(self, tau_judge: float):
        self.tau = tau_judge

    def verify(self, ex: Example, text: str, judge_p_yes: float) -> VerifierDecision:
        tr = run_tools(ex, text)
        if tr.verdict is Verdict.FAIL:
            return VerifierDecision(False, f"tool:{tr.tool}:fail", tr)
        if tr.verdict is Verdict.PASS:
            return VerifierDecision(True, f"tool:{tr.tool}:pass", tr)
        ok = judge_p_yes >= self.tau
        return VerifierDecision(ok, "judge:pass" if ok else "judge:fail", tr)
