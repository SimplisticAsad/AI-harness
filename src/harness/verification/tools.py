"""Deterministic verification tools. A tool may only use information that would exist at deployment time:
the prompt, the model's answer, and checks derivable from them.

Honest caveats, repeated in the report:
* ``instruction``: the constraint checker IS the grader (the constraints are stated in the prompt, so a deployed system
  could run it). It is therefore an *upper bound* tool.
* ``code``: only the first two test cases are exposed to the tool as 'visible examples'; grading uses all of them.
* ``factual_qa``: no retrieval was available (network policy) -> **retrieval verification: not evaluated**.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from harness.core.types import Example


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class ToolResult:
    verdict: Verdict
    tool: str


_ARITH = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)")


def arithmetic_tool(text: str) -> ToolResult:
    """Re-computes every 'a op b = c' appearing in a reasoning chain; FAIL if any is numerically wrong."""
    found = _ARITH.findall(text)
    if not found:
        return ToolResult(Verdict.UNKNOWN, "arithmetic")
    for a, op, b, c in found:
        a, b, c = float(a), float(b), float(c)
        v = {"+": a + b, "-": a - b, "*": a * b, "x": a * b, "×": a * b, "/": a / b if b else float("nan")}[op]
        if abs(v - c) > 1e-6 * max(1.0, abs(v)):
            return ToolResult(Verdict.FAIL, "arithmetic")
    return ToolResult(Verdict.UNKNOWN, "arithmetic")  # internally consistent arithmetic does not prove the answer


def ordering_tool(prompt: str, text: str) -> ToolResult:
    """Parses 'X is <comp> than Y' premises from the PROMPT, builds the transitive closure and FAILs if the response
    asserts a relation that contradicts it (or asserts both directions)."""
    m = re.search(r"Statements:(.*?)Question", prompt, re.S)
    if not m:
        return ToolResult(Verdict.UNKNOWN, "ordering")
    comp = re.findall(r"is (\w+er) than", m.group(1))
    if not comp:
        return ToolResult(Verdict.UNKNOWN, "ordering")
    c = comp[0]
    edges = set(re.findall(rf"([A-Z][a-z]+) is {c} than ([A-Z][a-z]+)", m.group(1)))
    closure = set(edges)
    changed = True
    while changed:
        changed = False
        for a, b in list(closure):
            for b2, d in list(closure):
                if b == b2 and (a, d) not in closure:
                    closure.add((a, d)); changed = True
    for a, b in re.findall(rf"([A-Z][a-z]+) is {c} than ([A-Z][a-z]+)", text):
        if (b, a) in closure:
            return ToolResult(Verdict.FAIL, "ordering")
    return ToolResult(Verdict.UNKNOWN, "ordering")


def constraint_tool(ex: Example, text: str) -> ToolResult:
    from benchmarks.graders import check_constraint
    ok = all(check_constraint(c, text) for c in ex.reference)
    return ToolResult(Verdict.PASS if ok else Verdict.FAIL, "constraints")


def code_visible_tests_tool(ex: Example, text: str, n_visible: int = 2) -> ToolResult:
    from benchmarks.graders import run_code_tests
    from harness.signals.answers import extract_lambda
    src = extract_lambda(text)
    if src is None:
        return ToolResult(Verdict.FAIL, "code_visible_tests")
    r = run_code_tests(src, ex.reference[:n_visible])
    return ToolResult(Verdict.PASS if r == "pass" else Verdict.FAIL, "code_visible_tests")


def run_tools(ex: Example, text: str) -> ToolResult:
    t = ex.task
    if t == "math":
        return arithmetic_tool(text)
    if t == "longform":
        return ordering_tool(ex.prompt, text)
    if t == "instruction":
        return constraint_tool(ex, text)
    if t == "code":
        return code_visible_tests_tool(ex, text)
    return ToolResult(Verdict.UNKNOWN, "none")  # factual_qa / logic: no deterministic tool; retrieval not evaluated
