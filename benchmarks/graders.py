"""Deterministic graders. Ground truth comes from dataset labels, arithmetic, solver logic or code execution -
never from an LLM."""
from __future__ import annotations

import json
import multiprocessing as mp
import re
from typing import Any

from harness.signals.answers import (extract_lambda, extract_letter, extract_name, extract_number,  # noqa: F401
                                      extract_yn)
from harness.core.types import Example, Grade

# ---------------------------------------------------------------- instruction constraints (also usable as a deployable tool)
def check_constraint(c: dict[str, Any], text: str) -> bool:
    t = text.strip()
    words = re.findall(r"[A-Za-z']+", t)
    k = c["kind"]
    if k == "min_words":
        return len(words) >= c["n"]
    if k == "max_words":
        return len(words) < c["n"]
    if k == "keywords":
        low = t.lower()
        return all(w in low for w in c["words"])
    if k == "lowercase":
        return t == t.lower() and len(t) > 0
    if k == "no_commas":
        return "," not in t and len(t) > 0
    if k == "end_phrase":
        return t.endswith(c["phrase"])
    if k == "n_sentences":
        return len([s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]) == c["n"]
    if k == "quotes":
        return len(t) > 1 and t[0] == '"' and t[-1] == '"'
    raise KeyError(k)


# ---------------------------------------------------------------- code execution (sandboxed subprocess, timeout)
def _run_lambda(src: str, tests: list[list[str]], q: "mp.Queue") -> None:
    try:
        f = eval(src, {"__builtins__": {"sum": sum, "len": len, "max": max, "min": min, "sorted": sorted, "abs": abs,
                                         "range": range, "list": list, "str": str, "int": int, "float": float,
                                         "round": round, "reversed": reversed, "all": all, "any": any, "set": set,
                                         "tuple": tuple, "enumerate": enumerate, "zip": zip, "map": map,
                                         "filter": filter, "bool": bool}})
        ok = all(f(json.loads(a)) == json.loads(b) for a, b in tests)
        q.put("pass" if ok else "wrong")
    except SyntaxError:
        q.put("syntax")
    except Exception:  # noqa: BLE001
        q.put("runtime")


def run_code_tests(src: str, tests: list[list[str]], timeout: float = 3.0) -> str:
    q: "mp.Queue" = mp.Queue()
    p = mp.Process(target=_run_lambda, args=(src, tests, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        return "timeout"
    return q.get() if not q.empty() else "runtime"


# ---------------------------------------------------------------- order consistency
def ordering_violations(ex: Example, text: str) -> int:
    order: list[str] = ex.meta["order"]
    rank = {n: i for i, n in enumerate(order)}
    comp = ex.meta["comparative"]
    bad = 0
    for a, b in re.findall(rf"([A-Z][a-z]+) is {comp} than ([A-Z][a-z]+)", text):
        if a in rank and b in rank and rank[a] > rank[b]:
            bad += 1
    return bad


# ---------------------------------------------------------------- dispatcher
def grade(ex: Example, text: str) -> Grade:
    t = ex.task
    if t == "factual_qa":
        a = extract_letter(text)
        if a is None:
            return Grade(False, None, "incomplete_answer", "no letter extracted")
        return Grade(a == ex.reference, a, None if a == ex.reference else "factual_hallucination")
    if t == "math":
        a = extract_number(text)
        if a is None:
            return Grade(False, None, "incomplete_answer", "no number extracted")
        try:
            ok = abs(float(a) - float(ex.reference)) < 1e-6
        except ValueError:
            return Grade(False, a, "incomplete_answer", "unparseable number")
        return Grade(ok, a, None if ok else "mathematical_error")
    if t == "logic":
        a = extract_yn(text)
        if a is None:
            return Grade(False, None, "incomplete_answer", "no yes/no/unknown extracted")
        return Grade(a == ex.reference, a, None if a == ex.reference else "logical_error")
    if t == "longform":
        a = extract_name(text)
        if a is None:
            return Grade(False, None, "incomplete_answer", "no name extracted")
        viol = ordering_violations(ex, text)
        if a != ex.reference:
            return Grade(False, a, "contradiction" if viol else "reasoning_error", f"violations={viol}")
        if viol:  # right conclusion but stated a relation contradicting the premises -> not valid derivation
            return Grade(False, a, "reasoning_error", f"conclusion right, {viol} false relation(s) asserted")
        return Grade(True, a, None)
    if t == "instruction":
        failed = [c["kind"] for c in ex.reference if not check_constraint(c, text)]
        return Grade(not failed, None, None if not failed else "instruction_violation", ",".join(failed))
    if t == "code":
        src = extract_lambda(text)
        if src is None:
            return Grade(False, None, "incomplete_answer", "no lambda found")
        r = run_code_tests(src, ex.reference)
        return Grade(r == "pass", src, None if r == "pass" else "code_failure", r)
    raise KeyError(t)
