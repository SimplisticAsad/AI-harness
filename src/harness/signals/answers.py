"""Label-free final-answer parsers (shared by graders, feature extractors and the verification layer)."""
from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Niaj"]


def _last_match(pattern: str, text: str, flags: int = re.I) -> str | None:
    m = list(re.finditer(pattern, text, flags))
    return m[-1].group(1) if m else None


def extract_letter(text: str) -> str | None:
    a = _last_match(r"answer\s*(?:is|:)?\s*\(?([ABCD])\b", text, 0) or _last_match(r"answer\s*(?:is|:)?\s*\(?([abcd])\b", text, 0)
    if a:
        return a.upper()
    b = re.findall(r"(?:^|[\s(])([ABCD])(?:[).:,]|\s|$)", text)
    return b[-1] if b else None


def extract_number(text: str) -> str | None:
    seg = _last_match(r"answer\s*(?:is|:)?\s*\$?\s*(-?\d[\d,]*\.?\d*)", text)
    if seg is None:
        nums = _NUM.findall(text)
        seg = nums[-1] if nums else None
    return seg.replace(",", "").rstrip(".") if seg else None


def extract_yn(text: str) -> str | None:
    a = _last_match(r"answer\s*(?:is|:)?\s*(yes|no|unknown)\b", text)
    if a:
        return a.lower()
    found = re.findall(r"\b(yes|no|unknown|cannot be determined|not possible to tell)\b", text, re.I)
    if not found:
        return None
    w = found[-1].lower()
    return "unknown" if w in ("cannot be determined", "not possible to tell") else w


def extract_name(text: str) -> str | None:
    a = _last_match(r"answer\s*(?:is|:)?\s*([A-Z][a-z]+)", text, 0)
    if a in NAMES:
        return a
    found = [w for w in re.findall(r"[A-Z][a-z]+", text) if w in NAMES]
    return found[-1] if found else None


def extract_lambda(text: str) -> str | None:
    m = re.search(r"lambda\b[^\n]*", text)
    return m.group(0).strip().rstrip(".") if m else None




def extract_answer(task: str, text: str) -> str | None:
    """Task-dispatched final-answer parser (no ground truth involved)."""
    return {"factual_qa": extract_letter, "math": extract_number, "logic": extract_yn, "longform": extract_name,
            "code": extract_lambda}.get(task, lambda t: t.strip().lower()[:200] or None)(text)
