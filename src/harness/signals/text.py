"""Shared lightweight text utilities (no model dependencies)."""
from __future__ import annotations

import re

_WORD = re.compile(r"[A-Za-z0-9']+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0
