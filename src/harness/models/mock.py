"""Deterministic mock model for unit/integration tests (no weights needed)."""
from __future__ import annotations

import hashlib
from typing import Callable, Sequence

import numpy as np

from harness.core.types import Generation, ModelInfo, TokenTrace


class MockModel:
    def __init__(self, responder: Callable[[str, int], str] | None = None):
        self.info = ModelInfo("mock", "mock", 0.0, "none", "python")
        self.responder = responder or (lambda p, k: f"Let's think. The answer: {int(hashlib.md5(p.encode()).hexdigest(), 16) % 4}")

    def generate(self, prompts: Sequence[str], *, n: int = 1, temperature: float = 0.0,
                 max_new_tokens: int = 96, seed: int | None = None) -> list[list[Generation]]:
        out = []
        for p in prompts:
            row = []
            for k in range(n):
                text = self.responder(p, k if temperature > 0 else -1)
                toks = text.split()
                rng = np.random.default_rng(abs(hash((p, k))) % (2**32))
                lp = (-rng.random(len(toks))).tolist()
                tr = TokenTrace(toks, lp, (rng.random(len(toks)) + 0.1).tolist(), np.exp(lp).tolist())
                row.append(Generation(text, tr, 0.001, len(toks), len(p.split())))
            out.append(row)
        return out

    def first_token_probs(self, prompts: Sequence[str], words: Sequence[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        x = rng.random((len(prompts), len(words)))
        return x / x.sum(1, keepdims=True)
