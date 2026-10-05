"""Model interface. Any backend (HF, llama.cpp, API) can be swapped in by implementing this protocol."""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from harness.core.types import Generation, ModelInfo


@runtime_checkable
class LanguageModel(Protocol):
    info: ModelInfo

    def generate(self, prompts: Sequence[str], *, n: int = 1, temperature: float = 0.0,
                 max_new_tokens: int = 96, seed: int | None = None) -> list[list[Generation]]:
        """Return ``n`` generations per prompt. ``temperature == 0`` means greedy decoding (n must be 1)."""
        ...

    def first_token_probs(self, prompts: Sequence[str], words: Sequence[str]) -> np.ndarray:
        """Probability of each word's first token as the first generated token, renormalised over ``words``.
        Shape (len(prompts), len(words))."""
        ...
