"""LLM-as-a-judge: P(model says 'yes') to 'Is the proposed answer correct?'. Used as a baseline detector and as the
independent judge in the verification stage. The judge never sees ground truth."""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from harness.models.base import LanguageModel

TEMPLATE = ("{prompt}\n\nProposed answer: {answer}\n\nIs the proposed answer correct? "
            "OPTIONS:\n- yes\n- no")


class LLMJudge:
    def __init__(self, model: LanguageModel, name: str):
        self.model, self.name = model, name

    def p_correct(self, prompts: Sequence[str], answers: Sequence[str]) -> tuple[np.ndarray, float]:
        """Returns (P(yes) per item, seconds per item)."""
        texts = [TEMPLATE.format(prompt=p, answer=a[:400]) for p, a in zip(prompts, answers)]
        t0 = time.perf_counter()
        probs = self.model.first_token_probs(texts, ["yes", "no"])
        return probs[:, 0], (time.perf_counter() - t0) / max(len(texts), 1)
