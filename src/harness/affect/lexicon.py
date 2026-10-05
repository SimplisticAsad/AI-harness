"""BASELINE ONLY: word-level emotion lexicon (VADER). Each word contributes independently of context apart from
VADER's small rule set, so it is the 'WORD -> emotion score' approach the task says must not be the main algorithm."""
from __future__ import annotations

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from harness.signals.base import AnswerContext
from harness.signals.text import words


class LexiconAffectExtractor:
    name = "affect_lexicon"
    group = "affect"

    def __init__(self) -> None:
        self.sia = SentimentIntensityAnalyzer()

    def score(self, text: str) -> dict[str, float]:
        s = self.sia.polarity_scores(text)
        toks = words(text)
        emo = sum(1 for w in toks if abs(self.sia.lexicon.get(w, 0.0)) >= 1.5)
        return {"lex_pos": s["pos"], "lex_neg": s["neg"], "lex_compound": s["compound"],
                "lex_emotional_word_density": emo / max(len(toks), 1)}

    def extract(self, ctx: AnswerContext) -> dict[str, float]:
        return self.score(ctx.greedy.text)
