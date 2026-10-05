"""Surface linguistic features. Used both as an ablation group and as *controls* (length, verbosity)."""
from __future__ import annotations

import numpy as np

from harness.signals.base import AnswerContext
from harness.signals.text import sentences, words


def linguistic_features(text: str) -> dict[str, float]:
    toks = words(text)
    sents = sentences(text)
    n = len(toks)
    tri = [tuple(toks[i: i + 3]) for i in range(max(0, n - 2))]
    return {
        "li_n_words": float(n),
        "li_n_chars": float(len(text)),
        "li_n_sentences": float(len(sents)),
        "li_avg_sentence_len": float(n / max(len(sents), 1)),
        "li_avg_word_len": float(np.mean([len(t) for t in toks])) if toks else 0.0,
        "li_type_token_ratio": float(len(set(toks)) / n) if n else 1.0,
        "li_repeated_trigram_ratio": float(1 - len(set(tri)) / len(tri)) if tri else 0.0,
        "li_digit_ratio": float(sum(c.isdigit() for c in text) / max(len(text), 1)),
        "li_punct_ratio": float(sum((not c.isalnum()) and (not c.isspace()) for c in text) / max(len(text), 1)),
        "li_upper_ratio": float(sum(c.isupper() for c in text) / max(len(text), 1)),
    }


class LinguisticExtractor:
    name = "linguistic"
    group = "linguistic"

    def extract(self, ctx: AnswerContext) -> dict[str, float]:
        f = linguistic_features(ctx.greedy.text)
        f["li_n_tokens"] = float(ctx.greedy.n_new_tokens)
        return f
