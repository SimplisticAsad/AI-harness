"""Epistemic-stance extraction: hedges, boosters, speculation, qualification, assertiveness.

Cue inventories follow the linguistics literature on epistemic modality and metadiscourse
(e.g. Hyland 1998 'Hedging in Scientific Research Articles'; Holmes 1988) and were fixed *before* looking at any
experimental data (no tuning on validation/test). Unlike a bag-of-words polarity dictionary, cues are
scope-sensitive: a negator within a short window flips a booster into an uncertainty marker
("I am not certain") and a hedge into a certainty marker ("it is not possible that..." is kept as hedge-negated).
"""
from __future__ import annotations

import re

from harness.signals.base import AnswerContext
from harness.signals.text import words

HEDGES = {"may", "might", "could", "perhaps", "possibly", "probably", "likely", "seems", "seem", "appears", "appear",
          "suggests", "approximately", "about", "around", "roughly", "somewhat", "generally", "typically", "usually",
          "often", "sometimes", "presumably", "apparently", "potentially", "unlikely", "maybe", "guess", "suppose",
          "estimate", "assume", "tends", "tend"}
BOOSTERS = {"certainly", "definitely", "clearly", "obviously", "undoubtedly", "always", "never", "must", "surely",
            "absolutely", "indeed", "certain", "sure", "evidently", "inevitably", "necessarily", "exactly",
            "precisely", "undeniably", "plainly"}
SPECULATION_PHRASES = ["i think", "i believe", "i guess", "i suppose", "it is possible", "it's possible", "not sure",
                       "unsure", "unclear", "uncertain", "i don't know", "i do not know", "no idea", "cannot be sure",
                       "hard to say", "it depends", "i'm not", "i am not sure"]
QUALIFIERS = {"however", "but", "although", "though", "unless", "except", "whereas", "yet", "otherwise", "if",
              "depends", "provided", "assuming"}
ASSERTIVE_PHRASES = ["the answer is", "the answer:", "therefore", "thus", "hence", "so the", "it is clear", "in conclusion"]
AMBIGUITY = {"or", "either", "unclear", "ambiguous", "somewhere", "someone", "something", "whichever"}
NEGATORS = {"not", "no", "never", "n't", "cannot", "can't", "don't", "isn't", "aren't", "wasn't", "won't", "without"}


def _per100(count: float, n_words: int) -> float:
    return 100.0 * count / max(n_words, 1)


def epistemic_profile(text: str) -> dict[str, float]:
    low = text.lower()
    toks = words(text)
    n = len(toks)
    hedge = boost = qual = amb = neg_boost = 0
    for i, w in enumerate(toks):
        window = toks[max(0, i - 3): i]
        negated = any(x in NEGATORS or x.endswith("n't") for x in window)
        if w in HEDGES:
            hedge += 1
        elif w in BOOSTERS:
            if negated:
                neg_boost += 1  # "not certain" -> counts as uncertainty, not certainty
            else:
                boost += 1
        if w in QUALIFIERS:
            qual += 1
        if w in AMBIGUITY:
            amb += 1
    spec = sum(low.count(p) for p in SPECULATION_PHRASES) + neg_boost
    assertive = sum(low.count(p) for p in ASSERTIVE_PHRASES)
    uncertainty = hedge + spec
    # Net expressed certainty in [-1, 1]; 0 when no epistemic marking at all.
    marked = boost + uncertainty + assertive
    certainty = (boost + assertive - uncertainty) / marked if marked else 0.0
    return {
        "hedge_per100": _per100(hedge, n),
        "booster_per100": _per100(boost, n),
        "speculation_per100": _per100(spec, n),
        "qualification_per100": _per100(qual, n),
        "assertive_per100": _per100(assertive, n),
        "ambiguity_per100": _per100(amb, n),
        "uncertainty_marker_count": float(uncertainty),
        "any_hedge": float(uncertainty > 0),
        "net_certainty": float(certainty),
        "n_epistemic_markers": float(marked),
    }


class CueEpistemicExtractor:
    name = "epistemic_cues"
    group = "epistemic"

    def extract(self, ctx: AnswerContext) -> dict[str, float]:
        feats = {f"ep_{k}": v for k, v in epistemic_profile(ctx.greedy.text).items()}
        # Conflicting final answers inside one response is an epistemic-ambiguity indicator.
        nums = set(re.findall(r"-?\d+\.?\d*", ctx.greedy.text))
        feats["ep_distinct_numbers"] = float(len(nums))
        return feats
