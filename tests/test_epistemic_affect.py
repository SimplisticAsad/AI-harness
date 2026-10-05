from harness.affect.lexicon import LexiconAffectExtractor
from harness.epistemic.cues import epistemic_profile


def test_spec_examples_have_distinct_profiles():
    certain = epistemic_profile("I am certain that X.")
    believe = epistemic_profile("I believe X.")
    probably = epistemic_profile("X is probably correct.")
    may = epistemic_profile("X may be correct.")
    assert certain["booster_per100"] > 0 and certain["net_certainty"] > 0
    assert believe["speculation_per100"] > 0 and believe["net_certainty"] < 0
    assert probably["hedge_per100"] > 0 and may["hedge_per100"] > 0


def test_negation_scope_flips_certainty():
    pos = epistemic_profile("I am certain it works.")
    neg = epistemic_profile("I am not certain it works.")
    assert pos["net_certainty"] > 0 > neg["net_certainty"]
    assert neg["booster_per100"] == 0


def test_lexicon_baseline_is_word_level():
    lex = LexiconAffectExtractor()
    assert lex.score("I am happy.")["lex_compound"] > 0
    assert lex.score("This is terrible.")["lex_compound"] < 0
