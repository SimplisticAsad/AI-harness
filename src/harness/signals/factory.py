"""Composition root for the extractor stack (dependency injection happens here, nowhere else)."""
from __future__ import annotations

import pickle
from pathlib import Path

from harness.affect.contextual import ContextualAffectExtractor
from harness.affect.lexicon import LexiconAffectExtractor
from harness.epistemic.cues import CueEpistemicExtractor
from harness.semantic.consistency import CandidateAgreementExtractor, SemanticExtractor
from harness.semantic.embedder import T5Embedder
from harness.signals.linguistic import LinguisticExtractor
from harness.uncertainty.token import TokenUncertaintyExtractor


def build_extractors(encoder_dir: str = ".cache/hf/flan_t5_base", spiece: str = ".cache/spiece.model",
                     affect_pkl: str = ".cache/affect_model.pkl", cache: str = ".cache/embeddings.npz"):
    emb = T5Embedder(encoder_dir, spiece, cache_path=cache)
    affect_model = pickle.load(open(affect_pkl, "rb"))
    affect_model.embedder = emb  # re-attach the live encoder
    ctx_aff = ContextualAffectExtractor(affect_model)
    extractors = [LexiconAffectExtractor(), ctx_aff, CueEpistemicExtractor(), SemanticExtractor(emb),
                  TokenUncertaintyExtractor(),
                  CandidateAgreementExtractor(emb, affect_fn=lambda t: ctx_aff.affect_of(t)["af_valence"]),
                  LinguisticExtractor()]
    return extractors, emb
