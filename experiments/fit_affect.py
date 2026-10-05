"""Fit + validate the contextual affect model; run negation sanity probes. Saves results/affect/."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

from harness.affect.contextual import ContextualAffectExtractor, ContextualAffectModel  # noqa: E402
from harness.affect.lexicon import LexiconAffectExtractor  # noqa: E402
from harness.semantic.embedder import T5Embedder  # noqa: E402

PROBES = ["I am happy.", "I am not happy.", "This is a terrible solution.", "This is not a terrible solution.",
          "I am afraid this is wrong.", "I'm not afraid at all, this is right.", "Wonderful, that worked!",
          "The answer is 42."]

if __name__ == "__main__":
    emb = T5Embedder(".cache/hf/flan_t5_base", ".cache/spiece.model", cache_path=".cache/embeddings.npz")
    m = ContextualAffectModel(emb).fit()
    emb.save()
    ctx, lex = ContextualAffectExtractor(m), LexiconAffectExtractor()
    probes = []
    for p in PROBES:
        a = ctx.affect_of(p)
        probes.append({"text": p, "lexicon_compound": lex.score(p)["lex_compound"],
                       "contextual_valence": a["af_valence"], "contextual_nonneutral": a["af_nonneutral"]})
    Path("results/affect").mkdir(parents=True, exist_ok=True)
    json.dump({"goemotions_validation": m.report, "negation_probes": probes}, open("results/affect/affect_validation.json", "w"), indent=1)
    pickle.dump(m, open(".cache/affect_model.pkl", "wb"))
    print(json.dumps(m.report, indent=1))
    for r in probes:
        print(f"{r['text']:40s} lexicon={r['lexicon_compound']:+.2f} contextual_valence={r['contextual_valence']:+.2f}")
