import numpy as np
import pandas as pd

from benchmarks.generators import build_all
from benchmarks.graders import grade
from harness.core.io import read_jsonl, split_of, write_jsonl
from harness.core.pipeline import generate_records
from harness.epistemic.cues import CueEpistemicExtractor
from harness.evaluation.runner import StatCfg, run_scope
from harness.models.base import LanguageModel
from harness.models.mock import MockModel
from harness.scoring.dataset import add_anomaly, build_frame
from harness.semantic.consistency import CandidateAgreementExtractor, SemanticExtractor
from harness.signals.linguistic import LinguisticExtractor
from harness.signals.pipeline import extract_features
from harness.uncertainty.token import TokenUncertaintyExtractor


def test_mock_satisfies_model_protocol():
    assert isinstance(MockModel(), LanguageModel)


def test_split_is_deterministic_and_disjoint():
    ids = [f"x/{i}" for i in range(2000)]
    sp = {i: split_of(i) for i in ids}
    assert sp == {i: split_of(i) for i in ids}
    frac = pd.Series(sp).value_counts(normalize=True)
    assert abs(frac["train"] - 0.6) < 0.05 and abs(frac["test"] - 0.2) < 0.05


def test_feature_extractors_never_see_labels():
    from harness.signals.base import AnswerContext
    assert "grade" not in AnswerContext.__dataclass_fields__ and "reference" not in AnswerContext.__dataclass_fields__


def test_end_to_end_mock(tmp_path, emb):
    ex = build_all({"logic": 150}, 5)
    import hashlib
    model = MockModel(lambda p, k: "Because of this. The answer is " + ["yes", "no", "unknown"][int(hashlib.md5(f"{p}{k}".encode()).hexdigest(), 16) % 3])
    gp = tmp_path / "m" / "generations.jsonl"
    generate_records(model, ex, gp, grade, n_samples=3, temperature=0.7, max_new_tokens=20, seed=1, chunk=8)
    assert len(read_jsonl(gp)) == 150
    generate_records(model, ex, gp, grade, n_samples=3, temperature=0.7, max_new_tokens=20, seed=1, chunk=8)
    assert len(read_jsonl(gp)) == 150  # resumable, no duplicates
    recs = read_jsonl(gp)
    ext = [CueEpistemicExtractor(), SemanticExtractor(emb), CandidateAgreementExtractor(emb), TokenUncertaintyExtractor(),
           LinguisticExtractor()]
    write_jsonl(tmp_path / "m" / "features.jsonl", extract_features(recs, ext))
    df = add_anomaly(build_frame(tmp_path / "m"))
    assert {"error", "split", "tk_mean_entropy", "cd_vote_frac", "ep_hedge_per100"} <= set(df.columns)
    rows, preds = run_scope(df, "mock", "pooled", StatCfg(B=20),
                            feature_sets={"B_epistemic": ["B"], "D_token": ["D"], "E_candidate": ["E"]})
    assert rows and {"auroc", "ece", "false_acceptance_rate"} <= set(rows[0])
    # test-split predictions only: nothing from train/val is reported as test
    assert all(preds[0].id.map(split_of) == "test")
