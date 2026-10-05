"""Contextual affect: sentence-level emotion/sentiment classifiers trained on GoEmotions over frozen encoder states.

GoEmotions (Demszky et al., 2020) provides 58k Reddit comments with 27 emotion labels + neutral. Labels are mapped to
Ekman's six emotions and to positive / negative / ambiguous sentiment using the mappings shipped with the dataset.
These are *linguistic-affect measurements of text*, not claims about model feelings. Valence/arousal/dominance
regressions need annotated VAD data that was not reachable here; arousal and dominance are therefore NOT EVALUATED.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from harness.semantic.embedder import Embedder
from harness.signals.base import AnswerContext
from harness.signals.text import sentences

EKMAN = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
SENT = ["positive", "negative", "ambiguous", "neutral"]


def _load_goemo(raw: Path, split: str, ekman: dict, sent: dict):
    emos = (raw / "goemo_emotions.txt").read_text().split()
    df = pd.read_csv(raw / f"goemo_{split}.tsv", sep="\t", header=None, names=["text", "ids", "cid"])
    e2k = {e: k for k, v in ekman.items() for e in v}
    e2s = {e: k for k, v in sent.items() for e in v}
    texts, ye, ys = [], [], []
    for t, ids in zip(df.text, df.ids):
        labs = [emos[int(i)] for i in str(ids).split(",")]
        ek = {e2k.get(l, "neutral") for l in labs}
        sn = {e2s.get(l, "neutral") for l in labs}
        if len(ek) == 1 and len(sn) == 1:  # keep unambiguous single-class rows
            texts.append(t); ye.append(next(iter(ek))); ys.append(next(iter(sn)))
    return texts, np.array(ye), np.array(ys)


class ContextualAffectModel:
    def __init__(self, embedder: Embedder, raw_dir: str | Path = "data/raw", max_train: int = 15000, seed: int = 0):
        self.embedder, self.raw, self.max_train, self.seed = embedder, Path(raw_dir), max_train, seed
        self.ekman_clf: LogisticRegression | None = None
        self.sent_clf: LogisticRegression | None = None
        self.report: dict[str, float] = {}

    def fit(self) -> "ContextualAffectModel":
        ekman = json.loads((self.raw / "goemo_ekman.json").read_text())
        sent = json.loads((self.raw / "goemo_sent.json").read_text())
        data = {s: _load_goemo(self.raw, s, ekman, sent) for s in ("train", "dev", "test")}
        tx, ye, ys = data["train"]
        rng = np.random.default_rng(self.seed)
        sel = rng.permutation(len(tx))[: self.max_train]
        X = self.embedder.embed([tx[i] for i in sel])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        self.mu, self.sd = mu, sd
        Xd = (self.embedder.embed(data["dev"][0]) - mu) / sd
        Xt = (self.embedder.embed(data["test"][0]) - mu) / sd
        Xs = (X - mu) / sd
        best = {}
        for name, ytr, ydev, yte in (("ekman", ye[sel], data["dev"][1], data["test"][1]),
                                      ("sent", ys[sel], data["dev"][2], data["test"][2])):
            scores = {}
            for C in (0.003, 0.01, 0.03, 0.1):  # tuned on GoEmotions dev only
                clf = LogisticRegression(C=C, max_iter=400, class_weight="balanced").fit(Xs, ytr)
                scores[C] = (f1_score(ydev, clf.predict(Xd), average="macro"), clf)
            C = max(scores, key=lambda c: scores[c][0])
            clf = scores[C][1]
            setattr(self, f"{name}_clf", clf)
            self.report[f"{name}_dev_macro_f1"] = float(scores[C][0])
            self.report[f"{name}_test_macro_f1"] = float(f1_score(yte, clf.predict(Xt), average="macro"))
            self.report[f"{name}_C"] = C
            self.report[f"{name}_n_test"] = int(len(yte))
        return self

    def probs(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        X = (self.embedder.embed(texts) - self.mu) / self.sd
        pe = np.zeros((len(texts), len(EKMAN)))
        pe[:, [EKMAN.index(c) for c in self.ekman_clf.classes_]] = self.ekman_clf.predict_proba(X)
        ps = np.zeros((len(texts), len(SENT)))
        ps[:, [SENT.index(c) for c in self.sent_clf.classes_]] = self.sent_clf.predict_proba(X)
        return pe, ps


class ContextualAffectExtractor:
    name = "affect_contextual"
    group = "affect"

    def __init__(self, model: ContextualAffectModel):
        self.model = model

    def affect_of(self, text: str) -> dict[str, float]:
        sents = sentences(text) or [text]
        pe, ps = self.model.probs(sents)
        out: dict[str, float] = {}
        for j, e in enumerate(EKMAN):
            out[f"af_{e}_mean"] = float(pe[:, j].mean())
            out[f"af_{e}_max"] = float(pe[:, j].max())
        for j, s in enumerate(SENT):
            out[f"af_sent_{s}_mean"] = float(ps[:, j].mean())
        out["af_valence"] = float((ps[:, 0] - ps[:, 1]).mean())  # P(positive) - P(negative)
        out["af_negative_affect"] = float(ps[:, 1].mean())
        out["af_positive_affect"] = float(ps[:, 0].mean())
        out["af_emotion_entropy"] = float(-(pe.mean(0) * np.log(pe.mean(0) + 1e-9)).sum())
        out["af_nonneutral"] = float(1.0 - pe[:, EKMAN.index("neutral")].mean())
        return out

    def extract(self, ctx: AnswerContext) -> dict[str, float]:
        return self.affect_of(ctx.greedy.text)
