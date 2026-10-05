"""Distress/risk scorers: interpretable fixed-weight formula and learned classifiers, all with a common interface.

Every scorer is fit on TRAIN only; hyper-parameters / model choice are selected on VALIDATION only;
TEST is touched exactly once, by the evaluation code, for the already-selected scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from harness.scoring.feature_sets import GROUP_OF, PRIORS


class Scorer(Protocol):
    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Scorer": ...
    def risk(self, X: pd.DataFrame) -> np.ndarray: ...


class Preprocessor:
    """Median-impute + standardise with TRAIN statistics; drops constant columns. Optional per-task z-scoring."""

    def __init__(self, per_task: bool = False):
        self.per_task = per_task

    def fit(self, X: pd.DataFrame, tasks: pd.Series | None = None) -> "Preprocessor":
        self.cols = [c for c in X.columns if X[c].notna().any() and X[c].nunique(dropna=True) > 1]
        self.stats: dict[str, tuple[pd.Series, pd.Series]] = {}
        keys = tasks.unique() if (self.per_task and tasks is not None) else [None]
        for k in keys:
            sub = X[self.cols] if k is None else X.loc[tasks == k, self.cols]
            self.stats[k] = (sub.median(), sub.std(ddof=0).replace(0, 1.0).fillna(1.0))
        self.global_stats = (X[self.cols].median(), X[self.cols].std(ddof=0).replace(0, 1.0).fillna(1.0))
        return self

    def transform(self, X: pd.DataFrame, tasks: pd.Series | None = None) -> np.ndarray:
        out = np.zeros((len(X), len(self.cols)))
        if self.per_task and tasks is not None:
            for k in tasks.unique():
                m = (tasks == k).values
                med, sd = self.stats.get(k, self.global_stats)
                z = (X.loc[m, self.cols] - med) / sd
                out[m] = z.fillna(0.0).values
        else:
            med, sd = self.stats[None] if None in self.stats else self.global_stats
            out = ((X[self.cols] - med) / sd).fillna(0.0).values
        return np.clip(out, -6, 6)


@dataclass
class SklearnScorer:
    name: str
    make: object  # callable -> estimator
    per_task: bool = False
    tasks_col: str = "task"
    pre: Preprocessor = field(default=None)

    def fit(self, X: pd.DataFrame, y: np.ndarray, tasks: pd.Series | None = None) -> "SklearnScorer":
        self.pre = Preprocessor(self.per_task).fit(X, tasks)
        self.est = self.make()
        self.est.fit(self.pre.transform(X, tasks), y)
        return self

    def risk(self, X: pd.DataFrame, tasks: pd.Series | None = None) -> np.ndarray:
        Z = self.pre.transform(X, tasks)
        return self.est.predict_proba(Z)[:, 1]


class FixedWeightScorer:
    """D = sum_g w_g * A_g,  A_g = mean_i sign_i * z_i  with equal group weights w_g = 1/|G| (no fitting of weights).

    Groups with no available features for the given feature set are skipped. z_i uses TRAIN median/std.
    """

    name = "fixed_equal_weights"

    def __init__(self, groups: list[str], per_task: bool = False):
        self.groups, self.per_task = groups, per_task

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None, tasks: pd.Series | None = None) -> "FixedWeightScorer":
        self.pre = Preprocessor(self.per_task).fit(X, tasks)
        return self

    def risk(self, X: pd.DataFrame, tasks: pd.Series | None = None) -> np.ndarray:
        Z = pd.DataFrame(self.pre.transform(X, tasks), columns=self.pre.cols, index=X.index)
        parts = []
        for g in self.groups:
            pri = {c: s for c, s in PRIORS[GROUP_OF[g]].items() if c in Z.columns}
            if pri:
                parts.append(sum(s * Z[c] for c, s in pri.items()) / len(pri))
        return np.asarray(sum(parts) / max(len(parts), 1)) if parts else np.zeros(len(X))


def learner_zoo(seed: int, per_task: bool) -> dict[str, list[SklearnScorer]]:
    """Candidate scorers with small validation-tuned grids."""
    return {
        "logreg": [SklearnScorer(f"logreg_C{C}", lambda C=C: LogisticRegression(C=C, max_iter=2000), per_task)
                   for C in (0.01, 0.1, 1.0)],
        "random_forest": [SklearnScorer(f"rf_leaf{l}", lambda l=l: RandomForestClassifier(
            n_estimators=200, min_samples_leaf=l, n_jobs=1, random_state=seed), per_task) for l in (3, 10)],
        "grad_boosting": [SklearnScorer(f"hgb_d{d}", lambda d=d: HistGradientBoostingClassifier(
            max_depth=d, learning_rate=0.05, max_iter=100, random_state=seed), per_task) for d in (2, 3)],
        "mlp": [SklearnScorer(f"mlp_a{a}", lambda a=a: MLPClassifier(
            hidden_layer_sizes=(16,), alpha=a, max_iter=600, early_stopping=False, random_state=seed), per_task)
            for a in (1e-1, 1.0)],
    }


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


class Calibrator:
    """Maps a raw risk score to P(error). Fit on VALIDATION predictions only."""

    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, s: np.ndarray, y: np.ndarray) -> "Calibrator":
        if len(np.unique(y)) < 2:
            self.const = float(np.mean(y)); self.m = None
            return self
        self.const = None
        if self.kind == "platt":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y)
        elif self.kind == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
        else:
            raise KeyError(self.kind)
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        if self.const is not None:
            return np.full(len(s), self.const)
        if self.kind == "platt":
            return self.m.predict_proba(s.reshape(-1, 1))[:, 1]
        return self.m.predict(s)
