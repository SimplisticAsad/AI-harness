"""Higher-level analyses on top of the evaluation grid: paired comparisons, controls, generalisation, error types."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from harness.evaluation import metrics as M
from harness.evaluation import stattests as T
from harness.evaluation.runner import CONTROL_COLS, StatCfg, with_task_dummies
from harness.scoring.feature_sets import FEATURE_SETS, columns_for
from harness.scoring.learners import Preprocessor

ESD = "ESD_core(A+B+C+D+E)"


def aligned(P: pd.DataFrame, model: str, scope: str, m1: str, f1: str, m2: str, f2: str):
    a = P[(P.model == model) & (P.scope == scope) & (P.method == m1) & (P.family == f1)].set_index("id")
    b = P[(P.model == model) & (P.scope == scope) & (P.method == m2) & (P.family == f2)].set_index("id")
    ids = a.index.intersection(b.index)
    if len(ids) < 10:
        return None
    return a.loc[ids, "y"].values, a.loc[ids, "score"].values, b.loc[ids, "score"].values


# ------------------------------------------------------------------------------------------------ H4 / ablation
def paired_comparisons(P: pd.DataFrame, sc: StatCfg, ref: str = ESD, family: str = "logreg",
                       others: list[str] | None = None) -> pd.DataFrame:
    """ESD_core vs each single signal family / sub-combination, same learner (logreg) -> isolates the features."""
    others = others or ["A_affective", "B_epistemic", "C_semantic", "D_token", "E_candidate", "F_affect+epistemic",
                        "G_affect+semantic", "H_epistemic+semantic", "ESD_text_only(A+B+C)"]
    rows = []
    for model in P.model.unique():
        for scope in P[P.model == model].scope.unique():
            for o in others:
                al = aligned(P, model, scope, ref, family, o, family)
                if al is None:
                    continue
                y, s1, s2 = al
                d, lo, hi = M.paired_bootstrap_diff(M.auroc, y, s1, s2, sc.B, sc.seed)
                rows.append({"model": model, "scope": scope, "reference": ref, "comparator": o, "n": len(y),
                             "auroc_ref": M.auroc(y, s1), "auroc_cmp": M.auroc(y, s2), "delta_auroc": d,
                             "delta_lo": lo, "delta_hi": hi,
                             "p_perm": T.permutation_auc_diff(y, s1, s2, 1000, sc.seed)})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = T.holm(df.p_perm.tolist())
    return df


def vs_chance(P: pd.DataFrame, sc: StatCfg, family_pref: str = "logreg") -> pd.DataFrame:
    """Permutation test of H0: AUROC = 0.5 for every (model, scope, method) with the given learner family or baseline."""
    rows = []
    d = P[(P.family == family_pref) | (P.family == "baseline") | (P.family == "fixed_equal_weights")]
    for (model, scope, method, fam), g in d.groupby(["model", "scope", "method", "family"]):
        if g.y.nunique() < 2 or g.score.nunique() < 2:
            continue
        rows.append({"model": model, "scope": scope, "method": method, "family": fam, "n": len(g),
                     "auroc": M.auroc(g.y.values, g.score.values),
                     "p_vs_chance": T.permutation_auc_vs_chance(g.y.values, g.score.values, 500, sc.seed)})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = T.holm(df.p_vs_chance.tolist())
    return df


def wilcoxon_across_cells(Mx: pd.DataFrame, a: str, b: str, family_a: str = "logreg", family_b: str = "logreg") -> dict:
    """Paired Wilcoxon over (model, task) cells of test AUROC for method a vs b."""
    x = Mx[(Mx.scope != "pooled") & (Mx.method == a) & (Mx.family == family_a)].set_index(["model", "scope"]).auroc
    y = Mx[(Mx.scope != "pooled") & (Mx.method == b) & (Mx.family == family_b)].set_index(["model", "scope"]).auroc
    idx = x.index.intersection(y.index)
    p, med = T.wilcoxon(x[idx].values, y[idx].values)
    return {"a": a, "b": b, "n_cells": len(idx), "median_delta_auroc": med, "p_wilcoxon": p}


# ------------------------------------------------------------------------------------------------ controls
def _rank_resid(v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    r = stats.rankdata(v)
    beta, *_ = np.linalg.lstsq(Z, r, rcond=None)
    return r - Z @ beta


def partial_spearman(df_test: pd.DataFrame, risk: np.ndarray, sc: StatCfg) -> dict:
    """Spearman(risk, error) after removing (linearly, on ranks) task, answer length, question length, difficulty."""
    d = with_task_dummies(df_test.assign(risk=risk))
    ctl = [c for c in d.columns if c.startswith("ctl_task_")]
    Z = np.column_stack([np.ones(len(d))] + [d[c].values for c in ctl] +
                        [stats.rankdata(d[c].fillna(d[c].median()).values) for c in ("ctl_q_words", "ctl_n_tokens", "difficulty")])
    rr, ry = _rank_resid(d.risk.values, Z), _rank_resid(d.error.values.astype(float), Z)
    r = float(np.corrcoef(rr, ry)[0, 1])
    rng = np.random.default_rng(sc.seed)
    perm = [np.corrcoef(rr, rng.permutation(ry))[0, 1] for _ in range(1000)]
    raw = M.correlations(d.error.values, d.risk.values)["spearman"]
    return {"raw_spearman": raw, "partial_spearman": r, "p_partial": float((np.sum(np.abs(perm) >= abs(r)) + 1) / 1001)}


def length_strata(df_test: pd.DataFrame, risk: np.ndarray) -> pd.DataFrame:
    """AUROC within (task x answer-length tercile): is ESD just a length detector?"""
    d = df_test.assign(risk=risk).copy()
    rows = []
    for t, g in d.groupby("task"):
        try:
            g = g.assign(terc=pd.qcut(g.ctl_n_tokens.rank(method="first"), 3, labels=["short", "mid", "long"]))
        except ValueError:
            continue
        for terc, h in g.groupby("terc", observed=True):
            rows.append({"task": t, "length_tercile": terc, "n": len(h), "error_rate": h.error.mean(),
                         "auroc": M.auroc(h.error.values, h.risk.values)})
    return pd.DataFrame(rows)


def lr_fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], C: float = 0.1, per_task: bool = True,
                   shuffle_seed: int | None = None) -> np.ndarray:
    pre = Preprocessor(per_task).fit(train[cols], train.task)
    y = train.error.values.copy()
    if shuffle_seed is not None:
        y = np.random.default_rng(shuffle_seed).permutation(y)
    clf = LogisticRegression(C=C, max_iter=2000).fit(pre.transform(train[cols], train.task), y)
    return clf.predict_proba(pre.transform(test[cols], test.task))[:, 1]


def placebo(df: pd.DataFrame, model: str, n_shuffles: int = 30) -> dict:
    """Label-shuffled training -> AUROC on true test labels. Gives the null distribution for the same pipeline."""
    d = df[df.model == model]
    cols = columns_for(d, FEATURE_SETS[ESD])
    tr, te = d[d.split == "train"], d[d.split == "test"]
    real = M.auroc(te.error.values, lr_fit_predict(tr, te, cols))
    null = [M.auroc(te.error.values, lr_fit_predict(tr, te, cols, shuffle_seed=i)) for i in range(n_shuffles)]
    return {"model": model, "real_auroc": real, "null_mean": float(np.nanmean(null)), "null_p95": float(np.nanpercentile(null, 95)),
            "null_p99": float(np.nanpercentile(null, 99)), "n_shuffles": n_shuffles}


# ------------------------------------------------------------------------------------------------ generalisation
def leave_one_task_out(df: pd.DataFrame, sc: StatCfg, keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or FEATURE_SETS[ESD]
    rows = []
    for model in df.model.unique():
        d = df[df.model == model]
        cols = columns_for(d, keys)
        for held in sorted(d.task.unique()):
            tr = d[(d.task != held) & (d.split.isin(["train", "val"]))]
            te = d[(d.task == held) & (d.split == "test")]
            if te.error.nunique() < 2 or tr.error.nunique() < 2:
                continue
            s = lr_fit_predict(tr, te, cols)
            a, lo, hi = M.bootstrap(M.auroc, [te.error.values, s], sc.B, sc.seed)
            # within-task trained reference (train split of the same task)
            tr_in = d[(d.task == held) & (d.split.isin(["train", "val"]))]
            s_in = lr_fit_predict(tr_in, te, cols, per_task=False) if tr_in.error.nunique() > 1 else np.zeros(len(te))
            rows.append({"model": model, "held_out_task": held, "n_test": len(te), "auroc_transfer": a, "lo": lo, "hi": hi,
                         "auroc_in_task": M.auroc(te.error.values, s_in), "error_rate": te.error.mean()})
    return pd.DataFrame(rows)


def cross_model(df: pd.DataFrame, sc: StatCfg, keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or FEATURE_SETS[ESD]
    models = list(df.model.unique())
    cols = columns_for(df, keys)
    rows = []
    for a in models:
        for b in models:
            tr = df[(df.model == a) & df.split.isin(["train", "val"])]
            te = df[(df.model == b) & (df.split == "test")]
            if te.error.nunique() < 2:
                continue
            s = lr_fit_predict(tr, te, cols)
            v, lo, hi = M.bootstrap(M.auroc, [te.error.values, s], sc.B, sc.seed)
            rows.append({"train_model": a, "test_model": b, "n_test": len(te), "auroc": v, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ error taxonomy
def error_type_detectability(df: pd.DataFrame, P: pd.DataFrame, sc: StatCfg, method: str = ESD, family: str = "logreg",
                             baselines: tuple[str, ...] = ("B3_token_entropy", "B5_llm_judge_self")) -> pd.DataFrame:
    """AUROC separating correct answers from each error type (pooled scope, test split)."""
    rows = []
    for model in P.model.unique():
        for name, fam in [(method, family)] + [(b, "baseline") for b in baselines]:
            g = P[(P.model == model) & (P.scope == "pooled") & (P.method == name) & (P.family == fam)]
            if g.empty:
                continue
            g = g.merge(df[df.model == model][["id", "error_type"]], on="id", how="left")
            correct = g[g.y == 0]
            for et, h in g[g.y == 1].groupby("error_type"):
                if len(h) < 8:
                    continue
                yy = np.r_[np.zeros(len(correct)), np.ones(len(h))]
                ss = np.r_[correct.score.values, h.score.values]
                v, lo, hi = M.bootstrap(M.auroc, [yy, ss], sc.B, sc.seed)
                rows.append({"model": model, "scorer": name, "error_type": et, "n_errors": len(h),
                             "n_correct": len(correct), "auroc": v, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def calibration_mismatch(df: pd.DataFrame, P: pd.DataFrame, model: str, method: str = ESD, family: str = "logreg") -> dict:
    """Over/under-confidence in *expressed* stance vs ESD risk: wrong answers that read as certain, correct ones that hedge."""
    g = P[(P.model == model) & (P.scope == "pooled") & (P.method == method) & (P.family == family)]
    g = g.merge(df[df.model == model][["id", "ep_net_certainty", "ep_any_hedge", "tk_mean_logprob"]], on="id")
    wrong, right = g[g.y == 1], g[g.y == 0]
    return {"model": model,
            "overconfident_wrong_share(net_certainty>0 | wrong)": float((wrong.ep_net_certainty > 0).mean()) if len(wrong) else np.nan,
            "hedged_correct_share(any_hedge | correct)": float((right.ep_any_hedge > 0).mean()) if len(right) else np.nan,
            "hedged_wrong_share(any_hedge | wrong)": float((wrong.ep_any_hedge > 0).mean()) if len(wrong) else np.nan,
            "n_wrong": len(wrong), "n_correct": len(right)}
