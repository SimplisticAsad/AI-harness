"""Evaluation grid: feature sets x learners x scopes x models, plus unsupervised baselines and control models.

Protocol (leakage-safe): fit on TRAIN; choose hyper-parameters, learner family and operating threshold on VALIDATION;
fit calibrators on VALIDATION; report TEST. Test rows are never used to select anything.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from harness.evaluation import metrics as M
from harness.scoring.feature_sets import FEATURE_SETS, columns_for
from harness.scoring.learners import Calibrator, FixedWeightScorer, learner_zoo, safe_auc

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Unsupervised baselines: column, sign (+1: larger value = higher error risk). Orientations fixed a priori.
BASELINES: dict[str, tuple[str, int]] = {
    "B2_token_logprob_confidence": ("tk_mean_logprob", -1),
    "B2b_answer_token_logprob": ("tk_ans_mean_logprob", -1),
    "B3_token_entropy": ("tk_mean_entropy", +1),
    "B4_self_consistency_disagreement": ("cd_vote_frac", -1),
    "B6_semantic_disagreement": ("cd_semantic_agreement", -1),
    "B5_llm_judge_self": ("bl_judge_self", -1),
    "B5b_llm_judge_independent(flan_t5_large)": ("bl_judge_flan_t5_large", -1),
    "lexicon_hedge_density": ("ep_hedge_per100", +1),
    "lexicon_word_emotion(VADER_negativity)": ("lex_neg", +1),
}
CONTROL_COLS = ["ctl_q_words", "ctl_n_tokens", "difficulty", "li_n_words", "li_n_chars"]


@dataclass
class StatCfg:
    B: int = 1000
    seed: int = 0


def with_task_dummies(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.get_dummies(df["task"], prefix="ctl_task").astype(float)
    return pd.concat([df.drop(columns=[c for c in df.columns if c.startswith("ctl_task_")], errors="ignore"), d], axis=1)


def eval_scores(y_va, s_va, y_te, s_te, sc: StatCfg, groups_te=None, full: bool = True) -> dict[str, float]:
    """Metrics for one scorer. Calibration (Platt on VAL) -> Brier/ECE on TEST. Threshold chosen on VAL."""
    out: dict[str, float] = {}
    a, lo, hi = M.bootstrap(M.auroc, [y_te, s_te], sc.B, sc.seed, groups_te)
    out.update(auroc=a, auroc_lo=lo, auroc_hi=hi)
    a, lo, hi = M.bootstrap(M.auprc, [y_te, s_te], sc.B, sc.seed, groups_te)
    out.update(auprc=a, auprc_lo=lo, auprc_hi=hi)
    out.update(M.correlations(y_te, s_te))
    cal = Calibrator("platt").fit(s_va, y_va)
    p = cal.predict(s_te)
    out.update(brier=M.brier(y_te, p), ece=M.ece(y_te, p))
    if full:
        iso = Calibrator("isotonic").fit(s_va, y_va).predict(s_te)
        out.update(ece_isotonic=M.ece(y_te, iso), brier_isotonic=M.brier(y_te, iso))
        t = M.choose_threshold(y_va, s_va)
        flag = (s_te >= t).astype(int)
        out.update(M.prf(y_te, flag))
        out.update(M.decision_rates(y_te, flag))
        out["threshold"] = t
        out["aurc"] = M.risk_coverage(y_te, s_te)["aurc"]
        out["risk_at_cov80"] = M.risk_at_coverage(y_te, s_te, 0.8)
        out["risk_at_cov50"] = M.risk_at_coverage(y_te, s_te, 0.5)
    return out


def _fit_families(Xtr, ytr, Xva, yva, Xte, ttr, tva, tte, per_task, seed, fixed_groups):
    """Return {family: (val_auc, s_val, s_test, chosen_name)}."""
    res = {}
    if len(np.unique(ytr)) < 2 or Xtr.shape[1] == 0:
        return res
    if fixed_groups:
        fx = FixedWeightScorer(fixed_groups, per_task).fit(Xtr, ytr, ttr)
        s_va, s_te = fx.risk(Xva, tva), fx.risk(Xte, tte)
        res["fixed_equal_weights"] = (safe_auc(yva, s_va), s_va, s_te, "fixed")
    for fam, cands in learner_zoo(seed, per_task).items():
        best = None
        for sc in cands:
            try:
                sc.fit(Xtr, ytr, ttr)
                s_va = sc.risk(Xva, tva)
            except Exception:  # noqa: BLE001
                continue
            a = safe_auc(yva, s_va)
            if best is None or np.nan_to_num(a, nan=-1) > np.nan_to_num(best[0], nan=-1):
                best = (a, sc, s_va)
        if best is not None:
            a, sc, s_va = best
            res[fam] = (a, s_va, sc.risk(Xte, tte), sc.name)
    return res


def run_scope(df: pd.DataFrame, model: str, scope: str, sc: StatCfg, seed: int = 0,
              feature_sets: dict[str, list[str]] | None = None) -> tuple[list[dict], list[pd.DataFrame]]:
    """All feature sets + baselines + controls for one (model, scope)."""
    feature_sets = feature_sets or FEATURE_SETS
    d = df[df.model == model]
    d = d if scope == "pooled" else d[d.task == scope]
    d = with_task_dummies(d)
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    if len(te) < 10 or te.error.nunique() < 2:
        return [], []
    per_task = scope == "pooled"
    ytr, yva, yte = tr.error.values, va.error.values, te.error.values
    rows, preds = [], []
    gte = te.id.values

    def record(method: str, family: str, s_va, s_te, extra: dict | None = None):
        m = eval_scores(yva, s_va, yte, s_te, sc, gte)
        rows.append({"model": model, "scope": scope, "method": method, "family": family, "n_test": len(te),
                     "base_error_rate": float(yte.mean()), "val_auroc": safe_auc(yva, s_va), **m, **(extra or {})})
        p_platt = Calibrator("platt").fit(s_va, yva).predict(s_te)
        p_iso = Calibrator("isotonic").fit(s_va, yva).predict(s_te)
        preds.append(pd.DataFrame({"model": model, "scope": scope, "method": method, "family": family,
                                   "id": te.id.values, "task": te.task.values, "y": yte, "score": s_te,
                                   "p_platt": p_platt, "p_iso": p_iso, "threshold": m["threshold"]}))

    # --- unsupervised baselines
    for name, (col, sign) in BASELINES.items():
        if col in d.columns and d[col].notna().any():
            fill = tr[col].median()
            record(name, "baseline", sign * va[col].fillna(fill).values, sign * te[col].fillna(fill).values)
    rng = np.random.default_rng(seed)
    record("random_scores", "control", rng.random(len(va)), rng.random(len(te)))
    record("B1_base_llm_accept_all", "baseline", np.zeros(len(va)), np.zeros(len(te)))

    # --- learned feature sets
    for fs_name, keys in feature_sets.items():
        cols = columns_for(d, keys)
        if not cols:
            continue
        groups = [k for k in keys if k in ("A", "B", "C", "D", "E")]
        fam = _fit_families(tr[cols], ytr, va[cols], yva, te[cols], tr.task, va.task, te.task, per_task, seed,
                            groups if (groups and fs_name != "A_lexicon_baseline") else None)
        if not fam:
            continue
        sel = max(fam, key=lambda f: np.nan_to_num(fam[f][0], nan=-1))
        for f, (vauc, s_va, s_te, nm) in fam.items():
            record(fs_name, f, s_va, s_te, {"chosen": nm, "selected_on_val": f == sel})

    # --- controls: confounds-only, and ESD core + confounds
    ctl = [c for c in CONTROL_COLS if c in d.columns] + [c for c in d.columns if c.startswith("ctl_task_")]
    for name, extra_keys in (("CTRL_length+difficulty+task", None), ("ESD_core+CTRL", ["A", "B", "C", "D", "E"])):
        cols = ctl + (columns_for(d, extra_keys) if extra_keys else [])
        fam = _fit_families(tr[cols], ytr, va[cols], yva, te[cols], tr.task, va.task, te.task, per_task, seed, None)
        if fam:
            sel = max(fam, key=lambda f: np.nan_to_num(fam[f][0], nan=-1))
            for f, (vauc, s_va, s_te, nm) in fam.items():
                record(name, f, s_va, s_te, {"chosen": nm, "selected_on_val": f == sel})
    if scope == "pooled":  # task base-rate prior only
        prior = tr.groupby("task").error.mean()
        record("CTRL_task_prior_only", "control", va.task.map(prior).fillna(ytr.mean()).values,
               te.task.map(prior).fillna(ytr.mean()).values)
    return rows, preds


def run_grid(df: pd.DataFrame, sc: StatCfg, progress: Callable[[str], None] = print) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows, all_preds = [], []
    for model in df.model.unique():
        for scope in ["pooled"] + sorted(df[df.model == model].task.unique()):
            progress(f"{model} / {scope}")
            r, p = run_scope(df, model, scope, sc)
            all_rows += r
            all_preds += p
    return pd.DataFrame(all_rows), pd.concat(all_preds, ignore_index=True)
