"""Error-detection, calibration and selective-prediction metrics. Positive class = 'answer is incorrect'."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [{"bin": b, "n": int((idx == b).sum()), "mean_pred": float(p[idx == b].mean()),
             "frac_pos": float(y[idx == b].mean())} for b in range(bins) if (idx == b).any()]


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    rel = reliability(y, p, bins)
    n = len(y)
    return float(sum(r["n"] / n * abs(r["mean_pred"] - r["frac_pos"]) for r in rel))


def correlations(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) < 2 or np.std(s) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {"pearson": float(stats.pearsonr(s, y)[0]), "spearman": float(stats.spearmanr(s, y)[0])}


def prf(y: np.ndarray, flag: np.ndarray) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y, flag, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f)}


def decision_rates(y: np.ndarray, flag: np.ndarray) -> dict[str, float]:
    """flag=1 -> reject/verify. False acceptance = P(accept | incorrect). False rejection = P(reject | correct)."""
    wrong, right = y == 1, y == 0
    return {
        "false_acceptance_rate": float((~flag.astype(bool))[wrong].mean()) if wrong.any() else float("nan"),
        "false_rejection_rate": float(flag.astype(bool)[right].mean()) if right.any() else float("nan"),
        "accepted_error_rate": float(y[~flag.astype(bool)].mean()) if (~flag.astype(bool)).any() else float("nan"),
        "coverage": float((~flag.astype(bool)).mean()),
    }


def risk_coverage(y: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray | float]:
    """Accept lowest-risk first. Returns coverage, selective risk (error among accepted) and AURC."""
    order = np.argsort(s, kind="stable")
    ye = y[order]
    k = np.arange(1, len(y) + 1)
    risk = np.cumsum(ye) / k
    cov = k / len(y)
    return {"coverage": cov, "risk": risk, "aurc": float(risk.mean())}


def risk_at_coverage(y: np.ndarray, s: np.ndarray, cov: float) -> float:
    rc = risk_coverage(y, s)
    i = max(int(round(cov * len(y))) - 1, 0)
    return float(rc["risk"][i])


def choose_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximising F1 for flagging errors on VALIDATION data (never on test)."""
    cands = np.unique(np.quantile(s, np.linspace(0.05, 0.95, 37)))
    best, best_f = cands[0], -1.0
    for t in cands:
        f = prf(y, (s >= t).astype(int))["f1"]
        if f > best_f:
            best, best_f = t, f
    return float(best)


def bootstrap(stat: Callable[..., float], arrays: list[np.ndarray], B: int = 1000, seed: int = 0,
              groups: np.ndarray | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap CI (95%). ``groups`` enables a cluster bootstrap (resample group ids)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    point = stat(*arrays)
    vals = []
    if groups is None:
        for _ in range(B):
            i = rng.integers(0, n, n)
            v = stat(*[a[i] for a in arrays])
            if not np.isnan(v):
                vals.append(v)
    else:
        ug = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in ug}
        for _ in range(B):
            pick = rng.choice(ug, len(ug))
            i = np.concatenate([members[g] for g in pick])
            v = stat(*[a[i] for a in arrays])
            if not np.isnan(v):
                vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(stat: Callable[[np.ndarray, np.ndarray], float], y: np.ndarray, s1: np.ndarray,
                          s2: np.ndarray, B: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    d0 = stat(y, s1) - stat(y, s2)
    ds = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v = stat(y[i], s1[i]) - stat(y[i], s2[i])
        if not np.isnan(v):
            ds.append(v)
    return float(d0), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
