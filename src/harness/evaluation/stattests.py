"""Hypothesis tests used in the study (each is named in the report with its rationale)."""
from __future__ import annotations

import numpy as np
from scipy import stats

from harness.evaluation.metrics import auroc


def permutation_auc_diff(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, n_perm: int = 2000, seed: int = 0) -> float:
    """Paired permutation test for H0: AUROC(s1) == AUROC(s2) on the same items. Under H0 the two scores are
    exchangeable within each item, so we swap them item-wise at random. Two-sided p-value."""
    rng = np.random.default_rng(seed)
    obs = abs(auroc(y, s1) - auroc(y, s2))
    cnt = 0
    for _ in range(n_perm):
        sw = rng.random(len(y)) < 0.5
        a, b = np.where(sw, s2, s1), np.where(sw, s1, s2)
        cnt += abs(auroc(y, a) - auroc(y, b)) >= obs - 1e-12
    return float((cnt + 1) / (n_perm + 1))


def permutation_auc_vs_chance(y: np.ndarray, s: np.ndarray, n_perm: int = 2000, seed: int = 0) -> float:
    """H0: score carries no information about the label (AUROC = 0.5). One-sided (AUROC > 0.5)."""
    rng = np.random.default_rng(seed)
    obs = auroc(y, s)
    cnt = sum(auroc(rng.permutation(y), s) >= obs for _ in range(n_perm))
    return float((cnt + 1) / (n_perm + 1))


def mcnemar_exact(wrong_a: np.ndarray, wrong_b: np.ndarray) -> tuple[float, int, int]:
    """McNemar exact test on paired binary outcomes (e.g. 'incorrect answer was accepted' under policy A vs B).
    Returns (p, n_a_only, n_b_only)."""
    b = int(((wrong_a == 1) & (wrong_b == 0)).sum())
    c = int(((wrong_a == 0) & (wrong_b == 1)).sum())
    if b + c == 0:
        return 1.0, b, c
    return float(stats.binomtest(b, b + c, 0.5).pvalue), b, c


def wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank for paired per-(model,task) metric differences. Returns (p, median diff)."""
    d = np.asarray(a) - np.asarray(b)
    d = d[~np.isnan(d)]
    if len(d) < 5 or np.all(d == 0):
        return float("nan"), float(np.median(d)) if len(d) else float("nan")
    return float(stats.wilcoxon(d).pvalue), float(np.median(d))


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (controls family-wise error rate)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(np.where(np.isnan(p), 2.0, p))
    m = int(np.sum(~np.isnan(p)))
    adj = np.full(len(p), np.nan)
    run = 0.0
    for rank, i in enumerate(order[:m]):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference between two proportions."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
