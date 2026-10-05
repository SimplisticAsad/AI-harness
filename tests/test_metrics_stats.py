import numpy as np

from harness.evaluation import metrics as M
from harness.evaluation import stattests as T


def test_auroc_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    assert M.auroc(y, np.array([.1, .2, .8, .9])) == 1.0
    assert M.auroc(y, np.zeros(4)) == 0.5
    assert np.isnan(M.auroc(np.zeros(4), np.arange(4)))


def test_ece_zero_for_perfect_calibration():
    p = np.r_[np.full(100, 0.2), np.full(100, 0.8)]
    rng = np.random.default_rng(0)
    y = np.r_[rng.random(100) < 0.2, rng.random(100) < 0.8].astype(int)
    assert M.ece(y, p) < 0.1
    assert M.ece(np.ones(10), np.zeros(10)) == 1.0


def test_risk_coverage_monotone_with_good_scorer():
    y = np.array([0, 0, 0, 1, 1])
    rc = M.risk_coverage(y, np.array([.1, .2, .3, .8, .9]))
    assert rc["risk"][2] == 0.0 and abs(rc["aurc"] - np.mean([0, 0, 0, .25, .4])) < 1e-9


def test_decision_rates_definitions():
    y = np.array([1, 1, 0, 0])
    r = M.decision_rates(y, np.array([1, 0, 0, 1]))
    assert r["false_acceptance_rate"] == 0.5 and r["false_rejection_rate"] == 0.5 and r["coverage"] == 0.5


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200); s = y + rng.normal(size=200)
    p, lo, hi = M.bootstrap(M.auroc, [y, s], 200, 0)
    assert lo <= p <= hi


def test_permutation_tests_and_holm():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300); good = y + rng.normal(size=300); noise = rng.normal(size=300)
    assert T.permutation_auc_vs_chance(y, good, 200) < 0.01
    assert T.permutation_auc_vs_chance(y, noise, 200) > 0.05
    assert T.permutation_auc_diff(y, good, noise, 200) < 0.05
    adj = T.holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and max(adj) <= 1.0 and all(a >= p for a, p in zip(adj, [0.01, 0.04, 0.03]))


def test_mcnemar():
    p, b, c = T.mcnemar_exact(np.array([1] * 10 + [0] * 2), np.array([0] * 10 + [1] * 2))
    assert b == 10 and c == 2 and p < 0.05
