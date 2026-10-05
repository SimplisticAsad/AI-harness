"""Within-task (macro) AUROC: removes the between-task base-rate confound that inflates/deflates pooled AUROC.

Post-hoc analysis decision (made after seeing that a task-prior-only control matched pooled ESD AUROC); pooled results
are still reported. Uses the saved test-split predictions of the POOLED scorers, evaluated inside each task, then
averaged with weights = task test size (tasks lacking both classes are skipped). CI: stratified bootstrap (resample items
within each task).
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")  # noqa: E402

import numpy as np
import pandas as pd

from harness.evaluation import metrics as M

P = pd.read_csv("results/main/eval/predictions_test.csv")
pooled = P[P.scope == "pooled"]
rng = np.random.default_rng(0)


def macro(g: pd.DataFrame) -> float:
    vals, w = [], []
    for _, h in g.groupby("task"):
        a = M.auroc(h.y.values, h.score.values) if h.score.nunique() > 1 else 0.5
        if not np.isnan(a):
            vals.append(a); w.append(len(h))
    return float(np.average(vals, weights=w)) if vals else float("nan")


rows = []
for (model, method, fam), g in pooled.groupby(["model", "method", "family"]):
    pt = macro(g)
    groups = [h for _, h in g.groupby("task")]
    bs = []
    for _ in range(500):
        res = pd.concat([h.iloc[rng.integers(0, len(h), len(h))] for h in groups])
        bs.append(macro(res))
    rows.append({"model": model, "method": method, "family": fam, "within_task_auroc": pt,
                 "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5)})
R = pd.DataFrame(rows)
R.to_csv("results/main/eval/within_task_auroc.csv", index=False)
pd.set_option("display.width", 220)
sel = [("ESD_core(A+B+C+D+E)", "logreg"), ("ESD_core(A+B+C+D+E)", "fixed_equal_weights"), ("A_affective", "logreg"),
       ("B_epistemic", "logreg"), ("C_semantic", "logreg"), ("D_token", "logreg"), ("E_candidate", "logreg"),
       ("L_linguistic", "logreg"), ("ESD_text_only(A+B+C)", "logreg"), ("CTRL_length+difficulty+task", "logreg"),
       ("ESD_core+CTRL", "logreg"), ("CTRL_task_prior_only", "control"), ("B2_token_logprob_confidence", "baseline"),
       ("B3_token_entropy", "baseline"), ("B4_self_consistency_disagreement", "baseline"), ("B5_llm_judge_self", "baseline"),
       ("B5b_llm_judge_independent(flan_t5_large)", "baseline"), ("B6_semantic_disagreement", "baseline"), ("random_scores", "control")]
out = []
for m, f in sel:
    r = {"method": f"{m[:36]}|{f[:5]}"}
    for mod in ("flan_t5_small", "flan_t5_base", "flan_t5_large"):
        x = R[(R.model == mod) & (R.method == m) & (R.family == f)]
        r[mod[-5:]] = f"{x.within_task_auroc.iloc[0]:.2f} [{x.lo.iloc[0]:.2f},{x.hi.iloc[0]:.2f}]" if len(x) else "-"
    out.append(r)
print(pd.DataFrame(out).to_string(index=False))
