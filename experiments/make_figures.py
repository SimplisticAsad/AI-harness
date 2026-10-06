"""Generate all research figures into reports/figures/ from saved result CSVs (no recomputation of experiments)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from harness.evaluation import metrics as M  # noqa: E402

OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]
ESD = "ESD_core(A+B+C+D+E)"
KEY = [(ESD, "logreg", "ESD core (A+B+C+D+E)"), ("B3_token_entropy", "baseline", "B3 token entropy"),
       ("B2_token_logprob_confidence", "baseline", "B2 token logprob"),
       ("B4_self_consistency_disagreement", "baseline", "B4 self-consistency"),
       ("B5_llm_judge_self", "baseline", "B5 LLM judge (self)"),
       ("B5b_llm_judge_independent(flan_t5_large)", "baseline", "B5 LLM judge (large)"),
       ("A_affective", "logreg", "B7 affect only"), ("B_epistemic", "logreg", "B8 epistemic only"),
       ("C_semantic", "logreg", "B9 semantic only")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.grid": True, "grid.alpha": 0.25})
skipped: list[str] = []


def save(fig, name: str, out: Path) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=150)
    plt.close(fig)


def get(P, model, scope, m, f):
    return P[(P.model == model) & (P.scope == scope) & (P.method == m) & (P.family == f)]


def fig_distribution(P, out):
    models = P.model.unique()
    fig, ax = plt.subplots(1, len(models), figsize=(4 * len(models), 3), squeeze=False)
    for a, model in zip(ax[0], models):
        g = get(P, model, "pooled", ESD, "logreg")
        for y, c, lab in ((0, OI[0], "correct"), (1, OI[1], "incorrect")):
            a.hist(g[g.y == y].p_platt, bins=np.linspace(0, 1, 21), alpha=0.6, color=c, label=f"{lab} (n={int((g.y == y).sum())})", density=True)
        a.set(title=model, xlabel="ESD distress (calibrated P(error))", ylabel="density")
        a.legend(frameon=False)
    save(fig, "01_distress_distribution.png", out)


def fig_dist_vs_error(P, out):
    models = P.model.unique()
    fig, ax = plt.subplots(1, len(models), figsize=(4 * len(models), 3.2), squeeze=False)
    for a, model in zip(ax[0], models):
        g = get(P, model, "pooled", ESD, "logreg").copy()
        g["bin"] = pd.qcut(g.score.rank(method="first"), 8, labels=False)
        b = g.groupby("bin").agg(x=("p_platt", "mean"), y=("y", "mean"), n=("y", "size"))
        lo_hi = [M_wilson(r.y * r.n, r.n) for r in b.itertuples()]
        a.errorbar(b.x, b.y, yerr=[b.y - [l for l, _ in lo_hi], [h for _, h in lo_hi] - b.y], fmt="o-", color=OI[0], capsize=2)
        a.axhline(g.y.mean(), color=OI[7], ls=":", label="base error rate")
        a.set(title=f"{model}  (Spearman {M.correlations(g.y.values, g.score.values)['spearman']:.2f})",
              xlabel="mean distress (octile bins)", ylabel="observed error rate", ylim=(0, 1.02))
        a.legend(frameon=False)
    save(fig, "02_distress_vs_error_probability.png", out)


def M_wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_curves(P, out, kind):
    models = P.model.unique()
    fig, ax = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.8), squeeze=False)
    for a, model in zip(ax[0], models):
        for (m, f, lab), c in zip(KEY, OI * 2):
            g = get(P, model, "pooled", m, f)
            if g.empty or g.score.nunique() < 2 or g.y.nunique() < 2:
                continue
            if kind == "roc":
                x, y, _ = roc_curve(g.y, g.score); a.plot(x, y, color=c, lw=1.3 if m == ESD else 0.9, label=f"{lab} ({M.auroc(g.y.values, g.score.values):.2f})")
            else:
                p, r, _ = precision_recall_curve(g.y, g.score); a.plot(r, p, color=c, lw=1.3 if m == ESD else 0.9, label=f"{lab} ({M.auprc(g.y.values, g.score.values):.2f})")
        if kind == "roc":
            a.plot([0, 1], [0, 1], color=OI[7], ls=":"); a.set(xlabel="false positive rate", ylabel="true positive rate")
        else:
            a.set(xlabel="recall (errors)", ylabel="precision (errors)")
        a.set_title(model); a.legend(frameon=False, fontsize=6)
    save(fig, "03_roc_curves.png" if kind == "roc" else "04_precision_recall_curves.png", out)


def fig_calibration(P, out):
    models = P.model.unique()
    fig, ax = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), squeeze=False)
    for a, model in zip(ax[0], models):
        g = get(P, model, "pooled", ESD, "logreg")
        for col, c, lab in (("p_platt", OI[0], "Platt (val-fit)"), ("p_iso", OI[2], "isotonic (val-fit)")):
            rel = M.reliability(g.y.values, g[col].values, 8)
            a.plot([r["mean_pred"] for r in rel], [r["frac_pos"] for r in rel], "o-", color=c, label=f"{lab} ECE={M.ece(g.y.values, g[col].values, 8):.3f}")
        a.plot([0, 1], [0, 1], color=OI[7], ls=":"); a.set(title=model, xlabel="predicted P(error)", ylabel="observed frequency")
        a.legend(frameon=False, fontsize=7)
    save(fig, "05_calibration_curves.png", out)


def fig_risk_coverage(P, out):
    models = P.model.unique()
    fig, ax = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False)
    for a, model in zip(ax[0], models):
        for (m, f, lab), c in zip(KEY[:6], OI):
            g = get(P, model, "pooled", m, f)
            if g.empty or g.score.nunique() < 2:
                continue
            rc = M.risk_coverage(g.y.values, g.score.values)
            a.plot(rc["coverage"], rc["risk"], color=c, lw=1.4 if m == ESD else 0.9, label=f"{lab} (AURC {rc['aurc']:.3f})")
        g = get(P, model, "pooled", ESD, "logreg")
        a.axhline(g.y.mean(), color=OI[7], ls=":", label="accept all")
        a.set(title=model, xlabel="coverage (fraction answered)", ylabel="selective risk (error among answered)")
        a.legend(frameon=False, fontsize=6)
    save(fig, "06_risk_coverage.png", out)


def fig_error_types(frame, out):
    t = frame[frame.error == 1].groupby(["model", "error_type"]).size().unstack(fill_value=0)
    t = t.div(t.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    t.plot.bar(stacked=True, ax=ax, color=OI[: len(t.columns)], width=0.7)
    ax.set(ylabel="share of errors", xlabel="", title="Error taxonomy (grader-assigned)"); ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1, 1))
    plt.xticks(rotation=0)
    save(fig, "07_error_type_distribution.png", out)


def fig_model_comparison(Mx, out):
    d = Mx[(Mx.scope == "pooled")]
    sel = [(ESD, "logreg", "ESD core"), ("D_token", "logreg", "token only"), ("E_candidate", "logreg", "candidate only"),
           ("B5_llm_judge_self", "baseline", "judge (self)"), ("B3_token_entropy", "baseline", "entropy"),
           ("CTRL_length+difficulty+task", "logreg", "length+diff+task")]
    models = list(d.model.unique())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / len(sel)
    for j, (m, f, lab) in enumerate(sel):
        vals, lo, hi = [], [], []
        for model in models:
            r = d[(d.model == model) & (d.method == m) & (d.family == f)]
            vals.append(r.auroc.iloc[0] if len(r) else np.nan); lo.append(r.auroc_lo.iloc[0] if len(r) else np.nan); hi.append(r.auroc_hi.iloc[0] if len(r) else np.nan)
        x = np.arange(len(models)) + j * w
        ax.bar(x, vals, w, color=OI[j], label=lab, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], capsize=1.5)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xticks(np.arange(len(models)) + 0.4 - w / 2); ax.set_xticklabels(models)
    ax.set(ylabel="test AUROC (95% CI)", ylim=(0.3, 1.0), title="Error detection by model (pooled tasks)"); ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "08_model_comparison.png", out)


def fig_within_task(ev, out):
    f = ev / "within_task_auroc.csv"
    if not f.exists():
        skipped.append("08c_within_task"); return
    W = pd.read_csv(f)
    sel = [(ESD, "logreg", "ESD core"), ("A_affective", "logreg", "affect"), ("B_epistemic", "logreg", "epistemic"), ("C_semantic", "logreg", "semantic"),
           ("D_token", "logreg", "token"), ("E_candidate", "logreg", "candidates"), ("B5_llm_judge_self", "baseline", "judge (self)"),
           ("B4_self_consistency_disagreement", "baseline", "self-consistency"), ("CTRL_length+difficulty+task", "logreg", "length+diff+task ctrl")]
    models = list(W.model.unique()); fig, ax = plt.subplots(figsize=(8, 3.6)); w = 0.8 / len(sel)
    for j, (m, fam, lab) in enumerate(sel):
        r = [W[(W.model == mo) & (W.method == m) & (W.family == fam)] for mo in models]
        v = np.array([x.within_task_auroc.iloc[0] if len(x) else np.nan for x in r]); lo = np.array([x.lo.iloc[0] if len(x) else np.nan for x in r]); hi = np.array([x.hi.iloc[0] if len(x) else np.nan for x in r])
        ax.bar(np.arange(len(models)) + j * w, v, w, color=OI[j % len(OI)], label=lab, yerr=[v - lo, hi - v], capsize=1.2)
    ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_xticks(np.arange(len(models)) + 0.4 - w / 2); ax.set_xticklabels(models)
    ax.set(ylabel="within-task macro AUROC (95% CI)", ylim=(0.3, 1.0), title="Task-confound-free comparison (post-hoc headline metric)"); ax.legend(frameon=False, fontsize=6, ncol=3)
    save(fig, "08c_within_task_auroc.png", out)


def fig_task_heatmap(Mx, out):
    d = Mx[(Mx.method == ESD) & (Mx.family == "logreg")].pivot(index="model", columns="scope", values="auroc")
    fig, ax = plt.subplots(figsize=(7, 2.4))
    im = ax.imshow(d.values, cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
    ax.set_xticks(range(d.shape[1])); ax.set_xticklabels(d.columns, rotation=30); ax.set_yticks(range(d.shape[0])); ax.set_yticklabels(d.index)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            ax.text(j, i, f"{d.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.grid(False); fig.colorbar(im, label="ESD core test AUROC"); ax.set_title("ESD error detection by task (0.5 = chance)")
    save(fig, "08b_task_heatmap.png", out)


def fig_ablation(Mx, out):
    order = ["A_affective", "B_epistemic", "C_semantic", "D_token", "E_candidate", "L_linguistic", "F_affect+epistemic",
             "G_affect+semantic", "H_epistemic+semantic", "ESD_text_only(A+B+C)", ESD, "I_all(+linguistic)"]
    d = Mx[(Mx.scope == "pooled") & (Mx.family == "logreg")]
    models = list(d.model.unique())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / len(models)
    for j, model in enumerate(models):
        r = d[d.model == model].set_index("method").reindex(order)
        ax.bar(np.arange(len(order)) + j * w, r.auroc, w, color=OI[j], label=model,
               yerr=[(r.auroc - r.auroc_lo).values, (r.auroc_hi - r.auroc).values], capsize=1.2)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xticks(np.arange(len(order)) + 0.4); ax.set_xticklabels([o.replace("_", "\n", 1) for o in order], rotation=45, ha="right", fontsize=7)
    ax.set(ylabel="test AUROC (logreg, pooled)", ylim=(0.3, 1.0), title="Ablation: which signal families carry information?"); ax.legend(frameon=False, fontsize=7)
    save(fig, "09_ablation.png", out)


def fig_repair(R, out):
    if R is None:
        skipped.append("10_repair_success (no repair results)"); return
    models = list(R.model.unique())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for j, model in enumerate(models):
        for pol, c in (("ESD_core", OI[0]), ("B3_entropy", OI[1]), ("B5_judge_independent", OI[2]), ("random", OI[7]), ("oracle_routing(upper bound)", OI[3])):
            r = R[(R.model == model) & (R.policy == pol)].sort_values("budget")
            if r.empty:
                continue
            ls = ["-", "--", ":"][j % 3]
            ax[0].plot(r.budget, r.error_leakage, ls, color=c, label=f"{pol}" if j == 0 else None, lw=1.2)
            ax[1].plot(r.budget, r.repair_success_rate, ls, color=c, lw=1.2)
    ax[0].set(xlabel="verification budget (fraction routed to verification)", ylabel="wrong answers reaching user (per request)", title="Accepted-error leakage (line style = model)")
    ax[1].set(xlabel="verification budget", ylabel="repair success among initially wrong & flagged", title="Repair success")
    ax[0].legend(frameon=False, fontsize=6)
    save(fig, "10_repair_success.png", out)


def fig_latency(Mx, S, out):
    d = S[(S.task == "pooled") & (S.selected_on_val.fillna(True).astype(bool) | S.method.str.contains("baseline|control|CTRL"))]
    sel = ["B2_token_logprob_confidence", "B3_token_entropy", "B4_self_consistency_disagreement", "B5_llm_judge_self", "B6_semantic_disagreement",
           ESD, "D_token", "A_affective", "B_epistemic", "C_semantic"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for model, mk in zip(d.model.unique(), ["o", "s", "^"]):
        for m, c in zip(sel, OI * 2):
            r = d[(d.model == model) & (d.method.str.startswith(m + " "))]
            r = r[r.method.str.contains("logreg|baseline")] if len(r) > 1 else r
            if r.empty:
                continue
            r = r.iloc[0]
            ax.scatter(r.latency, r.auroc, color=c, marker=mk, s=28, label=m if model == d.model.unique()[0] else None)
    ax.set(xscale="log", xlabel="inference latency per item (s, log scale; marker = model)", ylabel="pooled test AUROC", title="Detection quality vs cost")
    ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.legend(frameon=False, fontsize=6, ncol=2)
    save(fig, "11_latency_cost.png", out)


def fig_confusion(P, out):
    models = P.model.unique()
    fig, ax = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.2), squeeze=False)
    for a, model in zip(ax[0], models):
        g = get(P, model, "pooled", ESD, "logreg")
        flag = (g.score >= g.threshold).astype(int).values
        cm = np.array([[((g.y == 0) & (flag == 0)).sum(), ((g.y == 0) & (flag == 1)).sum()],
                       [((g.y == 1) & (flag == 0)).sum(), ((g.y == 1) & (flag == 1)).sum()]])
        a.imshow(cm, cmap="Blues"); a.grid(False)
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "k")
        a.set_xticks([0, 1]); a.set_xticklabels(["accept", "flag"]); a.set_yticks([0, 1]); a.set_yticklabels(["correct", "incorrect"])
        a.set_title(f"{model}\nthreshold chosen on validation")
    save(fig, "12_confusion_matrices.png", out)


def fig_generalization(out_dir, out):
    f1, f2 = out_dir / "gen_leave_one_task_out.csv", out_dir / "gen_cross_model.csv"
    if not (f1.exists() and f2.exists()):
        skipped.append("13_generalization"); return
    L, C = pd.read_csv(f1), pd.read_csv(f2)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    pv = L.pivot(index="model", columns="held_out_task", values="auroc_transfer")
    im = ax[0].imshow(pv.values, cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto"); ax[0].grid(False)
    ax[0].set_xticks(range(pv.shape[1])); ax[0].set_xticklabels(pv.columns, rotation=30, fontsize=7); ax[0].set_yticks(range(pv.shape[0])); ax[0].set_yticklabels(pv.index, fontsize=7)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax[0].text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax[0].set_title("H7: leave-one-task-out AUROC")
    cv = C.pivot(index="train_model", columns="test_model", values="auroc")
    ax[1].imshow(cv.values, cmap="RdBu_r", vmin=0.2, vmax=0.8); ax[1].grid(False)
    ax[1].set_xticks(range(cv.shape[1])); ax[1].set_xticklabels(cv.columns, rotation=20, fontsize=7); ax[1].set_yticks(range(cv.shape[0])); ax[1].set_yticklabels(cv.index, fontsize=7)
    for i in range(cv.shape[0]):
        for j in range(cv.shape[1]):
            ax[1].text(j, i, f"{cv.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax[1].set(xlabel="test model", ylabel="train model", title="H6: cross-model transfer AUROC")
    save(fig, "13_generalization.png", out)


def fig_adversarial(out_dir, out):
    f = out_dir / "adversarial_natural_cases.csv"
    if not f.exists():
        skipped.append("14_adversarial"); return
    A = pd.read_csv(f)
    A = A[A.scorer.isin(["ESD_core (logreg)", "B3 token entropy", "B5 LLM judge (self)"])]
    cases = list(dict.fromkeys(A.case))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    scorers = list(dict.fromkeys(A.scorer)); w = 0.8 / len(scorers)
    for j, s in enumerate(scorers):
        r = A[A.scorer == s].groupby("case").apply(lambda g: np.average(g.flag_rate.fillna(0), weights=g.n.clip(lower=1)), include_groups=False).reindex(cases)
        ax.bar(np.arange(len(cases)) + j * w, r.values, w, color=OI[j], label=s)
    ax.axhline(0.4, color="k", ls=":", lw=0.8, label="40% budget (chance flag rate)")
    ax.set_xticks(np.arange(len(cases)) + 0.4); ax.set_xticklabels(cases, rotation=30, ha="right", fontsize=7)
    ax.set(ylabel="flag rate (pooled over models)", title="Adversarial subsets: wrong-answer cases want HIGH flag rate, correct-answer cases want LOW"); ax.legend(frameon=False, fontsize=7)
    save(fig, "14_adversarial_cases.png", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="results/main")
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    ev, out = Path(a.exp) / "eval", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P, Mx = pd.read_csv(ev / "predictions_test.csv"), pd.read_csv(ev / "metrics_all.csv")
    S = pd.read_csv(Path(a.exp).parent / "summary.csv")
    frame = pd.read_csv(ev / "analysis_frame.csv")
    R = pd.read_csv(ev / "repair_policies.csv") if (ev / "repair_policies.csv").exists() else None
    fig_distribution(P, out); fig_dist_vs_error(P, out); fig_curves(P, out, "roc"); fig_curves(P, out, "pr")
    fig_calibration(P, out); fig_risk_coverage(P, out); fig_error_types(frame, out); fig_model_comparison(Mx, out)
    fig_within_task(ev, out); fig_task_heatmap(Mx, out); fig_ablation(Mx, out); fig_repair(R, out); fig_latency(Mx, S, out)
    fig_confusion(P, out); fig_generalization(ev, out); fig_adversarial(ev, out)
    json.dump({"skipped": skipped}, open(out / "_skipped.json", "w"))
    print("figures:", sorted(p.name for p in out.glob("*.png")), "skipped:", skipped)


if __name__ == "__main__":
    main()
