"""Stage 3 CLI: full evaluation grid + analyses -> results/<exp>/eval/*.csv and results/summary.csv.

python experiments/run_eval.py --config configs/main.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness.core.config import ExperimentConfig  # noqa: E402
from harness.core.hardware import detect_hardware  # noqa: E402
from harness.core.io import write_manifest  # noqa: E402
from harness.evaluation import analysis as AN  # noqa: E402
from harness.evaluation.runner import StatCfg, run_grid  # noqa: E402
from harness.scoring.dataset import add_anomaly, build_frame  # noqa: E402

BASELINE_LABEL = {
    "B1_base_llm_accept_all": "B1", "B2_token_logprob_confidence": "B2", "B2b_answer_token_logprob": "B2",
    "B3_token_entropy": "B3", "B4_self_consistency_disagreement": "B4", "B5_llm_judge_self": "B5",
    "B5b_llm_judge_independent(flan_t5_large)": "B5", "B6_semantic_disagreement": "B6",
    "A_affective": "B7_affect_only", "B_epistemic": "B8_epistemic_only", "C_semantic": "B9_semantic_only",
    "ESD_core(A+B+C+D+E)": "B10_combined_ESD",
}
USES_SAMPLES = {"B4_self_consistency_disagreement", "B6_semantic_disagreement"}


def method_cost(method: str, fam: str, d: pd.DataFrame, judge_col: str | None) -> tuple[float, float, float]:
    """(latency_s, tokens, model_calls) per item for the resources a method needs at inference time."""
    base_l, base_t = d.lat_greedy_s.mean(), d.tok_greedy.mean()
    ns = d.n_samples.mean()
    uses_s = method in USES_SAMPLES or "E_candidate" in method or method.startswith(("ESD_core", "I_all")) or "ESD_core+CTRL" in method
    judge = method.startswith("B5")
    lat, tok, calls = base_l, base_t, 1.0
    if uses_s:
        lat += d.lat_samples_s.mean(); tok += d.tok_samples.mean(); calls += ns
    if judge:
        jc = "lat_judge_flan_t5_large" if "independent" in method else judge_col
        if jc in d.columns:
            lat += d[jc].mean()
        calls += 1
    return float(lat), float(tok), float(calls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quick", action="store_true", help="tiny bootstrap for debugging")
    a = ap.parse_args()
    cfg = ExperimentConfig.load(a.config)
    root = Path(cfg.results_dir) / cfg.experiment
    out = root / "eval"
    out.mkdir(parents=True, exist_ok=True)
    sc = StatCfg(B=50 if a.quick else cfg.bootstrap_resamples, seed=cfg.stat_seed)

    frames = []
    for m in cfg.models:
        if not (root / m.name / "features.jsonl").exists():
            continue
        f = build_frame(root / m.name, judges=[m.name, "flan_t5_large"])
        f["bl_judge_self"] = f.get(f"bl_judge_{m.name}")
        f["lat_judge_self"] = f.get(f"lat_judge_{m.name}")
        frames.append(f)
    df = add_anomaly(pd.concat(frames, ignore_index=True))
    df.drop(columns=["answer_text"]).to_csv(out / "analysis_frame.csv", index=False)
    print("frame", df.shape, df.groupby(["model", "task"]).size().to_dict())

    Mx, P = run_grid(df, sc)
    Mx.to_csv(out / "metrics_all.csv", index=False)
    P.to_csv(out / "predictions_test.csv", index=False)

    # ---- summary.csv (required columns first)
    rows = []
    for _, r in Mx.iterrows():
        d = df[(df.model == r.model) & (df.split == "test") & ((df.task == r.scope) | (r.scope == "pooled"))]
        lat, tok, calls = method_cost(r.method, r.family, d, "lat_judge_self")
        rows.append({
            "model": r.model, "task": r.scope, "baseline": BASELINE_LABEL.get(r.method, "ablation/control"),
            "method": f"{r.method} [{r.family}{'' if pd.isna(r.get('chosen')) else ':' + str(r.chosen)}]",
            "accuracy": 1 - r.base_error_rate, "error_rate": r.base_error_rate, "precision": r.precision,
            "recall": r.recall, "f1": r.f1, "auroc": r.auroc, "auprc": r.auprc, "ece": r.ece,
            "latency": lat, "cost": tok, "model_calls": calls,
            "auroc_lo": r.auroc_lo, "auroc_hi": r.auroc_hi, "brier": r.brier, "ece_isotonic": r.ece_isotonic,
            "false_acceptance_rate": r.false_acceptance_rate, "false_rejection_rate": r.false_rejection_rate,
            "accepted_error_rate": r.accepted_error_rate, "coverage": r.coverage, "aurc": r.aurc,
            "pearson": r.pearson, "spearman": r.spearman, "val_auroc": r.val_auroc,
            "selected_on_val": r.get("selected_on_val"), "n_test": r.n_test})
    pd.DataFrame(rows).to_csv(Path(cfg.results_dir) / "summary.csv", index=False)

    # ---- analyses
    AN.paired_comparisons(P, sc).to_csv(out / "h4_paired_comparisons.csv", index=False)
    AN.vs_chance(P, sc).to_csv(out / "vs_chance.csv", index=False)
    w = [AN.wilcoxon_across_cells(Mx, AN.ESD, o) for o in
         ["A_affective", "B_epistemic", "C_semantic", "D_token", "E_candidate", "L_linguistic", "CTRL_length+difficulty+task"]]
    w.append(AN.wilcoxon_across_cells(Mx, "ESD_core+CTRL", "CTRL_length+difficulty+task"))
    pd.DataFrame(w).to_csv(out / "wilcoxon_cells.csv", index=False)

    ctl_rows, part_rows, strata, plc = [], [], [], []
    for model in df.model.unique():
        for scope in ["pooled"] + sorted(df.task.unique()):
            al = AN.aligned(P, model, scope, "ESD_core+CTRL", "logreg", "CTRL_length+difficulty+task", "logreg")
            if al is None:
                continue
            y, s1, s2 = al
            from harness.evaluation import metrics as M, stattests as T
            d, lo, hi = M.paired_bootstrap_diff(M.auroc, y, s1, s2, sc.B, sc.seed)
            ctl_rows.append({"model": model, "scope": scope, "auroc_esd+ctrl": M.auroc(y, s1), "auroc_ctrl_only": M.auroc(y, s2),
                             "delta": d, "lo": lo, "hi": hi, "p_perm": T.permutation_auc_diff(y, s1, s2, 1000, sc.seed)})
        g = P[(P.model == model) & (P.scope == "pooled") & (P.method == AN.ESD) & (P.family == "logreg")]
        te = df[(df.model == model) & (df.split == "test")].set_index("id").loc[g.id].reset_index()
        part_rows.append({"model": model, **AN.partial_spearman(te, g.score.values, sc)})
        s = AN.length_strata(te, g.score.values); s.insert(0, "model", model); strata.append(s)
        plc.append(AN.placebo(df, model, 10 if a.quick else 30))
    pd.DataFrame(ctl_rows).to_csv(out / "controls_incremental.csv", index=False)
    pd.DataFrame(part_rows).to_csv(out / "controls_partial_spearman.csv", index=False)
    pd.concat(strata).to_csv(out / "controls_length_strata.csv", index=False)
    pd.DataFrame(plc).to_csv(out / "controls_placebo.csv", index=False)
    AN.leave_one_task_out(df, sc).to_csv(out / "gen_leave_one_task_out.csv", index=False)
    AN.cross_model(df, sc).to_csv(out / "gen_cross_model.csv", index=False)
    AN.error_type_detectability(df, P, sc).to_csv(out / "error_type_detectability.csv", index=False)
    pd.DataFrame([AN.calibration_mismatch(df, P, m) for m in df.model.unique()]).to_csv(out / "overconfidence.csv", index=False)

    write_manifest(out / "manifest_eval.json", config=cfg.model_dump(), hardware=detect_hardware().to_dict(),
                   n_rows=len(df), bootstrap=sc.B, stat_seed=sc.seed)
    print("done ->", out)


if __name__ == "__main__":
    main()
