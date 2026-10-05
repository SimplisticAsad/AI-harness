"""Verification + repair experiments (Phases 10/11).

Compares routing policies at matched verification budgets: ESD risk vs token entropy vs judge vs self-consistency vs random
(vs oracle routing as an upper bound), plus verify-everything and no-harness. Verifier = independent judge + deterministic
tools. Ground truth is used only to SCORE outcomes.

python experiments/run_repair.py --config configs/main.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness.core.config import ExperimentConfig  # noqa: E402
from harness.core.io import read_jsonl, write_jsonl  # noqa: E402
from harness.core.types import Example  # noqa: E402
from harness.evaluation import analysis as AN  # noqa: E402
from harness.evaluation import stattests as T  # noqa: E402
from harness.models.flan_t5 import FlanT5Model  # noqa: E402
from harness.routing.simulate import Candidates, simulate  # noqa: E402
from harness.verification.judge import LLMJudge  # noqa: E402
from harness.verification.verifier import Verifier  # noqa: E402

K = 3  # original + 2 regenerations (retry limit = 2)
BUDGETS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]


def judge_candidates(cfg, model: str, recs: dict, ids: list[str], judge: LLMJudge) -> dict[tuple[str, int], float]:
    path = Path(cfg.results_dir) / cfg.experiment / model / "judge_candidates.jsonl"
    cache = {(r["id"], r["k"]): r["p_yes"] for r in read_jsonl(path)}
    todo = [(i, k) for i in ids for k in (1, 2) if (i, k) not in cache]
    for b in range(0, len(todo), 64):
        chunk = todo[b: b + 64]
        p, _ = judge.p_correct([recs[i]["example"]["prompt"] for i, _ in chunk],
                               [recs[i]["samples"][k - 1]["text"] for i, k in chunk])
        write_jsonl(path, [{"id": i, "k": k, "p_yes": float(v)} for (i, k), v in zip(chunk, p)], "a")
        cache.update({(i, k): float(v) for (i, k), v in zip(chunk, p)})
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = ExperimentConfig.load(a.config)
    root = Path(cfg.results_dir) / cfg.experiment
    out = root / "eval"
    df = pd.read_csv(out / "analysis_frame.csv")
    P = pd.read_csv(out / "predictions_test.csv")
    jm = next(m for m in cfg.models if m.name == "flan_t5_large")
    judge = LLMJudge(FlanT5Model(jm.name, jm.hf_dir, jm.spiece, jm.params_millions, 4, 16), jm.name)
    rows, tests, vrows = [], [], []
    for model in df.model.unique():
        d = df[df.model == model]
        recs = {r["example"]["id"]: r for r in read_jsonl(root / model / "generations.jsonl")}
        # judge threshold on VALIDATION: maximise balanced accuracy of 'P(yes) >= tau' for 'answer is correct'
        va = d[d.split == "val"].dropna(subset=["bl_judge_flan_t5_large"])
        grid = np.linspace(0.05, 0.95, 37)
        bal = [np.mean([(va.bl_judge_flan_t5_large[va.error == 0] >= t).mean(), (va.bl_judge_flan_t5_large[va.error == 1] < t).mean()]) for t in grid]
        tau = float(grid[int(np.argmax(bal))])
        verifier = Verifier(tau)
        g = P[(P.model == model) & (P.scope == "pooled") & (P.method == AN.ESD) & (P.family == "logreg")].set_index("id")
        ids = list(g.index)
        pj = judge_candidates(cfg, model, recs, ids, judge)
        dd = d.set_index("id")
        correct, vp = np.zeros((len(ids), K), bool), np.zeros((len(ids), K), bool)
        for n, i in enumerate(ids):
            r = recs[i]
            ex = Example.from_dict(r["example"])
            texts = [r["greedy"]["text"]] + [s["text"] for s in r["samples"][:K - 1]]
            grades = [r["grade"]] + r["sample_grades"][:K - 1]
            for k in range(K):
                correct[n, k] = grades[k]["correct"]
                p = dd.loc[i, "bl_judge_flan_t5_large"] if k == 0 else pj[(i, k)]
                vp[n, k] = verifier.verify(ex, texts[k], float(p)).passed
        cands = Candidates(correct, vp)
        # verifier quality when applied to everything (no routing)
        ver_pass0 = vp[:, 0]
        vrows.append({"model": model, "tau_judge": tau, "n": len(ids),
                      "verifier_false_accept(P(pass|wrong))": float(ver_pass0[~correct[:, 0]].mean()),
                      "verifier_false_reject(P(fail|correct))": float((~ver_pass0)[correct[:, 0]].mean()),
                      "initial_error_rate": float((~correct[:, 0]).mean())})
        risks = {"ESD_core": g.score.values}
        for name, m, f in (("ESD_text(A+B+C)", "ESD_text_only(A+B+C)", "logreg"), ("D_token(logreg)", "D_token", "logreg"),
                           ("B3_entropy", "B3_token_entropy", "baseline"), ("B4_self_consistency", "B4_self_consistency_disagreement", "baseline"),
                           ("B5_judge_independent", "B5b_llm_judge_independent(flan_t5_large)", "baseline")):
            x = P[(P.model == model) & (P.scope == "pooled") & (P.method == m) & (P.family == f)].set_index("id")
            if len(x):
                risks[name] = x.loc[ids, "score"].values
        rng = np.random.default_rng(cfg.stat_seed)
        risks["random"] = rng.random(len(ids))
        risks["oracle_routing(upper bound)"] = (~correct[:, 0]).astype(float) + 1e-6 * rng.random(len(ids))
        items = {}
        for pol, rk in risks.items():
            for b in BUDGETS:
                res, it = simulate(rk, b, cands, return_items=True)
                rows.append({"model": model, "policy": pol, "budget": b, **res, "verifier_tau": tau})
                items[(pol, b)] = it
        # SC majority-vote answer selection (no verifier): accuracy gain
        vote_err = d[d.id.isin(ids)].set_index("id").loc[ids, "vote_error"].values
        rows.append({"model": model, "policy": "self-consistency majority-vote answer", "budget": np.nan,
                     "initial_error_rate": float((~correct[:, 0]).mean()), "accepted_error_rate": float(vote_err.mean()),
                     "coverage": 1.0, "error_leakage": float(vote_err.mean()), "n": len(ids)})
        # paired McNemar at 40% budget on 'wrong answer reached the user'
        ref = items[("ESD_core", 0.4)]
        for pol in risks:
            if pol == "ESD_core":
                continue
            o = items[(pol, 0.4)]
            wa, wb = (ref["returned"] & ~ref["final_correct"]).astype(int), (o["returned"] & ~o["final_correct"]).astype(int)
            p, b_, c_ = T.mcnemar_exact(wa, wb)
            tests.append({"model": model, "budget": 0.4, "esd_leaks_only": b_, "other_leaks_only": c_, "comparator": pol,
                          "esd_leakage": wa.mean(), "comparator_leakage": wb.mean(), "p_mcnemar": p})
        print(model, "tau", tau, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "repair_policies.csv", index=False)
    Tt = pd.DataFrame(tests)
    if len(Tt):
        Tt["p_holm"] = T.holm(Tt.p_mcnemar.tolist())
    Tt.to_csv(out / "repair_mcnemar.csv", index=False)
    pd.DataFrame(vrows).to_csv(out / "verifier_quality.csv", index=False)


if __name__ == "__main__":
    main()
