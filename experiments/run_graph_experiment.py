"""Stage G2: self-correction against an evidence graph. Baseline vs graph-trust vs controls vs poisoned graph.

For each ARC question the model has ALREADY answered (stage 1, greedy, from results/main). Stage 2 shows the model its
previous answer plus evidence passages and asks for a final answer, scored by P(letter) over A-D (deterministic).

Conditions (k=3 passages, fixed a priori; nothing tuned on any data):
  C0 baseline            stage-1 answer, no second pass
  C1 re-ask, no evidence control for the effect of simply asking again
  C2 BM25 evidence       top-3 by retrieval score (no graph)
  C3 graph-trust         top-3 by PageRank on the clean graph
  C4 shuffled trust      top-3 by randomly permuted PageRank (control: evidence without informative trust)
  C5-C7                  same as C2-C4 on the POISONED graph (5 mutually-corroborating false passages added)
Non-LLM references: graph vote / BM25 vote / uniform vote over the same passages (no language model involved).

python experiments/run_graph_experiment.py --config configs/main.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness.core.config import ExperimentConfig  # noqa: E402
from harness.core.io import read_jsonl  # noqa: E402
from harness.evaluation import stattests as T  # noqa: E402
from harness.evidence.graph import content_terms  # noqa: E402
from harness.models.flan_t5 import FlanT5Model  # noqa: E402

K = 3
LETTERS = ["A", "B", "C", "D"]


def select(graph: dict, mode: str, rng: np.random.Generator, k: int = K) -> list[dict]:
    nodes = graph["nodes"]
    if mode == "bm25":
        key = [n["bm25"] for n in nodes]
    elif mode == "pagerank":
        key = [n["pagerank"] for n in nodes]
    elif mode == "shuffled":
        key = list(rng.permutation([n["pagerank"] for n in nodes]))
    else:
        raise KeyError(mode)
    order = np.argsort(-np.asarray(key), kind="stable")[:k]
    return [nodes[i] for i in order]


def prompt(stem: str, options: list[str], init: str | None, evidence: list[dict] | None) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    ev = ""
    if evidence:
        ev = "\n\nEvidence:\n" + "\n".join(f"[{i + 1}] {n['text'][:300]}" for i, n in enumerate(evidence))
    prev = f"\n\nA previous answer was {init}. Check it" + (" against the evidence" if evidence else "") + " and correct it if it is wrong." if init else ""
    return f"Question: {stem}\n{opts}{ev}{prev}\n\nGive the final answer as a single letter (A, B, C or D).\nAnswer:"


def vote(graph: dict, options: list[str], weight: str, rng: np.random.Generator) -> str:
    """Non-LLM: option score = sum_over_passages weight * (fraction of option's content terms found in the passage)."""
    sc = np.zeros(len(options))
    for n in graph["nodes"]:
        t = content_terms(n["text"])
        w = {"pagerank": n["pagerank"], "bm25": n["bm25"], "uniform": 1.0}[weight]
        for i, o in enumerate(options):
            ot = content_terms(o)
            if ot:
                sc[i] += w * len(ot & t) / len(ot)
    best = np.flatnonzero(sc == sc.max())
    return LETTERS[int(rng.choice(best))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--graphs", default="results/graph/graphs.jsonl")
    ap.add_argument("--out", default="results/graph")
    a = ap.parse_args()
    cfg = ExperimentConfig.load(a.config)
    graphs = {g["id"]: g for g in read_jsonl(a.graphs)}
    out = Path(a.out)
    rows, diag = [], []
    for mc in cfg.models:
        recs = [r for r in read_jsonl(Path(cfg.results_dir) / cfg.experiment / mc.name / "generations.jsonl")
                if r["example"]["task"] == "factual_qa" and r["example"]["id"] in graphs]
        if not recs:
            continue
        model = FlanT5Model(mc.name, mc.hf_dir, mc.spiece, mc.params_millions, 4, 16)
        conds: dict[str, list[str]] = {c: [] for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]}
        votes: dict[str, list[str]] = {f"{g}_{w}": [] for g in ("clean", "poisoned") for w in ("pagerank", "bm25", "uniform")}
        meta = []
        for r in recs:
            g = graphs[r["example"]["id"]]
            init = r["grade"]["extracted"]
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(r["example"]["id"].encode()).digest()[:4], "little"))
            for c, (gk, mode) in {"C2": ("clean", "bm25"), "C3": ("clean", "pagerank"), "C4": ("clean", "shuffled"),
                                  "C5": ("poisoned", "bm25"), "C6": ("poisoned", "pagerank"), "C7": ("poisoned", "shuffled")}.items():
                ev = select(g[gk], mode, np.random.default_rng(rng.integers(1 << 31)))
                conds[c].append(prompt(g["stem"], g["options"], init, ev))
                if c in ("C5", "C6", "C7"):
                    meta.append({"id": g["id"], "cond": c, "poison_share": float(np.mean([n.get("poison", False) for n in ev])),
                                 "gold_term_support": 0.0})
                else:
                    gold_terms = content_terms(g["options"]["ABCD".index(g["gold"])])
                    sup = float(np.mean([len(gold_terms & content_terms(n["text"])) / max(len(gold_terms), 1) >= 0.5 for n in ev]))
                    meta.append({"id": g["id"], "cond": c, "poison_share": 0.0, "gold_term_support": sup})
            conds["C1"].append(prompt(g["stem"], g["options"], init, None))
            for gk in ("clean", "poisoned"):
                for w in ("pagerank", "bm25", "uniform"):
                    votes[f"{gk}_{w}"].append(vote(g[gk], g["options"], w, np.random.default_rng(1)))
        res = {"C0": [r["grade"]["extracted"] or "?" for r in recs]}
        for c, ps in conds.items():
            p = model.first_token_probs(ps, LETTERS)
            res[c] = [LETTERS[i] for i in p.argmax(1)]
            print(mc.name, c, "done", flush=True)
        for k, v in votes.items():
            res["vote_" + k] = v
        gold = [r["example"]["reference"] for r in recs]
        for i, r in enumerate(recs):
            for c, ans in res.items():
                rows.append({"model": mc.name, "id": r["example"]["id"], "condition": c, "answer": ans[i], "gold": gold[i],
                             "correct": int(ans[i] == gold[i]), "initial_correct": int(res["C0"][i] == gold[i]),
                             "split": r["split"], "dataset": r["example"]["meta"]["dataset"]})
        diag += [{"model": mc.name, **m} for m in meta]
    df = pd.DataFrame(rows)
    df.to_csv(out / "condition_results.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "evidence_diagnostics.csv", index=False)
    # ---- summary with paired stats
    rng = np.random.default_rng(0)
    summ = []
    for m, g in df.groupby("model"):
        piv = g.pivot(index="id", columns="condition", values="correct")
        init = g.drop_duplicates("id").set_index("id").initial_correct.reindex(piv.index)
        for c in piv.columns:
            x = piv[c].values.astype(int)
            bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(1000)]
            fixed = int(((init.values == 0) & (x == 1)).sum()); broken = int(((init.values == 1) & (x == 0)).sum())
            p_vs0 = T.mcnemar_exact(1 - x, 1 - init.values)[0] if c != "C0" else np.nan
            ref = piv["C2"].values.astype(int) if "C2" in piv else None
            p_vs2 = T.mcnemar_exact(1 - x, 1 - ref)[0] if (ref is not None and c not in ("C0", "C2")) else np.nan
            summ.append({"model": m, "condition": c, "n": len(x), "accuracy": x.mean(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5),
                         "fixed": fixed, "broken": broken, "net": fixed - broken, "p_mcnemar_vs_C0": p_vs0, "p_mcnemar_vs_C2": p_vs2})
    S = pd.DataFrame(summ)
    S["p_holm_vs_C0"] = np.nan
    for m in S.model.unique():
        idx = S[(S.model == m) & S.p_mcnemar_vs_C0.notna()].index
        S.loc[idx, "p_holm_vs_C0"] = T.holm(S.loc[idx, "p_mcnemar_vs_C0"].tolist())
    S.to_csv(out / "condition_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))
    D = pd.DataFrame(diag).groupby(["model", "cond"])[["poison_share", "gold_term_support"]].mean().round(3)
    print(D)


if __name__ == "__main__":
    main()
