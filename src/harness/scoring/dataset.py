"""Assemble the analysis frame: one row per (model, example) with labels, features, controls and baselines."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from harness.core.io import read_jsonl
from harness.signals.text import words


def _vote_correct(rec: dict) -> tuple[float, float]:
    """Majority-vote answer correctness (self-consistency baseline) and mean sample accuracy."""
    from harness.signals.answers import extract_answer
    task = rec["example"]["task"]
    members = [(rec["greedy"]["text"], rec["grade"])] + [(s["text"], g) for s, g in zip(rec["samples"], rec["sample_grades"])]
    keys = [str(extract_answer(task, t)) if extract_answer(task, t) is not None else "<none>" for t, _ in members]
    cnt = Counter(keys)
    top = max(cnt.values())
    # ties broken in favour of the greedy answer's cluster, then first occurrence (deterministic)
    top_keys = [k for k in dict.fromkeys(keys) if cnt[k] == top]
    pick = keys[0] if keys[0] in top_keys else top_keys[0]
    correct = next(g["correct"] for k, (_, g) in zip(keys, members) if k == pick)
    samp = [g["correct"] for _, g in members[1:]]
    return float(correct), float(np.mean(samp)) if samp else float("nan")


def build_frame(model_dir: str | Path, judges: list[str] | None = None) -> pd.DataFrame:
    d = Path(model_dir)
    gens = read_jsonl(d / "generations.jsonl")
    feats = {r["id"]: r["features"] for r in read_jsonl(d / "features.jsonl")}
    jud = {j: {r["id"]: r for r in read_jsonl(d / f"judge_{j}.jsonl")} for j in (judges or [])}
    rows = []
    for g in gens:
        ex = g["example"]
        if ex["id"] not in feats:
            continue
        vc, sacc = _vote_correct(g)
        row = {
            "id": ex["id"], "model": g["model"], "task": ex["task"], "split": g["split"],
            "error": int(not g["grade"]["correct"]), "error_type": g["grade"]["error_type"],
            "dataset": ex["meta"].get("dataset"), "difficulty": ex["meta"].get("difficulty", np.nan),
            "vote_error": int(not vc), "sample_acc": sacc,
            "ctl_q_words": len(words(ex["meta"].get("question", ex["prompt"]))),
            "ctl_n_tokens": g["greedy"]["n_new_tokens"],
            "lat_greedy_s": g["greedy"]["latency_s"],
            "lat_samples_s": float(sum(s["latency_s"] for s in g["samples"])),
            "tok_greedy": g["greedy"]["n_new_tokens"],
            "tok_samples": int(sum(s["n_new_tokens"] for s in g["samples"])),
            "n_samples": len(g["samples"]), "answer_text": g["greedy"]["text"],
            "extracted": g["grade"]["extracted"],
        }
        row.update(feats[ex["id"]])
        for j, m in jud.items():
            r = m.get(ex["id"])
            row[f"bl_judge_{j}"] = r["p_yes"] if r else np.nan
            row[f"lat_judge_{j}"] = r["latency_s"] if r else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Label-free anomaly scores: Mahalanobis distance of the affect / epistemic vector from the TRAIN centroid of the
    same (model, task). Uses train rows only (no test statistics), and no labels."""
    df = df.copy()
    specs = {"af_anomaly": [c for c in df.columns if c.startswith("af_") and (c.endswith("_mean") or c == "af_valence")],
             "ep_anomaly": [c for c in df.columns if c.startswith("ep_") and c.endswith("per100")]}
    for out, cols in specs.items():
        df[out] = np.nan
        if not cols:
            continue
        for (m, t), g in df.groupby(["model", "task"]):
            tr = g[g.split == "train"][cols].fillna(0.0)
            if len(tr) < 10:
                continue
            mu = tr.mean().values
            cov = np.cov(tr.values.T) + 1e-3 * np.eye(len(cols))
            inv = np.linalg.pinv(cov)
            x = g[cols].fillna(0.0).values - mu
            df.loc[g.index, out] = np.sqrt(np.einsum("ij,jk,ik->i", x, inv, x))
    return df
