"""Stage 2: turn generation records into feature rows (label-free extractors) + attach labels separately."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from harness.core.io import read_jsonl, write_jsonl
from harness.core.types import Example, Generation
from harness.signals.answers import extract_answer
from harness.signals.base import AnswerContext, SignalExtractor


def context_from_record(rec: dict) -> AnswerContext:
    ex = Example.from_dict(rec["example"])
    g = Generation.from_dict(rec["greedy"])
    ss = [Generation.from_dict(s) for s in rec["samples"]]
    answers = [extract_answer(ex.task, x.text) for x in [g, *ss]]
    return AnswerContext(ex.id, ex.task, ex.meta.get("question", ex.prompt), ex.prompt, g, ss, answers)


def extract_features(records: Sequence[dict], extractors: Sequence[SignalExtractor]) -> list[dict]:
    rows = []
    for rec in records:
        ctx = context_from_record(rec)
        feats: dict[str, float] = {}
        for ex in extractors:
            feats.update(ex.extract(ctx))
        rows.append({"id": ctx.example_id, "model": rec["model"], "task": ctx.task, "features": feats})
    return rows


def run_feature_stage(gen_path: str | Path, out_path: str | Path, extractors: Sequence[SignalExtractor]) -> None:
    recs = read_jsonl(gen_path)
    write_jsonl(out_path, extract_features(recs, extractors))
