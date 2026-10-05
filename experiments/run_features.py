"""Stage 2 CLI: python experiments/run_features.py --config configs/main.yaml --model flan_t5_small"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

from harness.core.config import ExperimentConfig  # noqa: E402
from harness.core.io import read_jsonl, write_jsonl, write_manifest  # noqa: E402
from harness.signals.factory import build_extractors  # noqa: E402
from harness.signals.pipeline import extract_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    cfg = ExperimentConfig.load(a.config)
    base = Path(cfg.results_dir) / cfg.experiment / a.model
    recs = read_jsonl(base / "generations.jsonl")
    out = base / "features.jsonl"
    done = {r["id"] for r in read_jsonl(out)}
    recs = [r for r in recs if r["example"]["id"] not in done]
    extractors, emb = build_extractors()
    t0 = time.perf_counter()
    for i in range(0, len(recs), 50):
        write_jsonl(out, extract_features(recs[i: i + 50], extractors), "a")
        emb.save()
        print(a.model, i + 50, "/", len(recs), f"{(time.perf_counter() - t0) / (i + 50):.2f}s/rec", flush=True)
    write_manifest(base / "manifest_features.json", extractors=[e.name for e in extractors],
                   analysis_encoder="flan_t5_base encoder (mean-pooled, frozen)", n_records=len(recs),
                   seconds_per_record=(time.perf_counter() - t0) / max(len(recs), 1))


if __name__ == "__main__":
    main()
