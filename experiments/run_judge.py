"""Stage 1b: LLM-as-a-judge scores. Self-judge (same model) and independent judge (flan_t5_large) for every record.

python experiments/run_judge.py --config configs/main.yaml --target flan_t5_small --judge flan_t5_large
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

from harness.core.config import ExperimentConfig  # noqa: E402
from harness.core.io import read_jsonl, write_jsonl  # noqa: E402
from harness.models.flan_t5 import FlanT5Model  # noqa: E402
from harness.verification.judge import LLMJudge  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    cfg = ExperimentConfig.load(a.config)
    jm = next(m for m in cfg.models if m.name == a.judge)
    base = Path(cfg.results_dir) / cfg.experiment / a.target
    recs = read_jsonl(base / "generations.jsonl")
    out = base / f"judge_{a.judge}.jsonl"
    done = {r["id"] for r in read_jsonl(out)}
    recs = [r for r in recs if r["example"]["id"] not in done]
    judge = LLMJudge(FlanT5Model(jm.name, jm.hf_dir, jm.spiece, jm.params_millions, a.threads, 16), jm.name)
    for i in range(0, len(recs), 64):
        chunk = recs[i: i + 64]
        pg, dt = judge.p_correct([r["example"]["prompt"] for r in chunk], [r["greedy"]["text"] for r in chunk])
        write_jsonl(out, [{"id": r["example"]["id"], "p_yes": float(p), "latency_s": dt} for r, p in zip(chunk, pg)], "a")
        print(a.target, a.judge, i + len(chunk), "/", len(recs), flush=True)


if __name__ == "__main__":
    main()
