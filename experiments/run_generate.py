"""Stage 1 CLI: python experiments/run_generate.py --config configs/pilot.yaml --model flan_t5_small"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

from benchmarks.generators import build_all  # noqa: E402
from benchmarks.graders import grade  # noqa: E402
from harness.core.config import ExperimentConfig  # noqa: E402
from harness.core.hardware import detect_hardware  # noqa: E402
from harness.core.io import write_manifest  # noqa: E402
from harness.core.pipeline import generate_records  # noqa: E402
from harness.models.flan_t5 import FlanT5Model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap examples per task (debug)")
    a = ap.parse_args()
    cfg = ExperimentConfig.load(a.config)
    mc = next(m for m in cfg.models if m.name == a.model)
    n_per = {t: (min(n, a.limit) if a.limit else n) for t, n in (mc.n_per_task or cfg.data.n_per_task).items()}
    ex = build_all(n_per, cfg.data.seed)
    g = cfg.generation
    model = FlanT5Model(mc.name, mc.hf_dir, mc.spiece, mc.params_millions, mc.threads or g.threads, g.batch_size)
    out = Path(cfg.results_dir) / cfg.experiment / mc.name
    write_manifest(out / "manifest_generate.json", config=cfg.model_dump(), model=model.info.__dict__,
                   hardware=detect_hardware().to_dict(), seed=g.seed, temperature_sample=g.sample_temperature,
                   greedy_temperature=0.0, max_new_tokens=g.max_new_tokens, data_seed=cfg.data.seed,
                   dataset_versions={"ARC": "ARC-V1-Feb2018-2 (test)", "GSM8K": "openai/grade-school-math test",
                                     "synthetic": "benchmarks/generators.py"})
    generate_records(model, ex, out / "generations.jsonl", grade, n_samples=g.n_samples,
                     temperature=g.sample_temperature, max_new_tokens=g.max_new_tokens, seed=g.seed,
                     fractions=cfg.data.split_fractions)


if __name__ == "__main__":
    main()
