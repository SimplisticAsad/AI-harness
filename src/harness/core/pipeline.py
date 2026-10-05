"""Stage 1: generate greedy + sampled answers for each example and grade them with deterministic graders."""
from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Callable, Sequence

import psutil

from harness.core.io import read_jsonl, split_of, write_jsonl
from harness.core.log import get_logger
from harness.core.types import Example, Generation, Grade
from harness.models.base import LanguageModel

log = get_logger()


def _batches(todo: Sequence[Example], chunk: int) -> list[list[Example]]:
    """Homogeneous-task batches (per-task token limits, less padding)."""
    out: list[list[Example]] = []
    for _, grp in itertools.groupby(sorted(todo, key=lambda e: e.task), key=lambda e: e.task):
        g = list(grp)
        out.extend(g[i: i + chunk] for i in range(0, len(g), chunk))
    return out


def generate_records(model: LanguageModel, examples: Sequence[Example], out_path: str | Path,
                     grader: Callable[[Example, str], Grade], *, n_samples: int, temperature: float,
                     max_new_tokens: int | dict[str, int], seed: int, chunk: int = 16,
                     fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> None:
    """Resumable: examples already present in ``out_path`` are skipped."""
    done = {r["example"]["id"] for r in read_jsonl(out_path)}
    todo = [e for e in examples if e.id not in done]
    log.info("generate", extra={"ctx": {"model": model.info.name, "todo": len(todo), "done": len(done)}})
    proc = psutil.Process()
    n_done = 0
    for batch in _batches(todo, chunk):
        mnt = max_new_tokens[batch[0].task] if isinstance(max_new_tokens, dict) else max_new_tokens
        prompts = [e.prompt for e in batch]
        t0 = time.perf_counter()
        cpu0 = sum(proc.cpu_times()[:2])
        greedy = model.generate(prompts, n=1, temperature=0.0, max_new_tokens=mnt, seed=seed)
        samples = (model.generate(prompts, n=n_samples, temperature=temperature, max_new_tokens=mnt, seed=seed + 1)
                   if n_samples > 0 else [[] for _ in batch])
        wall = time.perf_counter() - t0
        cpu = sum(proc.cpu_times()[:2]) - cpu0
        rows = []
        for e, g, ss in zip(batch, greedy, samples):
            g0: Generation = g[0]
            rows.append({
                "example": e.to_dict(), "model": model.info.name, "split": split_of(e.id, fractions),
                "greedy": g0.to_dict(), "grade": grader(e, g0.text).to_dict(),
                "samples": [s.to_dict() for s in ss],
                "sample_grades": [grader(e, s.text).to_dict() for s in ss],
                "cost": {"wall_s_per_example": wall / len(batch), "cpu_s_per_example": cpu / len(batch),
                         "rss_mb": proc.memory_info().rss / 2**20, "model_calls": 1 + n_samples,
                         "new_tokens": g0.n_new_tokens + sum(s.n_new_tokens for s in ss)},
            })
        write_jsonl(out_path, rows, mode="a")
        n_done += len(batch)
        log.info("progress", extra={"ctx": {"model": model.info.name, "task": batch[0].task, "n": n_done,
                                            "of": len(todo), "s_per_ex": round(wall / len(batch), 2)}})
