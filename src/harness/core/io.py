"""JSONL / manifest helpers and deterministic splitting."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        for r in rows:
            f.write(json.dumps(r, default=_default) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(path)


def _default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def split_of(example_id: str, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2), salt: str = "esd-v1") -> str:
    """Deterministic train/val/test assignment keyed on the example id (identical across models)."""
    h = int(hashlib.sha256(f"{salt}:{example_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < fractions[0]:
        return "train"
    return "val" if h < fractions[0] + fractions[1] else "test"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_manifest(path: str | Path, **fields: Any) -> None:
    fields.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    fields.setdefault("git_commit", git_commit())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(fields, indent=2, default=_default))
