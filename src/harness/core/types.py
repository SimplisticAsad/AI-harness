"""Core data types shared by every layer of the harness."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Example:
    """One benchmark item. ``reference`` is ground truth from a dataset or deterministic generator."""

    id: str
    task: str
    prompt: str
    reference: Any
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. question text, difficulty, dataset name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Example":
        return Example(d["id"], d["task"], d["prompt"], d["reference"], d.get("meta", {}))


@dataclass
class TokenTrace:
    """Per-generated-token statistics from the *raw* (temperature-1) model distribution."""

    tokens: list[str]
    logprobs: list[float]  # log p(chosen token)
    entropies: list[float]  # entropy (nats) of the full next-token distribution
    top1_probs: list[float]  # max_v p(v)


@dataclass
class Generation:
    text: str
    trace: TokenTrace | None = None
    latency_s: float = 0.0  # batch wall-time divided by batch size (documented approximation)
    n_new_tokens: int = 0
    n_prompt_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Generation":
        tr = TokenTrace(**d["trace"]) if d.get("trace") else None
        return Generation(d["text"], tr, d.get("latency_s", 0.0), d.get("n_new_tokens", 0), d.get("n_prompt_tokens", 0))


@dataclass
class Grade:
    correct: bool
    extracted: str | None
    error_type: str | None  # taxonomy label from the task grader, None when correct
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelInfo:
    name: str
    family: str
    params_millions: float
    quantization: str
    framework: str
    version: str = ""
