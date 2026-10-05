"""Signal-extractor interface and the label-free context passed to extractors.

Extractors never see ground truth: ``AnswerContext`` carries the prompt, the greedy answer, and the sampled
candidates only. This is the structural guard against label leakage into features.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from harness.core.types import Example, Generation

GROUPS = ("affect", "epistemic", "semantic", "token", "candidate", "linguistic")


@dataclass
class AnswerContext:
    example_id: str
    task: str
    question: str  # the task question without instructions/few-shot boilerplate
    prompt: str
    greedy: Generation
    samples: list[Generation] = field(default_factory=list)
    final_answers: list[str | None] = field(default_factory=list)  # [greedy, *samples], parsed (label-free)


@runtime_checkable
class SignalExtractor(Protocol):
    name: str
    group: str  # one of GROUPS

    def extract(self, ctx: AnswerContext) -> dict[str, float]:
        """Return feature_name -> value. NaN is allowed for 'not applicable'."""
        ...
