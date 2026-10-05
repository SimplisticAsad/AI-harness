"""Typed experiment configuration (YAML -> pydantic)."""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    hf_dir: str
    spiece: str = ".cache/spiece.model"
    family: str = "flan-t5"
    params_millions: float
    n_per_task: dict[str, int] | None = None  # per-model override (cost control)
    threads: int | None = None


class GenerationConfig(BaseModel):
    max_new_tokens: dict[str, int] = Field(default_factory=lambda: {
        "factual_qa": 48, "math": 96, "logic": 48, "longform": 64, "instruction": 64, "code": 40})
    sample_temperature: float = 0.7
    n_samples: int = 5
    batch_size: int = 8
    threads: int = 4
    seed: int = 1234


class DataConfig(BaseModel):
    n_per_task: dict[str, int]
    seed: int = 7
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)


class ExperimentConfig(BaseModel):
    experiment: str
    models: list[ModelConfig]
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    data: DataConfig
    results_dir: str = "results"
    bootstrap_resamples: int = 1000
    stat_seed: int = 0

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        return ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))
