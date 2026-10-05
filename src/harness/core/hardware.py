"""Hardware detection, used both for model selection and for the reproducibility manifest."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass

import psutil


@dataclass
class HardwareInfo:
    cpu_model: str
    cpu_count: int
    ram_gb: float
    gpu: str
    platform: str
    python: str

    def to_dict(self) -> dict:
        return asdict(self)


def detect_hardware() -> HardwareInfo:
    cpu = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    gpu = "none"
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
        except Exception:  # noqa: BLE001
            gpu = "nvidia-smi failed"
    return HardwareInfo(cpu, os.cpu_count() or 1, round(psutil.virtual_memory().total / 2**30, 1), gpu,
                        platform.platform(), platform.python_version())


def max_model_params_millions(hw: HardwareInfo) -> float:
    """Rough feasibility rule: fp32 weights must use <=40% of RAM; CPU-only caps at ~3B for tractable runtime."""
    ram_cap = hw.ram_gb * 0.4 * 1024 / 4 * 1.0  # millions of fp32 params
    return min(ram_cap, 3000.0 if hw.gpu == "none" else 1e9)
