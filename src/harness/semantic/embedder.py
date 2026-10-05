"""Frozen-encoder sentence embeddings (mean-pooled Flan-T5 encoder states) with an on-disk cache.

Sentence-transformers / Hub encoders were unreachable in the build environment, so the analysis encoder is the
encoder half of an open Flan-T5 checkpoint. It is *not* a similarity-tuned model; this is a documented limitation.
The same fixed encoder is used for every evaluated LLM so features are comparable across models.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import torch
from transformers import T5ForConditionalGeneration

from harness.models.tokenizer import T5SPTokenizer


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class T5Embedder:
    def __init__(self, hf_dir: str | Path, spiece: str | Path, name: str = "flan_t5_base_encoder",
                 batch_size: int = 32, cache_path: str | Path | None = None, max_len: int = 128):
        self.name = name
        self.tok = T5SPTokenizer(spiece)
        self.enc = T5ForConditionalGeneration.from_pretrained(str(hf_dir)).encoder.eval()
        self.batch_size, self.max_len = batch_size, max_len
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path and self.cache_path.exists():
            z = np.load(self.cache_path, allow_pickle=True)
            self.cache = dict(zip(z["keys"].tolist(), z["vals"]))

    @staticmethod
    def _key(t: str) -> str:
        return hashlib.md5(t.encode()).hexdigest()

    @torch.inference_mode()
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        todo = sorted({t for t in texts if self._key(t) not in self.cache}, key=len)
        for b in range(0, len(todo), self.batch_size):
            chunk = todo[b: b + self.batch_size]
            ids, mask = self.tok.batch([c if c.strip() else "." for c in chunk], self.max_len)
            h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            v = ((h * m).sum(1) / m.sum(1)).numpy()
            for c, vec in zip(chunk, v):
                self.cache[self._key(c)] = vec.astype(np.float32)
        return np.stack([self.cache[self._key(t)] for t in texts]) if texts else np.zeros((0, 1))

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, keys=np.array(list(self.cache.keys())), vals=np.stack(list(self.cache.values())))
