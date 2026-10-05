"""Minimal SentencePiece wrapper for the T5 vocabulary (pad=0, eos=1, unk=2).

Avoids depending on the Hugging Face tokenizer classes, whose constructors changed across majors.
"""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch

PAD_ID, EOS_ID = 0, 1


class T5SPTokenizer:
    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str, max_len: int = 512) -> list[int]:
        return self.sp.encode(text)[: max_len - 1] + [EOS_ID]

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([i for i in ids if i > 2 and i < 32000])

    def pieces(self, ids: list[int]) -> list[str]:
        return [self.sp.id_to_piece(i) if i < 32000 else f"<extra_{i}>" for i in ids]

    def batch(self, texts: list[str], max_len: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
        enc = [self.encode(t, max_len) for t in texts]
        width = max(len(e) for e in enc)
        ids = torch.full((len(enc), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(enc), width), dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, : len(e)] = torch.tensor(e)
            mask[i, : len(e)] = 1
        return ids, mask
