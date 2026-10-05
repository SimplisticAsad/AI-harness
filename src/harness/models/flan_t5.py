"""Hugging Face seq2seq backend (Flan-T5) with per-token logprob / entropy traces, CPU friendly."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import T5ForConditionalGeneration

from harness.core.types import Generation, ModelInfo, TokenTrace
from harness.models.tokenizer import EOS_ID, PAD_ID, T5SPTokenizer


class FlanT5Model:
    def __init__(self, name: str, hf_dir: str | Path, spiece: str | Path, params_millions: float,
                 threads: int = 4, batch_size: int = 8):
        torch.set_num_threads(threads)
        self.tok = T5SPTokenizer(spiece)
        self.model = T5ForConditionalGeneration.from_pretrained(str(hf_dir), torch_dtype=torch.float32).eval()
        self.batch_size = batch_size
        self.info = ModelInfo(name=name, family="flan-t5", params_millions=params_millions, quantization="fp32",
                              framework=f"transformers-{__import__('transformers').__version__}+torch-{torch.__version__}",
                              version="google/flan-t5 T5X checkpoint (gs://t5-data) converted locally")

    # ------------------------------------------------------------------ generation
    @torch.inference_mode()
    def generate(self, prompts: Sequence[str], *, n: int = 1, temperature: float = 0.0,
                 max_new_tokens: int = 96, seed: int | None = None) -> list[list[Generation]]:
        if temperature == 0.0 and n != 1:
            raise ValueError("greedy decoding requires n == 1")
        out: list[list[Generation]] = [[] for _ in prompts]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # reduce padding
        for b0 in range(0, len(order), self.batch_size):
            idx = order[b0: b0 + self.batch_size]
            if seed is not None:
                torch.manual_seed(seed + b0)
            ids, mask = self.tok.batch([prompts[i] for i in idx])
            t0 = time.perf_counter()
            kw = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0,
                      num_return_sequences=n) if temperature > 0 else dict(do_sample=False)
            res = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                      output_logits=True, return_dict_in_generate=True, **kw)
            dt = (time.perf_counter() - t0) / (len(idx) * n)
            seqs = res.sequences[:, 1:]  # drop decoder start
            logits = torch.stack(res.logits, dim=1).float()  # (B*n, T, V)
            logp = torch.log_softmax(logits, dim=-1)
            ent = -(logp.exp() * logp).sum(-1)
            top1 = logp.max(-1).values.exp()
            chosen = logp.gather(-1, seqs[:, : logp.shape[1]].unsqueeze(-1)).squeeze(-1)
            for j, i in enumerate(idx):
                for k in range(n):
                    r = j * n + k
                    s = seqs[r].tolist()
                    L = s.index(EOS_ID) + 1 if EOS_ID in s else len(s)
                    L = min(L, logp.shape[1])
                    toks = s[:L]
                    trace = TokenTrace(tokens=self.tok.pieces(toks), logprobs=chosen[r, :L].tolist(),
                                       entropies=ent[r, :L].tolist(), top1_probs=top1[r, :L].tolist())
                    out[i].append(Generation(text=self.tok.decode(toks), trace=trace, latency_s=dt,
                                             n_new_tokens=L, n_prompt_tokens=int(mask[j].sum())))
        return out

    # ------------------------------------------------------------------ scoring
    @torch.inference_mode()
    def first_token_probs(self, prompts: Sequence[str], words: Sequence[str]) -> np.ndarray:
        wid = [self.tok.encode(w)[0] for w in words]
        res = np.zeros((len(prompts), len(words)), dtype=np.float64)
        for b0 in range(0, len(prompts), self.batch_size):
            chunk = list(prompts[b0: b0 + self.batch_size])
            ids, mask = self.tok.batch(chunk)
            dec = torch.full((len(chunk), 1), PAD_ID, dtype=torch.long)
            logits = self.model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec).logits[:, 0, :]
            p = torch.softmax(logits[:, wid].double(), dim=-1)
            res[b0: b0 + len(chunk)] = p.numpy()
        return res
