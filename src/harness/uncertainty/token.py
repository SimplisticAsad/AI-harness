"""Token-level uncertainty from the model's raw next-token distributions (logprobs / entropy)."""
from __future__ import annotations

import numpy as np

from harness.core.types import TokenTrace
from harness.signals.base import AnswerContext

_NAN = float("nan")


def _answer_span(trace: TokenTrace, answer: str | None) -> list[int]:
    """Indices of the tokens that spell the final answer (searched from the end); fallback: last 2 tokens."""
    n = len(trace.tokens)
    if n == 0:
        return []
    if answer:
        a = answer.lower().replace(",", "")
        for end in range(n, 0, -1):
            acc = ""
            for start in range(end - 1, max(end - 7, -1), -1):
                acc = trace.tokens[start].replace("▁", "").lower() + acc
                if acc.strip(".") == a:
                    return list(range(start, end))
    return list(range(max(0, n - 3), max(0, n - 1))) or [n - 1]  # skip trailing EOS/period


def trace_features(trace: TokenTrace | None, answer: str | None, prefix: str = "tk") -> dict[str, float]:
    if trace is None or not trace.logprobs:
        return {f"{prefix}_{k}": _NAN for k in (
            "mean_logprob", "min_prob", "mean_entropy", "max_entropy", "std_logprob", "seq_logprob", "entropy_spikes",
            "mean_top1", "sent_start_entropy", "digit_entropy", "ans_mean_logprob", "ans_min_prob", "ans_mean_entropy")}
    lp = np.asarray(trace.logprobs[:-1] or trace.logprobs)  # drop EOS token (always near-certain, length artefact)
    en = np.asarray(trace.entropies[:len(lp)])
    t1 = np.asarray(trace.top1_probs[:len(lp)])
    toks = trace.tokens[:len(lp)]
    starts = [i for i, t in enumerate(toks) if i == 0 or toks[i - 1].strip("▁") in (".", "!", "?")]
    digits = [i for i, t in enumerate(toks) if any(c.isdigit() for c in t)]
    span = [i for i in _answer_span(trace, answer) if i < len(lp)] or [len(lp) - 1]
    return {
        f"{prefix}_mean_logprob": float(lp.mean()),
        f"{prefix}_min_prob": float(np.exp(lp.min())),
        f"{prefix}_mean_entropy": float(en.mean()),
        f"{prefix}_max_entropy": float(en.max()),
        f"{prefix}_std_logprob": float(lp.std()),
        f"{prefix}_seq_logprob": float(lp.sum()),
        f"{prefix}_entropy_spikes": float((en > 2.0).sum()),  # count of tokens with >2 nats of entropy
        f"{prefix}_mean_top1": float(t1.mean()),
        f"{prefix}_sent_start_entropy": float(en[starts].mean()) if starts else _NAN,  # reasoning transitions
        f"{prefix}_digit_entropy": float(en[digits].mean()) if digits else _NAN,  # numeric / factual tokens
        f"{prefix}_ans_mean_logprob": float(lp[span].mean()),
        f"{prefix}_ans_min_prob": float(np.exp(lp[span].min())),
        f"{prefix}_ans_mean_entropy": float(en[span].mean()),
    }


class TokenUncertaintyExtractor:
    name = "token_uncertainty"
    group = "token"

    def extract(self, ctx: AnswerContext) -> dict[str, float]:
        return trace_features(ctx.greedy.trace, ctx.final_answers[0] if ctx.final_answers else None)
