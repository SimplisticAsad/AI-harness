"""Semantic signals: coherence, drift, relevance, self-contradiction heuristics, and cross-candidate agreement."""
from __future__ import annotations

import itertools
import re

import numpy as np

from harness.semantic.embedder import Embedder
from harness.signals.base import AnswerContext
from harness.signals.text import jaccard, sentences, words
from harness.signals.linguistic import linguistic_features  # noqa: F401  (re-export convenience)

_NEG = {"not", "no", "never", "n't", "cannot", "none", "neither", "nor", "isn't", "aren't", "wasn't", "don't"}
_STOP = set("the a an of to and in is are was were be it that this for on with as by at from or".split())


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _has_neg(toks: list[str]) -> bool:
    return any(t in _NEG or t.endswith("n't") for t in toks)


def contradiction_pairs(sents: list[str]) -> int:
    """Heuristic claim-level contradictions: two sentences with high lexical overlap but opposite negation,
    or the same sentence frame with different numbers. Deliberately conservative; NOT an NLI model."""
    n = 0
    toks = [words(s) for s in sents]
    for i, j in itertools.combinations(range(len(sents)), 2):
        ci = [t for t in toks[i] if t not in _STOP and t not in _NEG]
        cj = [t for t in toks[j] if t not in _STOP and t not in _NEG]
        if len(ci) < 2 or len(cj) < 2:
            continue
        ov = jaccard(ci, cj)
        if ov >= 0.6 and _has_neg(toks[i]) != _has_neg(toks[j]):
            n += 1
        else:
            ni, nj = re.findall(r"\d+\.?\d*", sents[i]), re.findall(r"\d+\.?\d*", sents[j])
            if ov >= 0.5 and ni and nj and set(ni) != set(nj) and len(ni) == len(nj) == 1 and ov < 1.0:
                n += 1
    return n


class SemanticExtractor:
    name = "semantic_text"
    group = "semantic"

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def extract(self, ctx: AnswerContext) -> dict[str, float]:
        text = ctx.greedy.text
        sents = sentences(text) or [text]
        embs = self.embedder.embed([ctx.question, text] + sents)
        q, a, se = embs[0], embs[1], embs[2:]
        adj = [_cos(se[i], se[i + 1]) for i in range(len(se) - 1)]
        pw = [w for w in words(ctx.prompt) if w not in _STOP]
        aw = [w for w in words(text) if w not in _STOP]
        novel = (len([w for w in aw if w not in set(pw)]) / len(aw)) if aw else float("nan")
        return {
            "sm_qa_similarity": _cos(q, a),
            "sm_coherence": float(np.mean(adj)) if adj else float("nan"),
            "sm_drift_first_last": _cos(se[0], se[-1]) if len(se) > 1 else float("nan"),
            "sm_min_adjacent_sim": float(np.min(adj)) if adj else float("nan"),
            "sm_self_contradictions": float(contradiction_pairs(sents)),
            "sm_begin_end_contradiction": float(contradiction_pairs([sents[0], sents[-1]]) if len(sents) > 1 else 0),
            # Proxy for 'unsupported claims': share of answer content words absent from the prompt.
            "sm_novel_word_ratio": novel,
        }


class CandidateAgreementExtractor:
    """Agreement among the greedy answer and N sampled candidates (answer-level, semantic, lexical, affective, epistemic)."""

    name = "candidate_agreement"
    group = "candidate"

    def __init__(self, embedder: Embedder, affect_fn=None, epistemic_fn=None):
        self.embedder, self.affect_fn, self.epistemic_fn = embedder, affect_fn, epistemic_fn

    def extract(self, ctx: AnswerContext) -> dict[str, float]:
        gens = [ctx.greedy] + ctx.samples
        n = len(gens)
        nan = float("nan")
        if n < 2:
            return {k: nan for k in ("cd_answer_agreement", "cd_vote_frac", "cd_answer_entropy", "cd_n_distinct",
                                      "cd_unparsed_frac", "cd_semantic_agreement", "cd_semantic_min",
                                      "cd_lexical_agreement", "cd_epistemic_std", "cd_affect_std", "cd_sample_mean_logprob")}
        ans = [a if a is not None else "<none>" for a in ctx.final_answers]
        votes: dict[str, int] = {}
        for a in ans:
            votes[a] = votes.get(a, 0) + 1
        p = np.array(list(votes.values())) / n
        embs = self.embedder.embed([g.text for g in gens])
        sims = [_cos(embs[i], embs[j]) for i, j in itertools.combinations(range(n), 2)]
        toks = [words(g.text) for g in gens]
        lex = [jaccard(toks[i], toks[j]) for i, j in itertools.combinations(range(n), 2)]
        from harness.epistemic.cues import epistemic_profile
        ep = [epistemic_profile(g.text)["net_certainty"] for g in gens]
        af = [self.affect_fn(g.text) for g in gens] if self.affect_fn else [0.0] * n
        slp = [float(np.mean(g.trace.logprobs[:-1] or g.trace.logprobs)) for g in ctx.samples if g.trace and g.trace.logprobs]
        return {
            "cd_answer_agreement": float(np.mean([a == ans[0] for a in ans[1:]])),
            "cd_vote_frac": float(p.max()),
            "cd_answer_entropy": float(-(p * np.log(p)).sum()),
            "cd_n_distinct": float(len(votes)),
            "cd_unparsed_frac": float(np.mean([a == "<none>" for a in ans])),
            "cd_semantic_agreement": float(np.mean(sims)),
            "cd_semantic_min": float(np.min(sims)),
            "cd_lexical_agreement": float(np.mean(lex)),
            "cd_epistemic_std": float(np.std(ep)),
            "cd_affect_std": float(np.std(af)),
            "cd_sample_mean_logprob": float(np.mean(slp)) if slp else nan,
        }
