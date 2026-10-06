"""Stage G1: retrieve passages for each ARC question from ARC_Corpus.txt (14.6M sentences) and build evidence graphs.

For every question we store (a) a CLEAN graph from the top-30 retrieved passages and (b) a POISONED graph that adds
5 mutually-corroborating false passages asserting a randomly chosen WRONG option (seeded by question id, independent of
any model output). Trust = PageRank of the containment-link graph.

python experiments/graph_build.py --n 300
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

from benchmarks.generators import build_factual_qa  # noqa: E402
from harness.evidence.graph import Node, build_graph, content_terms  # noqa: E402

CORPUS = Path("data/raw/ARC-V1-Feb2018-2/ARC_Corpus.txt")
WORD = re.compile(r"[a-z0-9]+")
TOPK_PER_Q, GRAPH_NODES = 80, 30
_G: dict = {}


def parse_question(ex) -> tuple[str, list[str]]:
    stem = ex.meta["question"]
    opts = re.findall(r"^[ABCD]\. (.*)$", ex.prompt, re.M)
    return stem, opts


def _init(term2q, qterms, idf):
    _G.update(term2q=term2q, qterms=qterms, idf=idf)


def score_terms(hit_terms: set[str], qi: int, n_toks: int) -> float:
    q, idf = _G["qterms"][qi], _G["idf"]
    return sum(idf.get(t, 0.0) for t in hit_terms if t in q) / (n_toks ** 0.25)


def _scan(rng: tuple[int, int]):
    start, end = rng
    term2q, qterms = _G["term2q"], _G["qterms"]
    heaps: dict[int, list] = defaultdict(list)
    with open(CORPUS, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()
        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            if not (40 <= len(raw) <= 350):
                continue
            line = raw.decode("utf-8", "ignore").strip()
            toks = set(WORD.findall(line.lower()))
            hit = toks & term2q.keys()
            if len(hit) < 3:
                continue
            cnt: Counter = Counter()
            for t in hit:
                for qi in term2q[t]:
                    cnt[qi] += 1
            for qi, c in cnt.items():
                if c >= 3:
                    s = score_terms(hit, qi, len(toks))
                    h = heaps[qi]
                    if len(h) < TOPK_PER_Q:
                        heapq.heappush(h, (s, line))
                    elif s > h[0][0]:
                        heapq.heapreplace(h, (s, line))
    return dict(heaps)


def make_poison(stem: str, opt: str, rng: random.Random) -> list[str]:
    fillers = [("It is well known", "Experts agree"), ("Studies show", "Textbooks state"), ("Research confirms", "Scientists report"),
               ("It has been established", "Sources confirm"), ("Reference works note", "Teachers explain")]
    base = re.sub(r"\s+", " ", stem.strip().rstrip("?"))
    out = []
    for a, b in fillers:
        out.append(f"{a} that regarding the question {base}, the correct answer is {opt}. {b} that {opt} is the answer to {base}.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default="results/graph")
    a = ap.parse_args()
    exs = build_factual_qa(a.n, 7)
    qs = [parse_question(e) for e in exs]
    qterms = [content_terms(s + " " + " ".join(o)) for s, o in qs]
    term2q: dict[str, list[int]] = defaultdict(list)
    for qi, T in enumerate(qterms):
        for t in T:
            term2q[t].append(qi)
    # approximate idf from a sample of the first 400k lines
    df: Counter = Counter()
    N = 0
    with open(CORPUS, "rb") as f:
        for _ in range(400_000):
            raw = f.readline()
            if not raw:
                break
            N += 1
            for t in set(WORD.findall(raw.decode("utf-8", "ignore").lower())) & term2q.keys():
                df[t] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) for t in term2q}
    size = CORPUS.stat().st_size
    cuts = [(i * size // a.procs, (i + 1) * size // a.procs) for i in range(a.procs)]
    t0 = time.time()
    with mp.Pool(a.procs, initializer=_init, initargs=(dict(term2q), qterms, idf)) as pool:
        parts = pool.map(_scan, cuts)
    print("scan seconds", round(time.time() - t0), flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    _init(dict(term2q), qterms, idf)
    for qi, ex in enumerate(exs):
        stem, opts = qs[qi]
        cand: dict[str, float] = {}
        for p in parts:
            for s, line in p.get(qi, []):
                k = line.lower()
                if k not in cand or s > cand[k][0]:
                    cand[k] = (s, line)
        top = sorted(cand.values(), reverse=True)[:GRAPH_NODES]
        clean_nodes = [Node(f"d{i}", t, float(s)) for i, (s, t) in enumerate(top)]
        clean = build_graph([Node(**n.__dict__) for n in clean_nodes])
        h = int(hashlib.sha256(ex.id.encode()).hexdigest()[:8], 16)
        wrong = [i for i, _ in enumerate(opts) if "ABCD"[i] != ex.reference]
        pw = wrong[h % len(wrong)]
        rng = random.Random(h)
        pnodes = []
        for k, txt in enumerate(make_poison(stem, opts[pw], rng)):
            toks = set(WORD.findall(txt.lower()))
            pnodes.append(Node(f"p{k}", txt, score_terms(toks & qterms[qi], qi, len(toks)), poison=True))
        poisoned = build_graph([Node(**n.__dict__) for n in clean_nodes] + pnodes)
        rows.append({"id": ex.id, "stem": stem, "options": opts, "gold": ex.reference, "poison_option": "ABCD"[pw],
                     "clean": clean.to_json(), "poisoned": poisoned.to_json()})
    with open(out / "graphs.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_nodes = [len(r["clean"]["nodes"]) for r in rows]
    n_edges = [len(r["clean"]["edges"]) for r in rows]
    print("questions", len(rows), "mean nodes", sum(n_nodes) / len(rows), "mean edges", sum(n_edges) / len(rows), "min nodes", min(n_nodes))


if __name__ == "__main__":
    main()
