"""Evidence graph: passages as nodes, containment links as directed edges, PageRank as trust.

Edge i -> j ("i cites j") exists when most of passage i's content terms also occur in passage j (and they share at least
``min_shared`` terms). A passage many others are covered by therefore collects many *incoming* edges, i.e. it is
corroborated. This is a CONSTRUCTED corroboration graph: the available corpus has no real hyperlinks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

WORD = re.compile(r"[a-z0-9]+")
STOP = set("""a an the of to in on at by for from with and or but is are was were be been being it its this that these those
as which who whom what when where why how not no do does did can could will would should may might must than then so such
into out up down over under about between among also more most some any each other their there they them his her he she we you
your our i if has have had one two three four of""".split())


def content_terms(text: str) -> set[str]:
    return {t for t in WORD.findall(text.lower()) if len(t) >= 3 and t not in STOP}


@dataclass
class Node:
    id: str
    text: str
    bm25: float
    poison: bool = False
    pagerank: float = 0.0
    in_deg: int = 0
    out_deg: int = 0


@dataclass
class EvidenceGraph:
    nodes: list[Node]
    edges: list[tuple[int, int, float]] = field(default_factory=list)  # (src_idx, dst_idx, weight)

    def to_json(self) -> dict:
        return {"nodes": [n.__dict__ for n in self.nodes],
                "edges": [{"src": self.nodes[i].id, "dst": self.nodes[j].id, "w": round(w, 3)} for i, j, w in self.edges]}


def build_graph(nodes: list[Node], min_cover: float = 0.5, min_shared: int = 4, damping: float = 0.85,
                iters: int = 60) -> EvidenceGraph:
    terms = [content_terms(n.text) for n in nodes]
    edges: list[tuple[int, int, float]] = []
    for i, ti in enumerate(terms):
        if not ti:
            continue
        for j, tj in enumerate(terms):
            if i == j:
                continue
            shared = len(ti & tj)
            w = shared / len(ti)
            if shared >= min_shared and w >= min_cover:
                edges.append((i, j, w))
    g = EvidenceGraph(nodes, edges)
    n = len(nodes)
    for i, j, _ in edges:
        nodes[i].out_deg += 1
        nodes[j].in_deg += 1
    pr = pagerank(n, edges, damping, iters)
    for node, p in zip(nodes, pr):
        node.pagerank = float(p)
    return g


def pagerank(n: int, edges: list[tuple[int, int, float]], d: float = 0.85, iters: int = 60) -> np.ndarray:
    """Weighted PageRank by power iteration; dangling nodes redistribute uniformly."""
    if n == 0:
        return np.zeros(0)
    W = np.zeros((n, n))
    for i, j, w in edges:
        W[i, j] += w
    out = W.sum(1)
    pr = np.full(n, 1.0 / n)
    for _ in range(iters):
        contrib = np.where(out > 0, pr / np.where(out > 0, out, 1.0), 0.0)
        dangling = pr[out == 0].sum()
        pr = (1 - d) / n + d * (contrib @ W + dangling / n)
    return pr / pr.sum()
