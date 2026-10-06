# Evidence-graph self-correction experiment (follow-up)

**Question.** If a model first answers, then cross-checks its answer against an evidence graph in which passages are trusted by
PageRank (many incoming/outgoing links), does it correct its mistakes - and does graph trust beat plain retrieval?

**Setup (all in `experiments/graph_build.py`, `experiments/run_graph_experiment.py`; data in `results/graph/`).**
Task: ARC science questions (same items/models as the main study; 300 small, 200 base, 100 large). Evidence: `ARC_Corpus.txt`
(14.6M web sentences; no hyperlinks exist in it). Graph: top-30 retrieved passages per question; **constructed** corroboration graph where
passage i -> j when most of i's content terms occur in j; trust = PageRank (weighted, d=0.85). Graphs are stored as JSON
(`results/graph/graphs.jsonl`). The model re-answers with 3 passages and its previous answer; answer = argmax P(A-D). k=3 and all settings fixed
before running; nothing was tuned. "Poisoned" graph = + 5 mutually-corroborating false passages asserting a random wrong option
(they reuse the question's wording, so this is a strong, adversarial case). Caveat: C0 (original generated answer) is scored differently from the second pass,
so **C1 (re-ask without evidence) is the right control** for the evidence conditions.

## Results (accuracy; 95% bootstrap CI in `condition_summary.csv`)
| condition | small (n=300) | base (n=200) | large (n=100) |
|---|---|---|---|
| C0 original answer | 0.307 | 0.460 | 0.520 |
| C1 re-ask, no evidence (control) | 0.240 | 0.505 | 0.670 |
| C2 BM25 evidence | 0.293 | 0.550 | 0.700 |
| C3 graph-trust (PageRank) evidence | 0.310 | 0.475 | 0.680 |
| C4 shuffled-trust control | 0.303 | 0.520 | 0.680 |
| C5 poisoned + BM25 | 0.197 | 0.085 | 0.190 |
| C6 poisoned + PageRank | 0.207 | 0.130 | 0.300 |
| C7 poisoned + shuffled trust | 0.280 | 0.420 | 0.570 |

## Findings
1. **Clean evidence did not reliably help beyond simply asking again.** Against C1, no clean-evidence condition survives Holm correction (`vs_C1.csv`; smallest adjusted p = 0.052, small model C3).
   For base and large the apparent gains over C0 are mostly the effect of the second pass, not the evidence.
2. **Graph trust did not beat plain retrieval or random trust.** C3 vs C2: base 0.475 vs 0.550 (p=0.053), large 0.680 vs 0.700, small 0.310 vs 0.293 (all n.s.).
   C3 vs the shuffled-trust control C4: differences of -0.045, 0.000, +0.007 (all p>0.26). The PageRank scores added no measurable information here.
3. **Centrality is exploitable.** With coordinated false passages, PageRank ordering put poison in 65-66% of the top-3 (BM25: 91-92%), and accuracy collapsed far below the no-evidence control:
   C6 vs C1 = -0.375 (base) and -0.370 (large), both Holm p<0.001. Shuffled trust was the *least* damaged (poison share 8%), i.e. random selection was safer than trusting centrality.
   Corroboration by many linked sources is not truth; a coordinated falsehood is highly corroborated.
4. **The non-LLM lexical vote** (option overlap weighted by trust/BM25/uniform) beat the small model (0.447-0.477 vs 0.307) but PageRank weighting was no better than uniform or BM25 weighting, and for base/large it was below the models' own accuracy.

## Limitations
Constructed (not real hyperlink) graph; strong lexical poison; small models that read evidence poorly; one dataset; second-pass vs C0 scoring mismatch; no hyper-parameter search (a different edge rule or k could change the clean-graph result);
single run with greedy/argmax scoring, n=100-300 per model. **A negative result here is evidence about this construction, not about PageRank-based trust on the real web.**
