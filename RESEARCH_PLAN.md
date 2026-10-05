# Research plan — Emotional-Semantic Distress (ESD) harness

*Written before any evaluation results were produced. Environment findings below are measured, not assumed.*

## 1. Question

> Can affective, epistemic, semantic and uncertainty-related characteristics of LLM-generated text be used as a reliable
> signal for detecting incorrect model outputs — and does adding an ESD harness reduce *accepted* errors relative to
> existing uncertainty and judging approaches?

Terminology: we measure **affective / epistemic characteristics of generated text** and **model output
representations**. We make **no claim** that the models experience emotions. Three things are kept apart throughout:
(i) internal computation (token distributions, encoder states), (ii) affect/epistemic stance *expressed in text*,
(iii) human-like emotional experience (never claimed, never measured).

## 2. Environment (measured)

| Item | Finding | Consequence |
|---|---|---|
| Hardware | 4 vCPU, 15 GB RAM, **no GPU** | Only small models; fp32 CPU inference; N samples per item must stay small |
| Hugging Face Hub, Ollama, ModelScope | **Blocked by network policy** (HTTP 403 at proxy) | No Llama/Qwen/Gemma/Phi, no HF pretrained emotion/NLI/embedding models. Not circumvented. |
| PyPI, GCS public buckets, GitHub raw, S3 | Reachable | Open weights obtained from Google's public **T5X Flan-T5 checkpoints** (`gs://t5-data`), converted locally |
| Python | 3.11, torch 2.14 (CPU use), transformers 5.18 | |

**Models actually usable:** Flan-T5 small (77M), base (248M), large (783M) — *one family, three sizes*. The requested
~1B/~4B/~8B models and multiple model **families** are **not testable here**, so H6 can only be tested across *sizes
of one family* (not across families). Flan-T5-xl (2.8B) is feasible on disk but too slow on 4 CPU cores for the sampling budget.

**Known consequence of the model family** (verified, see report): the T5 SentencePiece vocabulary cannot represent newlines,
`{ } < \ ^ ~`, so multi-line Python (HumanEval/MBPP) cannot be generated; the coding task is therefore one-line lambda
expressions with executable tests. Flan-T5 also answers tersely (a few tokens) on several tasks, which leaves little
text for affect/epistemic analysis. This is a **limitation of the regime studied**, and results must not be generalised
to chat-tuned 7B+ models that write long, expressive responses.

## 3. Hypotheses (tested independently; none assumed true)

H1 affect predicts errors · H2 epistemic stance predicts errors · H3 semantic consistency/contradiction helps ·
H4 the combination beats every single family · H5 the harness reduces accepted errors · H6 transfers across models/sizes ·
H7 transfers across tasks. A negative result for any of them is a valid finding.

## 4. Variables

* **Independent (signals)** — affect (A), epistemic (B), semantic (C), token uncertainty (D), candidate disagreement (E), surface linguistic (L).
* **Dependent** — `error = 1` if the greedy answer is incorrect under a *deterministic* grader (label, arithmetic, solver logic, executed tests, constraint checkers). **Never LLM-graded.**
* **Controls / confounds** — answer length (tokens/words/chars), question length, difficulty (dataset-provided or generator hop/step count), task identity (base rate), emotional vocabulary density, model size.

## 5. Tasks and ground truth

| Task | Data | Ground truth |
|---|---|---|
| factual_qa | ARC-Easy/Challenge test (real, 4-option) | dataset answer key |
| math | 50% GSM8K test (real) + 50% synthetic 1–2-step arithmetic | dataset answer / direct computation |
| logic | synthetic syllogisms (yes/no/unknown, 1–3 hops, invented kinds) | solver by construction |
| longform | synthetic multi-premise orderings, explanation required | order by construction **and** no asserted relation contradicting the premises |
| instruction | synthetic IFEval-style constraints (1–3 per prompt) | deterministic constraint checkers |
| code | synthetic one-line lambda specs (difficulty 0–3) | execution against tests in a sandboxed subprocess |

Examples per task: 300 (small, base), 120 (large; prefix of the same ordered list). Splits are by hash of the example id
(60/20/20 train/val/test), identical across models, so no question appears in two splits.

## 6. Signals

* **Affect (contextual, primary):** emotion (Ekman-6 + neutral) and sentiment (pos/neg/ambiguous) classifiers trained by us on GoEmotions
  over frozen encoder states, sentence-level, mean/max pooled; valence = P(pos)−P(neg); label-free Mahalanobis *affective anomaly*.
  Validated on GoEmotions dev/test and on negation probes ("I am not happy"). **Arousal and dominance: not evaluated** (no VAD data reachable).
* **Affect (baseline only):** VADER word-level lexicon.
* **Epistemic:** scope-sensitive cue profile (hedges, boosters, speculation, qualification, assertives, ambiguity, net certainty); lexicon fixed from the linguistics literature before any evaluation.
* **Semantic:** question–answer alignment, adjacent-sentence coherence, first/last drift, heuristic claim contradictions (not an NLI model), novel-word ratio (proxy for unsupported content).
* **Token:** mean/min logprob, entropy (mean, max, spikes), uncertainty at answer tokens, at sentence starts (reasoning transitions) and at digits.
* **Candidates:** with N = 5 samples (T = 0.7) + greedy: answer agreement/entropy/vote fraction, pairwise semantic and lexical agreement, spread of epistemic and affective scores.

## 7. Distress score

`D = Σ_g w_g · A_g`, `A_g = mean_i σ_i z_i`, with equal group weights `w_g = 1/|G|`, orientations `σ_i ∈ {−1,+1}` fixed *a priori*,
`z_i` standardised with TRAIN statistics. This has no fitted weights. Learned alternatives: logistic regression, random forest,
gradient boosting, small MLP, each with Platt/isotonic calibration. Selection of family and hyper-parameters on VALIDATION only.

## 8. Baselines (all evaluated under the same protocol)

B1 base LLM (accept all) · B2 token logprob · B3 token entropy · B4 self-consistency (disagreement; also majority-vote accuracy) ·
B5 LLM-as-judge (self, and independent Flan-T5-large) · B6 semantic disagreement · B7 affect only · B8 epistemic only · B9 semantic only ·
B10 combined ESD. Plus: word-lexicon emotion baseline, length/difficulty/task-only controls, task-prior-only, random scores.

## 9. Metrics and statistics

AUROC, AUPRC, precision/recall/F1 at a validation-chosen threshold, Brier, ECE (Platt and isotonic), reliability curves,
**false acceptance** (P(accept | wrong)), false rejection, accepted-error rate, risk–coverage / AURC, repair success, latency, tokens, model calls.
Inference: cluster bootstrap CIs (1000 resamples), paired bootstrap for differences, paired permutation tests for AUROC differences,
permutation test vs chance, McNemar (accepted-error comparisons), Wilcoxon across (model, task) cells, Holm correction, effect sizes.

## 10. Experiments

1. **Main:** distress vs error probability (Pearson, Spearman, AUROC, AUPRC, calibration, CIs) per model/task/pooled.
2. **Ablation:** A, B, C, D, E, A+B, A+C, B+C, all.
3. **Controls:** length/difficulty/task-only classifiers; ESD **incremental** over controls; within-length-tercile AUROC; partial correlations; shuffled-label placebo.
4. **Generalisation:** leave-one-task-out (H7), train-on-one-size/test-on-another (H6).
5. **Adversarial:** confident-wrong, uncertain-correct, neutral-wrong, expressive-correct, long-correct, short-wrong, negative-language-correct, calm-wrong, contradictory, plus text-manipulation attacks (hedge/booster injection) and **prompt injection** against the judge and ESD.
6. **Verification/repair loop** with retry limit and UNCERTAIN abstention; deterministic tools are used only when they do not require the gold label.
7. **Error taxonomy** and per-error-type detectability.

## 11. Expected outputs

`results/` (raw generations with token traces, features, judge scores, predictions, metrics, `summary.csv`, manifests with seed/model/hardware/timestamp),
`reports/figures/*.png`, `reports/final_report.md` (and PDF if a renderer is available), README with exact reproduction commands, unit + integration tests.

## 12. Honesty rules

Every number in the report is produced by code in this repo and read from `results/`. Anything not run is labelled "Not evaluated".
References are included only if I am confident they exist; any I cannot verify offline are marked **[verify]**.
