"""Assemble reports/final_report.md from saved result CSVs. Every number is read from results/; nothing is typed in.

Hypothesis verdicts use criteria that were fixed BEFORE results existed (see VERDICT RULES in section 13).
A hand-written interpretation (reports/interpretation.md) is spliced in if present and is labelled as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, ".")  # noqa: E702

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ESD = "ESD_core(A+B+C+D+E)"
REFS = """\
Verified against primary sources during this project (web search):
- Kuhn, L., Gal, Y., Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. ICLR 2023. arXiv:2302.09664.
- Kadavath, S. et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Manakul, P., Liusie, A., Gales, M. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. EMNLP 2023. arXiv:2303.08896.

Cited from memory - **[verify]** exact venue/pagination before reuse (offline for these):
- Wang, X. et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023. arXiv:2203.11171. [verify]
- Lin, S., Hilton, J., Evans, O. (2022). Teaching Models to Express Their Uncertainty in Words. TMLR. arXiv:2205.14334. [verify]
- Xiong, M. et al. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. ICLR 2024. arXiv:2306.13063. [verify]
- Farquhar, S. et al. (2024). Detecting hallucinations in large language models using semantic entropy. Nature 630. [verify]
- Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023. arXiv:2306.05685. [verify]
- Ji, Z. et al. (2023). Survey of Hallucination in Natural Language Generation. ACM Computing Surveys. [verify]
- Huang, L. et al. (2023). A Survey on Hallucination in Large Language Models. arXiv:2311.05232. [verify]
- Azaria, A., Mitchell, T. (2023). The Internal State of an LLM Knows When It's Lying. Findings of EMNLP 2023. [verify]
- Turpin, M. et al. (2023). Language Models Don't Always Say What They Think. NeurIPS 2023. [verify]
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML 2017. [verify]
- Naeini, M., Cooper, G., Hauskrecht, M. (2015). Obtaining Well Calibrated Probabilities Using Bayesian Binning. AAAI 2015. [verify]
- Geifman, Y., El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. NeurIPS 2017. [verify]
- Picard, R. (1997). Affective Computing. MIT Press. [verify]
- Ekman, P. (1992). An argument for basic emotions. Cognition & Emotion 6. [verify]
- Russell, J. (1980). A circumplex model of affect. J. Personality and Social Psychology 39. [verify]
- Demszky, D. et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020. [verify]
- Hutto, C., Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM. [verify]
- Hyland, K. (1998). Hedging in Scientific Research Articles. John Benjamins. [verify]
- Chung, H. W. et al. (2022). Scaling Instruction-Finetuned Language Models (Flan-T5). arXiv:2210.11416. [verify]
- Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR 21. [verify]
- Roberts, A. et al. (2022). Scaling Up Models and Data with t5x and seqio. arXiv:2203.17189. [verify]
- Clark, P. et al. (2018). Think you have Solved Question Answering? Try ARC. arXiv:1803.05457. [verify]
- Cobbe, K. et al. (2021). Training Verifiers to Solve Math Word Problems (GSM8K). arXiv:2110.14168. [verify]
- Zhou, J. et al. (2023). Instruction-Following Evaluation for Large Language Models (IFEval). arXiv:2311.07911. [verify]
- Chen, M. et al. (2021). Evaluating Large Language Models Trained on Code (HumanEval). arXiv:2107.03374. [verify]
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian J. Statistics 6. [verify]
- Efron, B., Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman & Hall. [verify]
"""


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def md(df: pd.DataFrame, cols=None, d=3) -> str:
    if df is None or len(df) == 0:
        return "_Not evaluated / no rows._\n"
    df = df[cols] if cols else df
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(f(v, d) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |\n"
    return head + body


def ci(r) -> str:
    return f"{f(r.auroc, 2)} [{f(r.auroc_lo, 2)}, {f(r.auroc_hi, 2)}]"


def load(ev: Path, name: str):
    p = ev / name
    return pd.read_csv(p) if p.exists() else None


def verdicts(Mx, H4, LOTO, CM, R, MC, W=None) -> list[tuple[str, str, str]]:
    out = []
    pooled = Mx[(Mx.scope == "pooled") & (Mx.family == "logreg")]

    def ci_rule(method):
        r = pooled[pooled.method == method]
        n_sig = int((r.auroc_lo > 0.5).sum())
        pts = ", ".join(f"{m}: {ci(x)}" for m, x in zip(r.model, r.itertuples()))
        v = "SUPPORTED" if n_sig >= max(2, int(np.ceil(0.67 * len(r)))) else ("WEAK/INCONCLUSIVE" if (r.auroc > 0.5).any() and n_sig >= 1 else "NOT SUPPORTED")
        return v, pts
    for h, m, desc in (("H1 affective signal", "A_affective", "affect-only AUROC CI lower bound > 0.5 in >= 2/3 of models"),
                       ("H2 epistemic signal", "B_epistemic", "epistemic-only AUROC CI lower bound > 0.5 in >= 2/3 of models"),
                       ("H3 semantic consistency", "C_semantic", "semantic-only (text) AUROC CI lower bound > 0.5 in >= 2/3 of models")):
        v, pts = ci_rule(m)
        extra = ""
        if W is not None:
            w = W[(W.method == m) & (W.family == "logreg")]
            nsw = int((w.lo > 0.5).sum())
            vw = "SUPPORTED" if nsw >= max(2, int(np.ceil(0.67 * len(w)))) else ("WEAK/INCONCLUSIVE" if (w.within_task_auroc > 0.5).any() and nsw >= 1 else "NOT SUPPORTED")
            extra = (f" || **within-task verdict (post-hoc, stricter, treated as primary): {vw}** - " +
                     ", ".join(f"{a}: {b.within_task_auroc:.2f} [{b.lo:.2f}, {b.hi:.2f}]" for a, b in zip(w.model, w.itertuples())))
        out.append((h, v + (" (pooled rule)" if W is not None else ""), f"{desc}. {pts}{extra}"))
    # H3b semantic disagreement across candidates
    r = Mx[(Mx.scope == "pooled") & (Mx.method == "B6_semantic_disagreement")]
    if len(r):
        out.append(("H3b cross-sample semantic disagreement (baseline B6)", "SUPPORTED" if (r.auroc_lo > 0.5).sum() >= 2 else ("WEAK/INCONCLUSIVE" if (r.auroc > 0.5).any() else "NOT SUPPORTED"),
                    "; ".join(f"{m}: {ci(x)}" for m, x in zip(r.model, r.itertuples()))))
    if H4 is not None and len(H4):
        h = H4[H4.scope == "pooled"]
        best_single = h[h.comparator.isin(["A_affective", "B_epistemic", "C_semantic", "D_token", "E_candidate"])]
        n_ok = 0; det = []
        for m, g in best_single.groupby("model"):
            top = g.sort_values("auroc_cmp", ascending=False).iloc[0]
            ok = top.delta_lo > 0
            n_ok += ok
            det.append(f"{m}: best single={top.comparator} ({f(top.auroc_cmp, 2)}); ESD core {f(top.auroc_ref, 2)}; ΔAUROC {f(top.delta_auroc, 3)} [{f(top.delta_lo, 3)}, {f(top.delta_hi, 3)}]")
        out.append(("H4 combination beats every single family", "SUPPORTED" if n_ok >= 2 else ("WEAK/INCONCLUSIVE" if n_ok == 1 else "NOT SUPPORTED"),
                    "paired-bootstrap ΔAUROC vs the BEST single family, CI lower bound > 0 in >= 2 models. " + "; ".join(det)))
    if R is not None and MC is not None and len(MC):
        sub = MC[MC.comparator.isin(["B3_entropy", "B5_judge_independent"])]
        base = R[(R.policy == "ESD_core") & (R.budget == 0.0)]
        r4 = R[(R.policy == "ESD_core") & (R.budget == 0.4)]
        red = [(m, float(base[base.model == m].error_leakage.iloc[0]), float(r4[r4.model == m].error_leakage.iloc[0])) for m in base.model]
        n_sig = int(((sub.p_holm < 0.05) & (sub.esd_leakage < sub.comparator_leakage)).sum())
        better_than_none = sum(b4 < b0 for _, b0, b4 in red)
        v = "SUPPORTED" if (n_sig >= 2 and better_than_none == len(red)) else ("WEAK/INCONCLUSIVE" if better_than_none else "NOT SUPPORTED")
        out.append(("H5 harness reduces accepted errors", v,
                    "at 40% verification budget: wrong answers reaching the user per request, no-harness -> ESD: " + "; ".join(f"{m}: {f(a, 3)} -> {f(b, 3)}" for m, a, b in red) +
                    f". ESD significantly better (Holm McNemar p<0.05) than entropy/judge routing in {n_sig} of {len(sub)} paired tests. (Note: this compares leakage under the same verifier; "
                    "see section 9 for what the verifier alone achieves.)"))
    if CM is not None and len(CM):
        off = CM[CM.train_model != CM.test_model]
        n = int((off.lo > 0.5).sum())
        out.append(("H6 model generalisation (sizes of ONE family)", "SUPPORTED" if n == len(off) else ("WEAK/INCONCLUSIVE" if n > 0 else "NOT SUPPORTED"),
                    f"cross-model transfer AUROC CI lower bound > 0.5 in {n}/{len(off)} off-diagonal cells. Different model *families*: Not evaluated."))
    if LOTO is not None and len(LOTO):
        per_model = LOTO.groupby("model").apply(lambda g: int((g.lo > 0.5).sum()), include_groups=False)
        n_ok = int((per_model >= 4).sum())
        out.append(("H7 task generalisation", "SUPPORTED" if n_ok >= 2 else ("WEAK/INCONCLUSIVE" if per_model.max() >= 2 else "NOT SUPPORTED"),
                    "leave-one-task-out AUROC CI lower bound > 0.5 on >= 4/6 held-out tasks for >= 2 models. Tasks passing per model: " + ", ".join(f"{k}: {v}/6" for k, v in per_model.items())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="results/main")
    ap.add_argument("--out", default="reports/final_report.md")
    a = ap.parse_args()
    root = Path(a.exp); ev = root / "eval"
    Mx = pd.read_csv(ev / "metrics_all.csv"); frame = pd.read_csv(ev / "analysis_frame.csv")
    S = pd.read_csv(root.parent / "summary.csv")
    H4, LOTO, CM = load(ev, "h4_paired_comparisons.csv"), load(ev, "gen_leave_one_task_out.csv"), load(ev, "gen_cross_model.csv")
    R, MC, VQ = load(ev, "repair_policies.csv"), load(ev, "repair_mcnemar.csv"), load(ev, "verifier_quality.csv")
    CI_, PS, ST, PL = (load(ev, n) for n in ("controls_incremental.csv", "controls_partial_spearman.csv", "controls_length_strata.csv", "controls_placebo.csv"))
    ET, OC, VC, WC = (load(ev, n) for n in ("error_type_detectability.csv", "overconfidence.csv", "vs_chance.csv", "wilcoxon_cells.csv"))
    WT = load(ev, "within_task_auroc.csv")
    ADV, ATK = load(ev, "adversarial_natural_cases.csv"), load(ev, "adversarial_text_attacks.csv")
    AV = json.loads(Path("results/affect/affect_validation.json").read_text()) if Path("results/affect/affect_validation.json").exists() else None
    man = json.loads((root / "flan_t5_small" / "manifest_generate.json").read_text()) if (root / "flan_t5_small" / "manifest_generate.json").exists() else {}
    hw = man.get("hardware", {})
    pooled = Mx[Mx.scope == "pooled"]
    L = []
    w = L.append

    w("# Emotional-Semantic Distress (ESD) Harness: Do Affective, Epistemic and Semantic Output Signals Predict LLM Errors?\n")
    w("*Auto-generated by `experiments/make_report.py` from `results/`. Every number below is read from a result file produced by code in this repository. Items not run are marked **Not evaluated**.*\n")
    # ------------------------------------------------------------------ abstract
    ev_rows = []
    for model in pooled.model.unique():
        e = pooled[(pooled.model == model) & (pooled.method == ESD) & (pooled.family == "logreg")].iloc[0]
        t = pooled[(pooled.model == model) & (pooled.method == "B3_token_entropy")].iloc[0]
        j = pooled[(pooled.model == model) & (pooled.method == "B5_llm_judge_self")]
        c = pooled[(pooled.model == model) & (pooled.method == "CTRL_length+difficulty+task") & (pooled.family == "logreg")]
        ev_rows.append(f"{model}: ESD core {ci(e)}, token entropy {ci(t)}" + (f", self-judge {ci(j.iloc[0])}" if len(j) else "") + (f", length/difficulty/task-only control {ci(c.iloc[0])}" if len(c) else ""))
    w("## Abstract\n")
    w("We test whether affective, epistemic, semantic and token-level characteristics of LLM output predict whether the output is wrong, using small open instruction-tuned models "
      "(Flan-T5 small/base/large; no GPU, Hugging Face Hub unreachable), six task types with deterministic ground truth, and a leakage-safe train/validation/test protocol. "
      "Pooled-task test AUROC for error detection (95% cluster-bootstrap CI): " + "; ".join(ev_rows) + ". "
      "See section 13 for pre-specified verdicts on H1-H7; section 15 for limitations that bound what can be concluded (one model family, <=783M parameters, terse outputs).\n")
    # ------------------------------------------------------------------ 1-4
    w("## 1. Introduction\n")
    w("LLMs produce fluent errors. Practical systems need a cheap signal for *when to distrust* an answer. Existing signals include token probabilities, "
      "sampling-consistency, and LLM judges. This work asks whether the *affective and epistemic stance expressed in the text*, together with semantic consistency, adds anything. "
      "**We make no claim that models feel emotions.** We separate (i) internal computation (token distributions, encoder states), (ii) affect/epistemic characteristics *expressed in generated text*, "
      "and (iii) human emotional experience, which is neither claimed nor measured. 'Distress' is a name for a risk score, not a psychological state.\n")
    w("## 2. Problem definition\n")
    w("Given prompt `x` and the model's greedy answer `y`, plus optionally N sampled answers, produce a score `D(x, y, samples)` such that higher D means higher probability that `y` is incorrect under a deterministic grader. "
      "The harness may route high-D answers to verification (independent judge + deterministic tools) and bounded regeneration, abstaining (`UNCERTAIN`) when correctness cannot be established.\n")
    w("## 3. Research questions\n")
    w("RQ1: Which signal families carry information about errors? RQ2: Does their combination beat the best single family and beat existing baselines? RQ3: Is any apparent signal explained by confounds (length, difficulty, task)? "
      "RQ4: Does it transfer across tasks and model sizes? RQ5: Does routing by ESD reduce accepted errors? RQ6: Can it be fooled?\n")
    w("## 4. Hypotheses\n")
    w("H1 affect predicts errors; H2 epistemic stance predicts errors; H3 semantic consistency/contradiction helps; H4 the combination beats every single signal; "
      "H5 the harness reduces accepted errors; H6 transfer across models/sizes; H7 transfer across tasks. None assumed true; negative results are reported as findings.\n")
    w("## 5. Conceptual background\n")
    w("**Hallucination and uncertainty.** LLMs generate plausible but unsupported or wrong content (Ji et al. 2023; Huang et al. 2023). Token-probability confidence can be informative (Kadavath et al. 2022) and "
      "verbalised confidence can be elicited (Lin et al. 2022; Xiong et al. 2024). **Semantic uncertainty:** entropy over meaning-clusters of sampled answers (Kuhn et al. 2023; Farquhar et al. 2024) and sampling-consistency checks "
      "(Wang et al. 2023; Manakul et al. 2023) motivate our candidate-agreement features. **LLM-as-a-judge** (Zheng et al. 2023) is our B5 baseline. **Calibration and selective prediction** "
      "(Guo et al. 2017; Naeini et al. 2015; Geifman & El-Yaniv 2017) motivate ECE/Brier and risk-coverage. **Affective computing** (Picard 1997) and dimensional/categorical affect "
      "(Russell 1980; Ekman 1992; Demszky et al. 2020) motivate our affect features; word-lexicon polarity (Hutto & Gilbert 2014) is the baseline we are told not to rely on. "
      "**Hedging/epistemic modality** (Hyland 1998) motivates the epistemic cue profile. A caution from the literature: stated reasoning need not reflect the computation that produced the answer (Turpin et al. 2023), "
      "while internal states can carry truthfulness information (Azaria & Mitchell 2023). Our token- and candidate-level features probe the former side; expressed-text features probe the latter. "
      "Citation status is given in the References section.\n")
    # ------------------------------------------------------------------ 6/7
    w("## 6. Proposed ESD architecture\n")
    w("```\nPROMPT -> LLM (greedy + N samples, token logprobs)\n            |\n            v\n   ESD ANALYZER  [Affect A | Epistemic B | Semantic C | Token D | Candidates E | (Linguistic L)]\n            |\n            v\n   DISTRESS / RISK SCORE  D  (fixed-weight formula or learned+calibrated)\n            |\n   low risk -+- high risk\n      |             |\n   RETURN     VERIFIER = independent judge + deterministic tools\n                      |\n                PASS -+- FAIL\n                  |        |\n               RETURN   REGENERATE (<= 2) -> VERIFY -> ... -> UNCERTAIN (abstain)\n```\n")
    w("Code layout: `src/harness/{core,models,signals,affect,epistemic,semantic,uncertainty,verification,scoring,routing,evaluation}`; extractors implement a common `SignalExtractor` protocol and never receive ground truth (`AnswerContext` has no label fields; unit-tested).\n")
    w("## 7. Algorithm\n")
    w("**Affect (contextual).** Sentence embeddings `h(s)` from a frozen Flan-T5-base encoder (mean-pooled); two multinomial logistic heads trained by us on GoEmotions: Ekman-6+neutral and positive/negative/ambiguous sentiment. "
      "Features: per-emotion mean/max over sentences, valence `= P(pos) - P(neg)`, emotion entropy, non-neutral mass, and a label-free Mahalanobis *affective anomaly* of the affect vector w.r.t. the TRAIN centroid of the same (model, task). "
      "Arousal and dominance: **Not evaluated** (no VAD-annotated data reachable). The word-level VADER dictionary is a baseline only.\n")
    w("**Epistemic.** Scope-sensitive cue profile per 100 words: hedges, boosters, speculation phrases, qualifiers, assertives, ambiguity markers; `net_certainty = (boost + assertive - uncertainty)/(marked)`. A negator within 3 tokens turns a booster into an uncertainty marker.\n")
    w("**Semantic.** Question-answer cosine, adjacent-sentence coherence, first-last drift, heuristic claim contradictions (lexical overlap + opposite negation or conflicting numbers; *not* an NLI model), novel-word ratio (proxy for unsupported content).\n")
    w("**Token.** From raw next-token distributions of the greedy answer: mean/min logprob, mean/max entropy, entropy spikes (>2 nats), and the same quantities restricted to answer tokens, sentence-initial tokens (reasoning transitions) and digit tokens (factual/numeric claims). EOS is excluded.\n")
    w("**Candidates (N=5, T=0.7).** Answer agreement/vote fraction/entropy, pairwise semantic and lexical agreement, spread of epistemic certainty and valence across candidates, mean sample logprob.\n")
    w("**Distress score.** Fixed-weight: `D = sum_g w_g A_g`, `A_g = (1/|S_g|) sum_{i in S_g} sigma_i z_i`, `w_g = 1/G` (equal), `z_i` standardised with TRAIN median/std, orientations `sigma_i` fixed a priori in `scoring/feature_sets.py` (no fitting). "
      "Learned: logistic regression, random forest, histogram gradient boosting, small MLP; hyper-parameters and family chosen on VALIDATION AUROC; calibration by Platt scaling and isotonic regression fit on VALIDATION. "
      "In pooled scope all features are z-scored *within task* so task identity cannot be read from feature scale.\n")
    # ------------------------------------------------------------------ 8
    w("## 8. Experimental setup\n")
    w(f"**Hardware:** {hw.get('cpu_model', 'n/a')}, {hw.get('cpu_count', 'n/a')} vCPU, {hw.get('ram_gb', 'n/a')} GB RAM, GPU: {hw.get('gpu', 'n/a')}. "
      f"Software: Python {hw.get('python', 'n/a')}, torch/transformers as recorded in `results/main/*/manifest_generate.json`.\n")
    mrows = []
    for m in ("flan_t5_small", "flan_t5_base", "flan_t5_large"):
        p = root / m / "manifest_generate.json"
        if p.exists():
            j = json.loads(p.read_text())
            n = frame[frame.model == m].shape[0]
            mrows.append({"model": m, "params(M)": j["model"]["params_millions"], "quantization": j["model"]["quantization"],
                          "framework": j["model"]["framework"], "greedy T": j["greedy_temperature"], "sample T": j["temperature_sample"],
                          "N samples": j["config"]["generation"]["n_samples"], "seed": j["seed"], "n items": n,
                          "mean s/item": frame[frame.model == m].eval("lat_greedy_s+lat_samples_s").mean()})
    w(md(pd.DataFrame(mrows)))
    w("Model weights: Google's public T5X Flan-T5 checkpoints (`gs://t5-data/pretrained_models/t5x/flan_t5_*`), converted to PyTorch by `src/harness/models/t5x_convert.py` (sanity-checked by translation/arithmetic probes). "
      "Max new tokens per task: " + json.dumps(man.get("config", {}).get("generation", {}).get("max_new_tokens", {})) + ". "
      "The requested ~1B/~4B/~8B models, other model families, and API models: **Not evaluated** (Hub blocked by network policy; CPU-only).\n")
    tasks = frame.groupby(["task", "dataset"]).size().reset_index(name="n (all models)")
    w("**Datasets / ground truth (all deterministic):**\n\n" + md(tasks) + "\nSplits: 60/20/20 by id hash (identical for every model). HumanEval/MBPP: **Not evaluated** - the T5 SentencePiece vocabulary cannot encode newlines or `{ } < \\ ^ ~` "
      "(verified in `tests`/notes), so multi-line Python is not representable; the coding task is one-line lambdas executed against tests in a sandboxed subprocess.\n")
    sp = frame.groupby(["model", "split"]).size().unstack()
    w("Split sizes:\n\n" + md(sp.reset_index(), d=0))
    if AV:
        r = AV["goemotions_validation"]
        w(f"**Affect model validation (GoEmotions, single-label rows):** Ekman-7 macro-F1 dev {f(r['ekman_dev_macro_f1'])}, test {f(r['ekman_test_macro_f1'])} (n={r['ekman_n_test']}); "
          f"sentiment macro-F1 dev {f(r['sent_dev_macro_f1'])}, test {f(r['sent_test_macro_f1'])}. Negation probes (word lexicon vs contextual):\n\n" + md(pd.DataFrame(AV["negation_probes"])))
    base = frame.groupby(["model", "task"]).agg(error_rate=("error", "mean"), mean_answer_tokens=("ctl_n_tokens", "mean"),
                                                 frac_le_3_tokens=("ctl_n_tokens", lambda x: (x <= 3).mean()), n=("error", "size")).reset_index()
    w("**Base error rates and answer length (all splits)** - note how terse some outputs are, which limits the text available to affect/epistemic analysis:\n\n" + md(base))
    w("**Baselines:** B1 base LLM (accept all), B2 token logprob, B3 token entropy, B4 self-consistency disagreement (and majority-vote accuracy), B5 LLM judge (self; independent Flan-T5-large), "
      "B6 semantic disagreement, B7 affect only, B8 epistemic only, B9 semantic only, B10 combined ESD. Extra controls: word-lexicon emotion, hedge density, length/difficulty/task-only classifiers, task prior, random scores.\n")
    # ------------------------------------------------------------------ 9 results
    w("## 9. Results\n")
    w("### 9.1 Most important experiment: distress vs error (pooled tasks, test split)\n")
    rows = []
    for model in pooled.model.unique():
        g = pooled[(pooled.model == model) & (pooled.method == ESD) & (pooled.family == "logreg")].iloc[0]
        rows.append({"model": model, "n_test": int(g.n_test), "error_rate": g.base_error_rate, "pearson": g.pearson, "spearman": g.spearman,
                     "AUROC [95% CI]": ci(g), "AUPRC": g.auprc, "base AUPRC (=error rate)": g.base_error_rate, "ECE (Platt)": g.ece, "ECE (isotonic)": g.ece_isotonic, "Brier": g.brier})
    w(md(pd.DataFrame(rows)))
    w("![distress vs error](figures/02_distress_vs_error_probability.png)\n\n![distribution](figures/01_distress_distribution.png)\n")
    w("### 9.2 Baseline and ablation comparison (pooled; test AUROC with 95% CI; `Δ` vs chance via permutation test in `vs_chance.csv`)\n")
    show = [("B1_base_llm_accept_all", "baseline"), ("random_scores", "control"), ("B2_token_logprob_confidence", "baseline"), ("B3_token_entropy", "baseline"),
            ("B4_self_consistency_disagreement", "baseline"), ("B5_llm_judge_self", "baseline"), ("B5b_llm_judge_independent(flan_t5_large)", "baseline"),
            ("B6_semantic_disagreement", "baseline"), ("lexicon_word_emotion(VADER_negativity)", "baseline"), ("lexicon_hedge_density", "baseline"),
            ("A_lexicon_baseline", None), ("A_affective", "logreg"), ("B_epistemic", "logreg"), ("C_semantic", "logreg"), ("D_token", "logreg"), ("E_candidate", "logreg"),
            ("L_linguistic", "logreg"), ("F_affect+epistemic", "logreg"), ("G_affect+semantic", "logreg"), ("H_epistemic+semantic", "logreg"),
            ("ESD_text_only(A+B+C)", "logreg"), (ESD, "logreg"), (ESD, "fixed_equal_weights"), (ESD, "random_forest"), (ESD, "grad_boosting"), (ESD, "mlp"),
            ("I_all(+linguistic)", "logreg"), ("CTRL_task_prior_only", "control"), ("CTRL_length+difficulty+task", "logreg"), ("ESD_core+CTRL", "logreg")]
    tbl = {}
    for m, fam in show:
        row = {}
        for model in pooled.model.unique():
            g = pooled[(pooled.model == model) & (pooled.method == m)]
            if fam:
                g = g[g.family == fam]
            elif len(g):
                g = g[g.family == "logreg"]
            row[model] = ci(g.iloc[0]) if len(g) else "n/a"
        tbl[f"{m} [{fam or 'logreg'}]"] = row
    w(md(pd.DataFrame(tbl).T.reset_index().rename(columns={"index": "method [learner]"})))
    w("![ablation](figures/09_ablation.png)\n\n![models](figures/08_model_comparison.png)\n\n![roc](figures/03_roc_curves.png)\n\n![pr](figures/04_precision_recall_curves.png)\n")
    if WT is not None:
        w("### 9.2b Within-task (macro) AUROC - the confound-free comparison\n")
        w("**Why this exists.** A scorer that knows only the task and predicts its base error rate reaches the pooled AUROC shown above (`CTRL_task_prior_only`), so pooled AUROC largely measures task identity, and unnormalised baselines (e.g. token entropy) can even fall below 0.5 pooled because token statistics differ by task. "
          "Within-task AUROC (test-size-weighted mean over tasks; stratified bootstrap) removes this. **This metric was adopted after seeing the task-prior result - it is a post-hoc decision, disclosed here; pooled results are still reported.**\n")
        order = ["ESD_core(A+B+C+D+E)", "A_affective", "B_epistemic", "C_semantic", "D_token", "E_candidate", "L_linguistic", "ESD_text_only(A+B+C)", "CTRL_length+difficulty+task", "ESD_core+CTRL",
                 "B2_token_logprob_confidence", "B3_token_entropy", "B4_self_consistency_disagreement", "B5_llm_judge_self", "B5b_llm_judge_independent(flan_t5_large)", "B6_semantic_disagreement", "random_scores"]
        wt = WT[WT.method.isin(order) & WT.family.isin(["logreg", "baseline", "control"])]
        wt = wt.assign(cell=lambda d: d.apply(lambda r: f"{r.within_task_auroc:.2f} [{r.lo:.2f}, {r.hi:.2f}]", axis=1)).pivot_table(index="method", columns="model", values="cell", aggfunc="first").reindex(order).reset_index()
        w(md(wt) + "\n\n![within-task](figures/08c_within_task_auroc.png)\n")
    w("### 9.3 Per-task AUROC of ESD core (logreg) and the strongest simple baselines\n")
    pt = Mx[(Mx.scope != "pooled") & (((Mx.method == ESD) & (Mx.family == "logreg")) | Mx.method.isin(["B3_token_entropy", "B4_self_consistency_disagreement", "B5_llm_judge_self"]))]
    pt = pt.assign(m=pt.method.str.slice(0, 22)).assign(cell=lambda d: d.apply(ci, axis=1)).pivot_table(index=["model", "scope"], columns="m", values="cell", aggfunc="first").reset_index()
    w(md(pt) + "\n![task heat](figures/08b_task_heatmap.png)\n")
    w("### 9.4 Calibration and decision metrics at the validation-chosen threshold (pooled; ESD core logreg)\n")
    dec = []
    for model in pooled.model.unique():
        for m, fam in ((ESD, "logreg"), ("B3_token_entropy", "baseline"), ("B5_llm_judge_self", "baseline"), ("B4_self_consistency_disagreement", "baseline")):
            g = pooled[(pooled.model == model) & (pooled.method == m) & (pooled.family == fam)]
            if len(g):
                g = g.iloc[0]
                dec.append({"model": model, "method": m[:30], "precision": g.precision, "recall": g.recall, "F1": g.f1, "false_acceptance": g.false_acceptance_rate,
                            "false_rejection": g.false_rejection_rate, "accepted_error_rate": g.accepted_error_rate, "coverage": g.coverage, "AURC": g.aurc, "Brier": g.brier, "ECE": g.ece})
    w(md(pd.DataFrame(dec)) + "\n**False acceptance** = P(harness passes | answer wrong). **False rejection** = P(harness rejects | answer right). Threshold = F1-maximiser on VALIDATION. "
      "A flag-everything or flag-nothing rule can look good on one of these alone, so read them with coverage and AURC.\n\n![calibration](figures/05_calibration_curves.png)\n\n![risk-coverage](figures/06_risk_coverage.png)\n\n![confusion](figures/12_confusion_matrices.png)\n")
    w("### 9.5 Verification and repair\n")
    if R is not None:
        if VQ is not None:
            w("Verifier (independent Flan-T5-large judge, threshold chosen on validation, + deterministic tools) applied to every answer:\n\n" + md(VQ))
        r = R[R.policy.isin(["ESD_core", "B3_entropy", "B5_judge_independent", "B4_self_consistency", "random", "oracle_routing(upper bound)"]) & R.budget.isin([0.0, 0.2, 0.4, 0.6, 1.0])]
        w(md(r[["model", "policy", "budget", "coverage", "accepted_error_rate", "error_leakage", "repair_success_rate", "harm_rate", "correct_answers_lost", "verifications_per_item", "regenerations_per_item"]]))
        mv = R[R.policy == "self-consistency majority-vote answer"]
        w("Majority-vote answer selection (self-consistency, no verifier): error rate vs greedy:\n\n" + md(mv[["model", "initial_error_rate", "accepted_error_rate"]]))
        if MC is not None:
            w("Paired exact McNemar tests (ESD routing vs comparator routing, 40% budget, outcome = 'wrong answer reached user'; Holm-adjusted):\n\n" + md(MC))
        w("Regeneration reuses stored i.i.d. temperature samples (`routing/simulate.py`). Ground truth is used only to score outcomes. **Retrieval-based verification: Not evaluated.** "
          "The instruction-following tool is the grader's own checker (an upper-bound tool); code tool sees only 2 visible tests.\n\n![repair](figures/10_repair_success.png)\n\n![latency](figures/11_latency_cost.png)\n")
    else:
        w("_Not evaluated (repair experiment not run)._\n")
    # ------------------------------------------------------------------ 10-12
    w("## 10. Ablation study (A..I)\n")
    ab = Mx[(Mx.scope == "pooled") & (Mx.family == "logreg") & Mx.method.isin(["A_affective", "B_epistemic", "C_semantic", "D_token", "E_candidate", "F_affect+epistemic", "G_affect+semantic", "H_epistemic+semantic", ESD, "I_all(+linguistic)"])]
    ab = ab.assign(AUROC=ab.apply(ci, axis=1)).pivot_table(index="method", columns="model", values="AUROC", aggfunc="first").reset_index()
    w(md(ab))
    if H4 is not None:
        w("Paired comparisons of ESD core against each component (same learner; paired bootstrap ΔAUROC; paired permutation p; Holm-adjusted over all comparisons):\n\n" + md(H4[H4.scope == "pooled"][["model", "comparator", "auroc_ref", "auroc_cmp", "delta_auroc", "delta_lo", "delta_hi", "p_perm", "p_holm"]]))
    if WC is not None:
        w("Wilcoxon signed-rank across (model, task) cells (per-task test AUROC, logreg):\n\n" + md(WC))
    w("### Controls: is 'distress' just length / difficulty / task / emotional vocabulary?\n")
    if CI_ is not None:
        w("ESD+controls vs controls only (length, question length, difficulty, task dummies): incremental AUROC.\n\n" + md(CI_))
    if PS is not None:
        w("Partial Spearman (risk vs error) after removing task, answer length, question length and difficulty on ranks:\n\n" + md(PS))
    if PL is not None:
        w("Label-shuffle placebo (same pipeline trained on permuted TRAIN labels):\n\n" + md(PL))
    if ST is not None:
        w("AUROC of ESD core within (task x answer-length tercile):\n\n" + md(ST.groupby(["model", "length_tercile"]).auroc.agg(["mean", "min", "max"]).reset_index()))
    w("### Generalisation\n")
    if LOTO is not None:
        w("**H7 leave-one-task-out** (train on other tasks' TRAIN+VAL, test on held-out task's TEST):\n\n" + md(LOTO))
    if CM is not None:
        w("**H6 cross-model** (train on one model's TRAIN+VAL, test on another's TEST; same family, different sizes):\n\n" + md(CM) + "\n![gen](figures/13_generalization.png)\n")
    w("## 11. Error analysis\n")
    et = frame[frame.error == 1].groupby(["model", "error_type"]).size().reset_index(name="n")
    w(md(et, d=0) + "\n![errors](figures/07_error_type_distribution.png)\n")
    if ET is not None:
        w("Detectability by error type (AUROC separating correct answers from each error type; pooled; test):\n\n" + md(ET))
    if OC is not None:
        w("Expressed-stance vs correctness (overconfidence / underconfidence in the *text*):\n\n" + md(OC))
    w("Taxonomy note: grader-assigned types are `factual_hallucination` (wrong MC option), `mathematical_error`, `logical_error`, `reasoning_error`/`contradiction` (long-form), `instruction_violation`, `code_failure`, `incomplete_answer`. "
      "'Unsupported claim' is only proxied (novel-word ratio); 'irrelevant answer' and 'uncertainty failure' are not separately labelled - **Not evaluated** as distinct classes.\n")
    w("## 12. Adversarial testing\n")
    if ADV is not None:
        w("Natural adversarial subsets (flagging = top 40% risk; wrong-answer cases want high flag rate, correct-answer cases low):\n\n" + md(ADV[ADV.scorer.isin(["ESD_core (logreg)", "B3 token entropy", "B5 LLM judge (self)"])][["model", "case", "scorer", "n", "subset_error_rate", "flag_rate", "lo", "hi", "reading"]]) + "\n![adv](figures/14_adversarial_cases.png)\n")
    if ATK is not None:
        w("Controlled text attacks (token/candidate features cannot be changed by editing text and are left as generated):\n\n" + md(ATK))
    else:
        w("_Text attacks / prompt injection: Not evaluated._\n")
    w("## 13. Statistical analysis and pre-specified verdicts\n")
    w("Tests used: **percentile cluster bootstrap** (1000 resamples; resampling items) for CIs of AUROC/AUPRC; **paired bootstrap** for ΔAUROC between scorers on the same items; **paired permutation test** (swap the two scores item-wise) for AUROC differences; "
      "**permutation test vs chance** for AUROC>0.5; **exact McNemar** for paired accepted-error outcomes; **Wilcoxon signed-rank** across (model, task) cells; **Holm** step-down correction within each family of tests; effect sizes as ΔAUROC (and Cohen's h available in `stattests.py`). "
      "Class balance matters for AUPRC, so base error rate is shown beside it.\n")
    w("**VERDICT RULES (fixed before results):** SUPPORTED if the stated CI criterion holds in >= 2/3 of models (or all cells where stated); WEAK/INCONCLUSIVE if the point estimate beats chance but the criterion fails; NOT SUPPORTED otherwise.\n")
    vv = verdicts(Mx, H4, LOTO, CM, R, MC, WT)
    w("| hypothesis | verdict | evidence |\n|---|---|---|\n" + "\n".join(f"| {h} | **{v}** | {e} |" for h, v, e in vv) + "\n")
    # ------------------------------------------------------------------ 14-17
    interp = Path("reports/interpretation.md")
    w("## 14. Discussion\n")
    w(interp.read_text() + "\n" if interp.exists() else "_No hand-written interpretation found; see tables above._\n")
    w("## 15. Limitations\n")
    w("- **One model family, three sizes (77M-783M).** No 1B/4B/8B models, no chat-tuned LLMs, no other families (Hub blocked). H6 is tested across sizes only. Conclusions must not be extrapolated to large chat models that write long, hedged, emotive responses.\n"
      "- **Terse outputs.** Many answers are a few tokens; affect/epistemic extractors then see almost nothing. A null result for A/B here is weak evidence about richer text.\n"
      "- **Analysis encoder is not similarity-tuned**; the contradiction detector is heuristic, not NLI; the affect classifier is trained on Reddit comments and applied to short model outputs (domain shift); arousal/dominance not evaluated.\n"
      "- **Modest test sizes** (see `n_test`); per-task CIs are wide; multiple comparisons corrected only within families. Single train/val/test split per model.\n"
      "- **Synthetic tasks** (logic, long-form, instruction, code) are controlled but not natural distributions; two tools (constraints, visible code tests) are close to the grader.\n"
      "- **Regeneration is simulated** from stored samples; latency/cost are CPU wall-clock for this machine and batch-averaged.\n"
      "- No retrieval verification; no API models.\n")
    w("## 16. Conclusion\n")
    w("See the verdict table in section 13 and the hand-written interpretation in section 14. In one sentence: the supported claims are exactly those whose confidence intervals exclude chance in the tables above; everything else is unproven in this regime.\n")
    w("## 17. Future work\n")
    w("Repeat with chat-tuned 3-8B models (GPU or quantised CPU) that produce long free-form text; add an NLI contradiction model and a VAD-annotated affect model; richer epistemic classifiers trained on hedge corpora; "
      "retrieval verification; natural long-form and open-domain QA with human-verified labels; cross-family transfer; multiple random splits with repeated cross-validation.\n")
    w("## Reproduction\n")
    w("See `README.md`. Raw artefacts: `results/main/<model>/{generations,features,judge_*}.jsonl`, `results/main/eval/*.csv`, `results/summary.csv`, manifests with seed/model/hardware/timestamp.\n")
    w("## References\n")
    w(REFS)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(L))
    print("wrote", a.out, len("\n".join(L)), "chars")


if __name__ == "__main__":
    main()
