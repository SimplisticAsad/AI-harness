"""Feature-set definitions for the ablation grid (A..I) and a-priori risk orientations for the fixed-weight score.

Orientations (+1 = larger value means higher error risk) were fixed from first principles BEFORE any evaluation and are
never fitted, so the fixed-weight ESD score has zero free parameters beyond train-set standardisation.
"""
from __future__ import annotations

import pandas as pd

PREFIX = {"A": ["af_"], "A_lex": ["lex_"], "B": ["ep_"], "C": ["sm_"], "D": ["tk_"], "E": ["cd_"], "L": ["li_"]}

FEATURE_SETS: dict[str, list[str]] = {
    "A_affective": ["A"], "A_lexicon_baseline": ["A_lex"], "B_epistemic": ["B"], "C_semantic": ["C"],
    "D_token": ["D"], "E_candidate": ["E"], "L_linguistic": ["L"],
    "F_affect+epistemic": ["A", "B"], "G_affect+semantic": ["A", "C"], "H_epistemic+semantic": ["B", "C"],
    "ESD_core(A+B+C+D+E)": ["A", "B", "C", "D", "E"], "I_all(+linguistic)": ["A", "B", "C", "D", "E", "L"],
    "ESD_text_only(A+B+C)": ["A", "B", "C"],
}

GROUP_OF = {"A": "affect", "B": "epistemic", "C": "semantic", "D": "token", "E": "candidate", "L": "linguistic"}

PRIORS: dict[str, dict[str, float]] = {
    "affect": {"af_negative_affect": +1, "af_valence": -1, "af_fear_mean": +1, "af_sadness_mean": +1,
               "af_anger_mean": +1, "af_surprise_mean": +1, "af_nonneutral": +1, "af_anomaly": +1},
    "epistemic": {"ep_hedge_per100": +1, "ep_speculation_per100": +1, "ep_uncertainty_marker_count": +1,
                  "ep_qualification_per100": +1, "ep_ambiguity_per100": +1, "ep_net_certainty": -1,
                  "ep_booster_per100": -1, "ep_distinct_numbers": +1, "ep_anomaly": +1},
    "semantic": {"sm_qa_similarity": -1, "sm_coherence": -1, "sm_drift_first_last": -1, "sm_self_contradictions": +1,
                 "sm_begin_end_contradiction": +1, "sm_novel_word_ratio": +1, "sm_min_adjacent_sim": -1},
    "token": {"tk_mean_logprob": -1, "tk_min_prob": -1, "tk_mean_entropy": +1, "tk_max_entropy": +1,
              "tk_entropy_spikes": +1, "tk_ans_mean_logprob": -1, "tk_ans_mean_entropy": +1, "tk_ans_min_prob": -1},
    "candidate": {"cd_answer_agreement": -1, "cd_vote_frac": -1, "cd_answer_entropy": +1, "cd_semantic_agreement": -1,
                  "cd_semantic_min": -1, "cd_lexical_agreement": -1, "cd_n_distinct": +1, "cd_epistemic_std": +1,
                  "cd_affect_std": +1, "cd_unparsed_frac": +1},
}


def columns_for(df: pd.DataFrame, keys: list[str]) -> list[str]:
    pre = tuple(p for k in keys for p in PREFIX[k])
    return [c for c in df.columns if c.startswith(pre)]
