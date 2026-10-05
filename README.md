# ESD harness — Emotional-Semantic Distress for LLM error detection

Research code that tests whether **affective, epistemic, semantic, token-level and candidate-agreement signals** of an LLM's
output predict whether the output is wrong — and whether routing by that signal reduces accepted errors.
It makes **no claim that models feel emotions**; "affect" means measurements of expressed text. Negative results are first-class:
the report states exactly what did and did not work. See `RESEARCH_PLAN.md` (written first) and `reports/final_report.md`.

> **Scope of this repository's results.** Built in a CPU-only sandbox (4 vCPU, 15 GB) where Hugging Face is blocked. Only
> Flan-T5 small/base/large (open weights from Google's public T5X GCS bucket) were evaluated — one family, ≤783M parameters.
> Larger models and other families are *Not evaluated*; the code supports them via the `LanguageModel` protocol.

## Layout
```
src/harness/  core (types, config, io, hardware, pipeline) · models (T5X->HF converter, Flan-T5 backend, mock)
              signals · affect · epistemic · semantic · uncertainty · scoring · evaluation · verification · routing
benchmarks/   task generators + deterministic graders (ARC, GSM8K, synthetic logic/ordering/instruction/code)
experiments/  run_generate · run_judge · fit_affect · run_features · run_eval · run_adversarial · run_repair · make_figures · make_report
configs/      main.yaml (reported run) · debug.yaml
tests/        unit + integration tests (24)
results/      raw generations (token traces), features, judge scores, predictions, metrics, summary.csv, manifests
reports/      final_report.md, figures/
```

## Reproduce (exact commands)
```bash
pip install torch numpy scipy scikit-learn pandas matplotlib transformers sentencepiece pyyaml pydantic vaderSentiment psutil pytest

# 1. open weights (Google T5X checkpoints) + vocab, then convert to PyTorch
mkdir -p .cache/t5x && cd .cache/t5x
for m in flan_t5_small flan_t5_base flan_t5_large; do gsutil -m cp -r gs://t5-data/pretrained_models/t5x/$m .; done
cd .. && gsutil cp gs://t5-data/vocabs/cc_all.32000/sentencepiece.model spiece.model && cd ..
python - <<'EOF'
import sys; sys.path.insert(0,'src')
from pathlib import Path
from harness.models.t5x_convert import convert
for n in ['flan_t5_small','flan_t5_base','flan_t5_large']:
    convert(n, Path('.cache/t5x'), Path('.cache/spiece.model'), Path(f'.cache/hf/{n}'))
EOF
# (if you saved config.json with tie_word_embeddings=true, set it to false in each .cache/hf/*/config.json)

# 2. datasets (ARC, GSM8K, GoEmotions) -> data/raw
bash scripts/download_data.sh

# 3. tests
python -m pytest -q

# 4. pipeline (resumable)
bash scripts/run_all.sh          # generation -> affect fit -> judge -> features -> eval -> repair -> adversarial -> figures -> report
```
Individual stages: `python experiments/<script>.py --config configs/main.yaml [--model flan_t5_small]`.
Seeds, model versions, hardware and timestamps are written to `results/**/manifest_*.json`.

## Leakage controls
* Splits are by SHA-256 of the example id (60/20/20), identical across models; unit-tested.
* Feature extractors receive `AnswerContext` which has **no label fields**; ground truth comes only from deterministic graders.
* Scorers are fit on TRAIN; family/hyper-parameters/threshold/calibration are chosen on VALIDATION; TEST is only scored.
* Affect model is trained on GoEmotions only; its hyper-parameters are tuned on GoEmotions dev, never on benchmark data.
* The verification loop never sees gold labels (documented exceptions: constraint checker = grader; code tool sees 2 of the test cases).
