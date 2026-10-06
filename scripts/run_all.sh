#!/usr/bin/env bash
# Full pipeline for configs/main.yaml. Every stage is resumable.
set -e
cd "$(dirname "$0")/.."
C=configs/main.yaml
bash scripts/run_generation.sh
python experiments/fit_affect.py
for m in flan_t5_small flan_t5_base flan_t5_large; do
  python experiments/run_judge.py --config $C --target $m --judge flan_t5_large
  python experiments/run_judge.py --config $C --target $m --judge $m
  python experiments/run_features.py --config $C --model $m
done
python experiments/run_eval.py --config $C
python experiments/run_within_task.py
python experiments/run_repair.py --config $C
python experiments/run_adversarial.py --config $C
python experiments/make_figures.py
python experiments/make_report.py
