#!/usr/bin/env bash
# Stage 1 for all models. small + base in parallel (2 threads each); large starts when small finishes.
# Resumable: re-running skips examples already present in results/main/<model>/generations.jsonl.
cd "$(dirname "$0")/.."
mkdir -p results/main/logs
python3 experiments/run_generate.py --config configs/main.yaml --model flan_t5_small >> results/main/logs/gen_small.log 2>&1 &
S=$!
python3 experiments/run_generate.py --config configs/main.yaml --model flan_t5_base >> results/main/logs/gen_base.log 2>&1 &
B=$!
wait $S
python3 experiments/run_generate.py --config configs/main.yaml --model flan_t5_large >> results/main/logs/gen_large.log 2>&1 &
L=$!
wait $B $L
