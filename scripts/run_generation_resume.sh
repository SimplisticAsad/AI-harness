#!/usr/bin/env bash
# Resume helper used in this study: base now; large after the already-running small job (pid $1) exits.
cd "$(dirname "$0")/.."
python3 experiments/run_generate.py --config configs/main.yaml --model flan_t5_base >> results/main/logs/gen_base.log 2>&1 &
B=$!
while kill -0 "$1" 2>/dev/null; do sleep 20; done
python3 experiments/run_generate.py --config configs/main.yaml --model flan_t5_large >> results/main/logs/gen_large.log 2>&1 &
L=$!
wait $B $L
