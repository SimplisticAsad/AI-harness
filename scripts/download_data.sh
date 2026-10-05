#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."; mkdir -p data/raw; cd data/raw
curl -sSL -o gsm8k_test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
curl -sSLO https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip && unzip -qo ARC-V1-Feb2018.zip && rm -rf ARC-V1-Feb2018.zip __MACOSX
G=https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data
for s in train dev test; do curl -sSL -o goemo_$s.tsv $G/$s.tsv; done
curl -sSL -o goemo_emotions.txt $G/emotions.txt; curl -sSL -o goemo_ekman.json $G/ekman_mapping.json; curl -sSL -o goemo_sent.json $G/sentiment_mapping.json
