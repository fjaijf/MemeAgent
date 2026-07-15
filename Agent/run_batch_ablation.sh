#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="/data/ggbond/test_annotation/FHM_test_seen_batch.jsonl"
COMMON_ARGS=(
  --dataset "$DATASET"
  --main-concurrency 10
  --controller-concurrency 10
  --max-rounds 3
  --final-concurrency 10
)

echo "[1/2] Running main model with separate controller model"
python batch_agent_evaluator.py "${COMMON_ARGS[@]}"

echo "[2/2] Running main model as controller ablation"
python batch_agent_evaluator_main_as_controller.py "${COMMON_ARGS[@]}"

echo "Both batch evaluations completed"
