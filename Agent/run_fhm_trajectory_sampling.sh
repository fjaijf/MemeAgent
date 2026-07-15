#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/ggbond/MemeAgent/Agent"
DATASET="${DATASET:-/data/ggbond/cache_10.160.8.205/FHM_data/train_trajectory.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-/data/ggbond/cache_10.160.8.205/FHM_data}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/fhm_train_trajectories}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
OFFSET="${OFFSET:-0}"
LIMIT="${LIMIT:-0}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.8}"

mkdir -p "${RUN_ROOT}"

for sample_index in $(seq 1 "${NUM_SAMPLES}"); do
  output_dir=$(printf "%s/sample_%02d_offset_%06d_limit_%06d" \
    "${RUN_ROOT}" "${sample_index}" "${OFFSET}" "${LIMIT}")

  python "${ROOT}/batch_agent_evaluator_main_as_controller.py" \
    --dataset "${DATASET}" \
    --image-root "${IMAGE_ROOT}" \
    --offset "${OFFSET}" \
    --limit "${LIMIT}" \
    --max-rounds "${MAX_ROUNDS}" \
    --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
    --save-prompts \
    --output-dir "${output_dir}"
done
