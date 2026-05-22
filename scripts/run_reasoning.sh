#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${BACKEND:-semantiq}"
TASK="${TASK:-math500}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/reasoning}"
MODEL="${MODEL:-/home/ytm/models/Llama-3.1-8B-Instruct}"
TP_SIZE="${TP_SIZE:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
QUANT_METHOD="${QUANT_METHOD:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
SEMANTIQ_PRIOR_PATH="${SEMANTIQ_PRIOR_PATH:-}"
RUN_NAME="${RUN_NAME:-}${BACKEND}_${QUANT_METHOD}"
if [[ -z "${SEMANTIQ_PRIOR_PATH}" ]]; then
  SEMANTIQ_PRIOR_PATH="${ROOT_DIR}/outputs/priors/_debug_hybrid_k_base_tp4.json"
fi

REASONING_ARGS=(
  --backend "${BACKEND}"
  --dataset "${TASK}"
  --data-dir "${DATA_DIR}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${RUN_NAME}" ]]; then
  REASONING_ARGS=(
    "${REASONING_ARGS[@]}"
    --run-name "${RUN_NAME}"
  )
fi

REASONING_ARGS=(
  "${REASONING_ARGS[@]}"
  --max-samples "${MAX_SAMPLES}"
  --model "${MODEL}"
  --tensor-parallel-size "${TP_SIZE}"
  --block-size "${BLOCK_SIZE}"
  --no-enable-prefix-caching
)

if [[ "${BACKEND}" == "semantiq" ]]; then
  export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
  REASONING_ARGS=(
    "${REASONING_ARGS[@]}"
    --semantiq-fake-quant-enable
    --semantiq-segment-page-size "${BLOCK_SIZE}"
  )
  if [[ "${QUANT_METHOD}" == "1" && -n "${SEMANTIQ_PRIOR_PATH}" ]]; then
    REASONING_ARGS=(
      "${REASONING_ARGS[@]}"
      --semantiq-prior-path "${SEMANTIQ_PRIOR_PATH}"
    )
  fi
  REASONING_ARGS=(
    "${REASONING_ARGS[@]}"
    --semantiq-quant-method "${QUANT_METHOD}"
  )
fi

PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/vllm" \
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m eval.bench.reasoning \
  "${REASONING_ARGS[@]}"
