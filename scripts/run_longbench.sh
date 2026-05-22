#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${BACKEND:-semantiq}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
QUANT_METHOD="${QUANT_METHOD:-0}"
RUN_NAME="${RUN_NAME:-semantiq_random}"

LONG_BENCH_ARGS=(
  --dataset qasper
  --data-dir /home/ytm/datasets/LongBench/data
  --output-dir "${ROOT_DIR}/outputs/longbench"
)

if [[ -n "${RUN_NAME}" ]]; then
  LONG_BENCH_ARGS=(
    "${LONG_BENCH_ARGS[@]}"
    --run-name "${RUN_NAME}"
  )
fi

LONG_BENCH_ARGS=(
  "${LONG_BENCH_ARGS[@]}"
  --model /home/ytm/models/Llama-3.1-8B-Instruct
  --tensor-parallel-size 4
  --block-size "${BLOCK_SIZE}"
)

case "${BACKEND}" in
  vllm)
    LONG_BENCH_ARGS=(
      --no-enable-prefix-caching
      "${LONG_BENCH_ARGS[@]}"
    )
    ;;
  semantiq)
    LONG_BENCH_ARGS=(
      --backend semantiq
      --semantiq-fake-quant-enable
      --semantiq-quant-method "${QUANT_METHOD}"
      --semantiq-segment-page-size "${BLOCK_SIZE}"
      "${LONG_BENCH_ARGS[@]}"
    )
    ;;
  *)
    echo "Unsupported BACKEND: ${BACKEND}. Expected 'vllm' or 'semantiq'." >&2
    exit 1
    ;;
esac

PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/vllm" \
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m eval.bench.longbench \
  "${LONG_BENCH_ARGS[@]}"
