#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/eval_qwen2_5_vl.sh [K] [KMIN] [TASKS]
# Example:
# Paper-recommended K_min: K=64 -> 10, K=128 -> 20, K=256 -> 40
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/eval_qwen2_5_vl.sh 64 10 mme

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LMMS_ROOT="${REPO_ROOT}/lmms-eval"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC="${NPROC:-4}"
PORT="${PORT:-29641}"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
K="${1:-64}"
KMIN="${2:?The minimum anchor size K_min must be specified explicitly; see the paper-recommended settings in the script header.}"
TASKS="${3:-mme}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/logs/qwen2_5_vl}"

TASK_TAG="${TASKS//,/_}"
OUT="${OUT_ROOT}/${TASK_TAG}/k${K}_kmin${KMIN}"

export PYTHONPATH="${REPO_ROOT}:${LMMS_ROOT}:${PYTHONPATH:-}"

MODEL_ARGS_LIST=(
  "pretrained=${MODEL}"
  "attn_implementation=sdpa"
  "use_cache=true"
  "prune=true"
  "prune_k_total=${K}"
  "prune_anchor_k=${KMIN}"
)
MODEL_ARGS="$(IFS=,; echo "${MODEL_ARGS_LIST[*]}")"

echo "[AnchorPrune] Qwen2.5-VL-7B | tasks=${TASKS} | K=${K} | KMIN=${KMIN} | output=${OUT}"
cd "${LMMS_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC}" \
  --master_port="${PORT}" \
  -m lmms_eval \
  --model qwen2_5_vl \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --log_samples_suffix "qwen25_anchorprune_k${K}" \
  --output_path "${OUT}" \
  --verbosity INFO
