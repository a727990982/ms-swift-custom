#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
PATCHED_VLLM_ENCODE_NUM_WORKERS="${PATCHED_VLLM_ENCODE_NUM_WORKERS:-8}"
PATCHED_VLLM_INFER_PREFETCH_SIZE="${PATCHED_VLLM_INFER_PREFETCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
WRITE_BATCH_SIZE="${WRITE_BATCH_SIZE:-32}"
SAMPLE_COUNT="${SAMPLE_COUNT:-32}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-16}"
VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-{\"image\":5,\"video\":2}}"

WORK_DIR="${WORK_DIR:-${REPO_ROOT}/tmp/qwen3_vl_vllm_patch}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/qwen3_vl_vllm_patch}"
DATASET_PATH="${DATASET_PATH:-${WORK_DIR}/qwen3_vl_eval.jsonl}"
RESULT_PATH="${RESULT_PATH:-${OUTPUT_DIR}/infer_result.jsonl}"
IMAGE_URL="${IMAGE_URL:-https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png}"
LOCAL_IMAGE_PATH="${LOCAL_IMAGE_PATH:-${WORK_DIR}/cat.png}"

mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"

IMAGE_SOURCE="${IMAGE_URL}"
if [[ ! -f "${LOCAL_IMAGE_PATH}" ]]; then
    if command -v curl >/dev/null 2>&1; then
        curl -L "${IMAGE_URL}" -o "${LOCAL_IMAGE_PATH}"
        IMAGE_SOURCE="${LOCAL_IMAGE_PATH}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${LOCAL_IMAGE_PATH}" "${IMAGE_URL}"
        IMAGE_SOURCE="${LOCAL_IMAGE_PATH}"
    fi
else
    IMAGE_SOURCE="${LOCAL_IMAGE_PATH}"
fi

: > "${DATASET_PATH}"
for ((i = 0; i < SAMPLE_COUNT; ++i)); do
    printf '{"messages":[{"role":"user","content":"<image>Describe this image in one sentence."}],"images":["%s"]}\n' \
        "${IMAGE_SOURCE}" >> "${DATASET_PATH}"
done

echo "Repo root: ${REPO_ROOT}"
echo "Model: ${MODEL}"
echo "Dataset: ${DATASET_PATH}"
echo "Result path: ${RESULT_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "VLLM_TP: ${VLLM_TP}"
echo "SAMPLE_COUNT: ${SAMPLE_COUNT}"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES
export IMAGE_MAX_TOKEN_NUM
export VIDEO_MAX_TOKEN_NUM
export FPS_MAX_FRAMES

python3 scripts/run_vllm_infer_patched.py \
    --model "${MODEL}" \
    --infer_backend vllm \
    --val_dataset "${DATASET_PATH}" \
    --result_path "${RESULT_PATH}" \
    --write_batch_size "${WRITE_BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_tensor_parallel_size "${VLLM_TP}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_limit_mm_per_prompt "${VLLM_LIMIT_MM_PER_PROMPT}" \
    --patched-vllm-encode-num-workers "${PATCHED_VLLM_ENCODE_NUM_WORKERS}" \
    --patched-vllm-infer-prefetch-size "${PATCHED_VLLM_INFER_PREFETCH_SIZE}" \
    "$@"
