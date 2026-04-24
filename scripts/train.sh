#!/usr/bin/env bash
# Train a multi-response reward model on Molmo2-4B with LoRA.
#
# Edit the variables below (or override via environment) to match your setup.
# Runs single-node multi-GPU via torchrun. For multi-node, set NODE_RANK/MASTER_ADDR
# externally and launch this script on each node.
#
# Expected input data format (JSONL), one sample per line:
#   {"prompt": "...",
#    "responses": ["A", "B", "C", "D"],
#    "rankings": [3, 1, 2, 4],    # 1 = best, higher = worse; ties allowed
#    "image": "path/to/img.jpg",  # optional
#    "video": "path/to/vid.mp4"}  # optional

set -euo pipefail

# -----------------------------------------------------------------------------
# User configuration — edit or override via env vars
# -----------------------------------------------------------------------------
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"allenai/Molmo2-4B"}
TRAIN_DATA=${TRAIN_DATA:-"path/to/your/train.jsonl"}
IMAGE_BASE_DIR=${IMAGE_BASE_DIR:-"path/to/images"}   # root that relative image paths resolve under; set "" if none
VIDEO_BASE_DIR=${VIDEO_BASE_DIR:-""}                  # set "" if no videos
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/mr2rm"}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# Effective batch size = BATCH_SIZE * NPROC_PER_NODE * NUM_NODES * GRADIENT_ACCUMULATION_STEPS
# Default: 1 * 8 * 1 * 8 = 64
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}

LEARNING_RATE=${LEARNING_RATE:-1e-4}
NUM_EPOCHS=${NUM_EPOCHS:-3}
MAX_LENGTH=${MAX_LENGTH:-24576}
MAX_VISION_TOKENS=${MAX_VISION_TOKENS:-16384}
WARMUP_RATIO=${WARMUP_RATIO:-0}

LORA_R=${LORA_R:-128}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}

VALUE_HEAD_TYPE=${VALUE_HEAD_TYPE:-"mlp"}              # mlp | linear
VALUE_HEAD_HIDDEN_DIM=${VALUE_HEAD_HIDDEN_DIM:-1024}
VALUE_HEAD_ACTIVATION=${VALUE_HEAD_ACTIVATION:-"silu"} # silu | relu | gelu | selu | tanh
RESP_REPR_MODE=${RESP_REPR_MODE:-"last"}               # last | first | first_last_concat | first_last_add | first_last_sub | response_mean

SAVE_STEPS=${SAVE_STEPS:-500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

USE_WANDB=${USE_WANDB:-"false"}                         # "true" to enable
WANDB_PROJECT=${WANDB_PROJECT:-"mr2rm"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-"molmo2-4b-mr2rm"}

DTYPE=${DTYPE:-"bfloat16"}                              # bfloat16 | float16 | float32

# -----------------------------------------------------------------------------
# Derived / guards
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/mr2rm/train.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "Cannot find mr2rm/train.py at ${TRAIN_SCRIPT}" >&2
    exit 1
fi

GLOBAL_BATCH=$(( BATCH_SIZE * NPROC_PER_NODE * NUM_NODES * GRADIENT_ACCUMULATION_STEPS ))
echo "Effective global batch size: ${GLOBAL_BATCH}"

mkdir -p "${OUTPUT_DIR}"

WANDB_FLAGS=()
if [ "${USE_WANDB}" = "true" ]; then
    WANDB_FLAGS=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_run_name "${WANDB_RUN_NAME}")
fi

IMAGE_FLAGS=()
[ -n "${IMAGE_BASE_DIR}" ] && IMAGE_FLAGS=(--image_base_dir "${IMAGE_BASE_DIR}")
VIDEO_FLAGS=()
[ -n "${VIDEO_BASE_DIR}" ] && VIDEO_FLAGS=(--video_base_dir "${VIDEO_BASE_DIR}")

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes="${NUM_NODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
        --base_model_path "${BASE_MODEL_PATH}" \
        --train_data_path "${TRAIN_DATA}" \
        "${IMAGE_FLAGS[@]}" \
        "${VIDEO_FLAGS[@]}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size "${BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --learning_rate "${LEARNING_RATE}" \
        --num_epochs "${NUM_EPOCHS}" \
        --max_length "${MAX_LENGTH}" \
        --max_vision_tokens "${MAX_VISION_TOKENS}" \
        --warmup_ratio "${WARMUP_RATIO}" \
        --dtype "${DTYPE}" \
        --lora_r "${LORA_R}" \
        --lora_alpha "${LORA_ALPHA}" \
        --lora_dropout "${LORA_DROPOUT}" \
        --lora_target_modules att_proj attn_out ff_proj ff_out \
        --value_head_type "${VALUE_HEAD_TYPE}" \
        --value_head_hidden_dim "${VALUE_HEAD_HIDDEN_DIM}" \
        --value_head_activation "${VALUE_HEAD_ACTIVATION}" \
        --resp_repr_mode "${RESP_REPR_MODE}" \
        --save_steps "${SAVE_STEPS}" \
        --save_total_limit "${SAVE_TOTAL_LIMIT}" \
        --shuffle \
        --gradient_checkpointing \
        "${WANDB_FLAGS[@]}"
