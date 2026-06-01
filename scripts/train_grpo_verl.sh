#!/usr/bin/env bash
# D8/D9 — GRPO via verl (volcengine/HybridFlow) on **single RTX 5090 32 GB**.
#
# Run INSIDE the verl Docker container. Pull the latest Blackwell-compatible tag
# from Docker Hub (https://hub.docker.com/r/verlai/verl/tags) — image name will
# be something like:
#     verlai/verl:app-verl0.8-cu128-...   (verify exact suffix at install time)
#
# Mount the project + SFT'd model + parquet data into /workspace inside the container:
#     docker run --gpus all -it --rm \
#         -v $(pwd):/workspace \
#         -v $(pwd)/outputs:/outputs \
#         -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#         verlai/verl:<latest-blackwell>  bash
#     cd /workspace
#     bash scripts/train_grpo_verl.sh dryrun
#
# Modes:
#     dryrun  — 5 prompts × 3 steps, sanity check (M4 acceptance)
#     formal  — 300 prompts × 30 steps (M5 acceptance)
#
# Memory strategy on 32 GB:
#   * LoRA r=16 (not full FT) — saves ~3 GB of optimizer state thrash
#   * rollout.gpu_memory_utilization=0.30  (was 0.4 on A100-40GB)
#   * data.train_batch_size=16             (was 32)
#   * data.max_response_length=640         (was 768)
#   * actor.ppo_micro_batch_size_per_gpu=1 (was 2)
#   * actor + ref + grad + optimizer all CPU-offloaded
#   * Estimated peak: 25-28 GB · headroom: 3-4 GB
#
# OOM fallback (override at call site):
#     ROLLOUT_GPU_MEM=0.25 ROLLOUT_N=2 MAX_RESPONSE_LEN=512 bash scripts/train_grpo_verl.sh dryrun

set -euo pipefail

MODE="${1:-dryrun}"

DATA_DIR="${DATA_DIR:-data}"
MODEL_PATH="${MODEL_PATH:-outputs/sft_merged}"
PROJECT="${PROJECT:-code_grpo_verl}"
REWARD_FN="${REWARD_FN:-/workspace/scripts/code_reward.py}"
REWARD_FN_NAME="${REWARD_FN_NAME:-compute_score}"

# Memory knobs (5090-tuned)
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-640}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_GPU_MEM="${ROLLOUT_GPU_MEM:-0.30}"
KL_COEF="${KL_COEF:-0.04}"

# LoRA knobs (key win on 32 GB — eliminates full-FT optimizer offload churn)
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

case "$MODE" in
    dryrun)
        TRAIN_FILE="${DATA_DIR}/code_train_dryrun.parquet"
        VAL_FILE="${DATA_DIR}/code_val_dryrun.parquet"
        TRAIN_BS=4
        MINI_BS=2
        TOTAL_EPOCHS=1
        SAVE_FREQ=99
        TEST_FREQ=99
        EXPERIMENT="grpo-dryrun"
        ;;
    formal)
        TRAIN_FILE="${DATA_DIR}/code_train.parquet"
        VAL_FILE="${DATA_DIR}/code_val.parquet"
        TRAIN_BS=16       # was 32 on A100
        MINI_BS=4         # was 8
        TOTAL_EPOCHS=1
        SAVE_FREQ=10
        TEST_FREQ=5
        EXPERIMENT="grpo-formal"
        ;;
    *)
        echo "usage: $0 [dryrun|formal]" >&2
        exit 2
        ;;
esac

if [[ ! -f "$TRAIN_FILE" ]]; then
    echo "ERROR: train file not found: $TRAIN_FILE" >&2
    echo "Did you run scripts/convert_to_verl_parquet.py (D6)?" >&2
    exit 2
fi
if [[ ! -d "$MODEL_PATH" && ! "$MODEL_PATH" =~ ^[A-Za-z0-9_.-]+/ ]]; then
    echo "ERROR: MODEL_PATH not found: $MODEL_PATH" >&2
    echo "Did you run scripts/train_sft.py with --save-merged (D3)?" >&2
    exit 2
fi

# Allocator config — fights fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== verl GRPO :: $MODE (5090-tuned, LoRA r=$LORA_RANK) ==="
echo "  train     : $TRAIN_FILE"
echo "  val       : $VAL_FILE"
echo "  model     : $MODEL_PATH"
echo "  exp       : $EXPERIMENT"
echo "  bs/mini   : $TRAIN_BS / $MINI_BS · micro=1"
echo "  memory    : max_resp=$MAX_RESPONSE_LEN rollout_n=$ROLLOUT_N gpu_mem=$ROLLOUT_GPU_MEM kl=$KL_COEF"
echo "  alloc     : $PYTORCH_CUDA_ALLOC_CONF"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size=$TRAIN_BS \
  data.max_prompt_length=512 \
  data.max_response_length=$MAX_RESPONSE_LEN \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BS \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.lora_rank=$LORA_RANK \
  actor_rollout_ref.actor.lora_alpha=$LORA_ALPHA \
  actor_rollout_ref.actor.target_modules=all-linear \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  custom_reward_function.path="$REWARD_FN" \
  custom_reward_function.name="$REWARD_FN_NAME" \
  trainer.logger=['console','wandb'] \
  trainer.project_name="$PROJECT" \
  trainer.experiment_name="$EXPERIMENT" \
  trainer.total_epochs=$TOTAL_EPOCHS \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=$TEST_FREQ \
  trainer.n_gpus_per_node=1 \
  ray_kwargs.ray_init.num_cpus=8
