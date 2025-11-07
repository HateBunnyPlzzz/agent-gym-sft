#!/bin/bash

# BabyAI Training Script for RunPod A40
# Optimized for storage constraints and memory efficiency

set -e

echo "🚀 BabyAI Training on RunPod A40"
echo "================================"

# Configuration
TASK_NAME="babyai"
EXP_NAME="babyai_2xa40_run"
ENV_SERVER_URL="http://127.0.0.1:36005"
DATASET_PATH="/root/.cache/huggingface/hub/datasets--AgentGym--AgentGym-RL-Data-ID/snapshots/99d0b7923bf126d9c6cdea5f362d2d41ccb5d493/train/babyai_train.json"

# Training parameters (optimized for 2x A40 96GB VRAM total)
KL_COEF=0.001
POLICY_LEARNING_RATE=2e-6          # Slightly higher LR for faster convergence with more compute
ROLLOUT_SAMPLE_NUM=8              # Increased rollouts per sample
TRAIN_BATCH_SIZE=16               # Increased batch size for 2x GPUs
PPO_MINI_BATCH_SIZE=8             # Increased for better gradient estimation
PPO_MICRO_BATCH_SIZE_PER_GPU=2    # Better GPU utilization
PPO_INNER_EPOCHS=1
TOTAL_EPOCHS=8                    # More epochs with better hardware

# Model configuration
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"

# Output configuration
MODEL_SAVE_DIR="saves/babyai_2xa40"
mkdir -p "$MODEL_SAVE_DIR"

echo "Configuration:"
echo "- Task: $TASK_NAME"
echo "- Model: $MODEL_PATH"
echo "- GPUs: 2x A40 (96GB VRAM total)"
echo "- Batch Size: $TRAIN_BATCH_SIZE"
echo "- Learning Rate: $POLICY_LEARNING_RATE"
echo "- Rollout Samples: $ROLLOUT_SAMPLE_NUM"
echo "- Epochs: $TOTAL_EPOCHS"
echo "- Environment Server: $ENV_SERVER_URL"
echo ""

# Prerequisites check
echo "🔍 Checking prerequisites..."

# Check environment server
if ! curl -s "$ENV_SERVER_URL" > /dev/null 2>&1; then
    echo "❌ Environment server not running at $ENV_SERVER_URL"
    echo "Start it with:"
    echo "cd /root/AgentGym-RL/AgentGym/agentenv-babyai"
    echo "babyai --host 0.0.0.0 --port 36005"
    exit 1
fi
echo "✅ Environment server running"

# Check dataset
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Dataset not found: $DATASET_PATH"
    echo "Download with: huggingface-cli download AgentGym/AgentGym-RL-Data-ID --repo-type dataset"
    exit 1
fi
echo "✅ Dataset found"

# Check disk space
AVAILABLE_SPACE=$(df /root | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt 5242880 ]; then
    echo "⚠️  Low disk space: $(($AVAILABLE_SPACE/1024/1024))GB available"
    echo "Cleanup with: pip cache purge"
fi

echo ""
echo "🎯 Starting training..."

# Environment variables
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

# Disable Ray warnings
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Training command
HYDRA_FULL_ERROR=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=disabled \
python3 -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=20 \
    data.train_file="$DATASET_PATH" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.agentgym.task_name=$TASK_NAME \
    actor_rollout_ref.agentgym.env_addr=$ENV_SERVER_URL \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$ROLLOUT_SAMPLE_NUM \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_tokens=400 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.actor.ppo_epochs=$PPO_INNER_EPOCHS \
    actor_rollout_ref.actor.optim.lr=$POLICY_LEARNING_RATE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.rollout_log_dir="$MODEL_SAVE_DIR/logs" \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    trainer.default_local_dir="$MODEL_SAVE_DIR" \
    trainer.project_name=agentgym-rl-babyai \
    trainer.experiment_name=$EXP_NAME \
    trainer.save_freq=50 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.n_gpus_per_node=2

TRAINING_STATUS=$?

echo ""
if [ $TRAINING_STATUS -eq 0 ]; then
    echo "🎉 Training completed!"
    echo "📂 Results: $MODEL_SAVE_DIR"
else
    echo "❌ Training failed (status: $TRAINING_STATUS)"
    echo "📊 Logs: $MODEL_SAVE_DIR/logs"
fi

exit $TRAINING_STATUS