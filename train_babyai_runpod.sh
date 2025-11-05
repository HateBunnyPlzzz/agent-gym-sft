#!/bin/bash

# BabyAI Training Script for RunPod A40 GPU
# Adapted from AgentGym-RL/examples/train/AgentGym-RL/babyai_train.sh
# Uses UV environment and paths configured for RunPod setup
# Optimized for Qwen2.5-7B-Instruct on A40 (24GB VRAM)

set -e  # Exit on any error

# Environment variables
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

# Task configuration
task_name="babyai"
exp_name="babyai_runpod_qwen7b"

# Paths (configured for RunPod setup)
WORK_DIR="/root/agent-gym-sft"
AGENTGYM_RL_DIR="$WORK_DIR/AgentGym-RL/AgentGym-RL"
AGENTENV_DIR="$WORK_DIR/AgentGym-RL/AgentGym/agentenv"

# Model configuration - Qwen2.5-7B-Instruct (original size from example)
pure_agent_model_name="Qwen2.5-7B-Instruct"
agent_model_path="models/${pure_agent_model_name}"

# Training hyperparameters optimized for A40 (24GB VRAM) with 7B model
kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num=4          # Reduced from 8 to fit 7B model in memory
train_batch_size=8            # Reduced from 16 for 7B model
ppo_mini_batch_size=4         # Reduced from 8 for memory efficiency
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=1

total_epoches=10              # Same as original

# Model and logging configuration
model_save_dir="$WORK_DIR/saves"
mkdir -p "$model_save_dir"
model_save_path="$model_save_dir/$exp_name"
mkdir -p "$model_save_path"

# Environment server configuration
env_server_url="http://127.0.0.1:36005"

echo "🚀 Starting BabyAI Training on RunPod A40"
echo "=========================================="
echo "Task: $task_name"
echo "Model: $pure_agent_model_name"
echo "Batch Size: $train_batch_size"
echo "Learning Rate: $policy_learning_rate"
echo "Epochs: $total_epoches"
echo "Environment Server: $env_server_url"
echo ""

# Change to AgentGym-RL directory
cd "$AGENTGYM_RL_DIR"

# Activate the UV RL training environment
echo "🔧 Activating RL training environment..."
source "$WORK_DIR/python=3.10/bin/activate"

# Verify VERL installation and GPU memory
echo "🔍 Verifying setup..."
python -c "
import verl
print(f'✅ VERL version: {verl.__version__}')
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'✅ GPU Memory: {gpu_memory:.1f} GB')
    if gpu_memory < 20:
        print('⚠️  Warning: GPU memory might be insufficient for 7B model')
"

echo ""
echo "🤖 Starting BabyAI environment server..."

# Start BabyAI environment server in background
cd "$AGENTENV_DIR"
source "$WORK_DIR/agentenvs/bin/activate"

# Start the environment server
cd /root/agent-gym-sft/AgentGym-RL/AgentGym/agentenv-babyai
python -m agentenv_babyai.launch --host 0.0.0.0 --port 36005 > "$model_save_path/env_server.log" 2>&1 &
ENV_SERVER_PID=$!

# Give the environment server time to start
echo "⏳ Waiting for environment server to start..."
sleep 10

# Check if environment server is running
if ! kill -0 $ENV_SERVER_PID 2>/dev/null; then
    echo "❌ Environment server failed to start. Check logs: $model_save_path/env_server.log"
    exit 1
fi

echo "✅ Environment server started (PID: $ENV_SERVER_PID)"

# Switch back to RL training environment
cd "$AGENTGYM_RL_DIR"
source "$WORK_DIR/python=3.10/bin/activate"

echo ""
echo "📥 Checking/Downloading Qwen2.5-7B-Instruct model..."

# Download/check model
if [ ! -d "$agent_model_path" ]; then
    echo "📥 Downloading Qwen2.5-7B-Instruct model..."
    mkdir -p models
    cd models

    # Use huggingface-cli to download the model
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir "$pure_agent_model_name"
    else
        echo "Installing huggingface-cli..."
        pip install -U "huggingface_hub[cli]"
        huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir "$pure_agent_model_name"
    fi

    cd ..

    if [ ! -d "$agent_model_path" ]; then
        echo "❌ Failed to download model. Using HuggingFace identifier instead."
        agent_model_path="Qwen/Qwen2.5-7B-Instruct"
    fi
else
    echo "✅ Model found at $agent_model_path"
fi

echo ""
echo "🎯 Starting GRPO training with 7B model..."

# GRPO Training command (same parameters as original, optimized for A40)
HYDRA_FULL_ERROR=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
WANDB_MODE=disabled \
python3 -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=20 \
    data.train_file=AgentItemId/${task_name}_train.json \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_tokens=200 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=agentgym-rl-babyai \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=25 \
    trainer.total_epochs=${total_epoches}

TRAINING_STATUS=$?

# Cleanup
echo ""
echo "🧹 Cleaning up..."

# Kill environment server
if kill -0 $ENV_SERVER_PID 2>/dev/null; then
    echo "🛑 Stopping environment server..."
    kill $ENV_SERVER_PID
    sleep 2
    # Force kill if still running
    if kill -0 $ENV_SERVER_PID 2>/dev/null; then
        kill -9 $ENV_SERVER_PID
    fi
fi

# Check training results
if [ $TRAINING_STATUS -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📂 Results saved to: $model_save_path"
    echo "📊 Check logs: $model_save_path/executer_logs"
    echo "🔍 Environment server logs: $model_save_path/env_server.log"
else
    echo ""
    echo "❌ Training failed with status: $TRAINING_STATUS"
    echo "📋 Check logs for debugging:"
    echo "   - Training logs: $model_save_path/executer_logs"
    echo "   - Environment logs: $model_save_path/env_server.log"
fi

exit $TRAINING_STATUS