#!/bin/bash

# AgentGym-SFT Complete Setup Script for RunPod
# This script sets up both RL training pipeline and AgentGym environments
# RunPod A40 GPU | CUDA 12.4 | Python 3.10

set -e  # Exit on any error

echo "🚀 Starting AgentGym-SFT complete setup on RunPod..."
echo "📍 Branch: dev-0"
echo "🎯 Setup: RL Training + AgentGym Environments"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print section headers
print_section() {
    echo ""
    echo "📍 $1"
    echo "----------------------------------------"
}

# Step 1: Environment Check
print_section "Step 1: GPU and Environment Check"
nvidia-smi
echo "✅ GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "✅ CUDA Version: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}')"

# Step 2: Install UV
print_section "Step 2: Installing UV Package Manager"
if command_exists uv; then
    echo "✅ UV already installed: $(uv --version)"
else
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    echo "✅ UV installed: $(uv --version)"
fi

# Step 3: Clone agent-gym-sft repository
print_section "Step 3: Cloning agent-gym-sft Repository (dev-0 branch)"
if [ -d "agent-gym-sft" ]; then
    echo "⚠️  agent-gym-sft directory already exists, removing..."
    rm -rf agent-gym-sft
fi

git clone -b dev-0 https://github.com/HateBunnyPlzzz/agent-gym-sft.git
cd agent-gym-sft
echo "✅ Repository cloned successfully"
echo "📂 Current directory: $(pwd)"
echo "🌿 Current branch: $(git branch --show-current)"

# Step 4: Setup RL Training Pipeline
print_section "Step 4: Setting up RL Training Pipeline"

# Clone AgentGym-RL if not present as submodule
if [ ! -d "AgentGym-RL" ]; then
    echo "📥 Cloning AgentGym-RL submodule..."
    git clone --recursive https://github.com/WooooDyy/AgentGym-RL.git
else
    echo "✅ AgentGym-RL already exists"
fi

# Create RL training environment
echo "🐍 Creating Python 3.10 environment for RL training..."
uv venv python=3.10 --name rl-training
source .venv/bin/activate

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch 2.4.0 with CUDA 12.4..."
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention (optional)
echo "⚡ Installing Flash Attention (optional)..."
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
if uv pip install $FLASH_ATTENTION_NAME; then
    echo "✅ Flash Attention installed successfully"
else
    echo "⚠️  Flash Attention installation failed - continuing without it (optional)"
fi
rm -f $FLASH_ATTENTION_NAME

# Navigate to AgentGym-RL package and install
cd AgentGym-RL
if [ -d "AgentGym-RL" ]; then
    cd AgentGym-RL
    echo "🤖 Installing AgentGym-RL RL training package..."
    uv pip install -e .
    cd ../..
else
    echo "❌ AgentGym-RL package directory not found"
    exit 1
fi

# Install RL dependencies
echo "📚 Installing RL training dependencies..."
uv pip install transformers==4.51.3
uv pip install accelerate peft bitsandbytes
uv pip install wandb tensorboard
uv pip install gymnasium
uv pip install datasets numpy scipy matplotlib

cd ..  # Back to agent-gym-sft root

# Step 5: Setup AgentGym Environments (separate venv)
print_section "Step 5: Setting up AgentGym Environments"

# Create separate environment for AgentGym environments
echo "🐍 Creating Python 3.10 environment for AgentGym environments..."
uv venv python=3.10 --name agentgym-envs
source .venv/bin/activate

# Install PyTorch for environments
echo "🔥 Installing PyTorch for environments..."
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install AgentGym environments
cd AgentGym-RL/AgentGym/agentenv
echo "🌍 Installing AgentGym environments..."
uv pip install -e .
cd ../../..

# Install environment dependencies
echo "📚 Installing environment dependencies..."
uv pip install transformers==4.51.3
uv pip install accelerate peft datasets bitsandbytes

cd ../../../..  # Back to agent-gym-sft root

# Step 6: Verification
print_section "Step 6: Installation Verification"

# Test RL training environment
echo "🔍 Testing RL Training environment..."
source .venv/bin/activate  # rl-training venv
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

python -c "
try:
    import agentgym_rl
    print('✅ AgentGym-RL RL training package imported successfully')
except ImportError as e:
    print(f'❌ AgentGym-RL import failed: {e}')
"

# Test AgentGym environments
echo "🔍 Testing AgentGym environments..."
# Check if we can access environment configs
if [ -d "AgentGym-RL/AgentGym/agentenv" ]; then
    echo "✅ AgentGym environment package found"
    echo "📂 Available environments:"
    ls AgentGym-RL/AgentGym/agentenv/ | head -10
else
    echo "❌ AgentGym environment package not found"
fi

# Step 7: Final Instructions
print_section "Step 7: Setup Complete! 🎉"

echo "🎯 AgentGym-SFT setup completed successfully!"
echo ""
echo "📋 Environment Activation Commands:"
echo ""
echo "🤖 For RL Training:"
echo "   source ~/agent-gym-sft/.venv/bin/activate"
echo "   cd AgentGym-RL/AgentGym-RL"
echo "   python examples/train_grpo.py --config configs/qwen2.5-3b.yaml"
echo ""
echo "🌍 For AgentGym Environments:"
echo "   source ~/agent-gym-sft/.venv/bin/activate"
echo "   cd AgentGym-RL/AgentGym/agentenv"
echo "   python -c \"import agentenv; print('Environments ready')\""
echo ""
echo "📂 Project Structure:"
echo "   ~/agent-gym-sft/                    # Your codebase"
echo "   ├── AgentGym-RL/                    # RL training framework"
echo "   │   └── AgentGym-RL/               # RL package"
echo "   └── .venv/                         # RL training environment"
echo ""
echo "🔗 Useful Commands:"
echo "   - Check GPU: nvidia-smi"
echo "   - Activate RL env: source .venv/bin/activate"
echo "   - Check models: ls AgentGym-RL/AgentGym-RL/configs/"
echo "   - TensorBoard: tensorboard --logdir=logs"
echo ""
echo "🚀 Ready to train AgentGym-RL with your custom codebase!"
echo "💡 Branch: dev-0 | GPU: A40 | Environments: Ready"