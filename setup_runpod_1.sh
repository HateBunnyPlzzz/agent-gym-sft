#!/bin/bash

# AgentGym-SFT Clean Setup Script for RunPod A40 GPU
# Sets up RL training pipeline and AgentGym environments
# Usage: Run from any directory - script handles setup automatically

set -e  # Exit on any error

echo "🚀 AgentGym-SFT Setup for RunPod A40 GPU"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "${BLUE}📍 $1${NC}"
    echo "----------------------------------------"
}

# Step 1: Environment Check
print_header "Step 1: GPU Environment Check"
nvidia-smi
print_status "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
print_status "CUDA: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}')"

# Step 2: UV Installation
print_header "Step 2: UV Package Manager"
if command -v uv &> /dev/null; then
    print_status "UV already installed: $(uv --version)"
else
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
    print_status "UV installed successfully"
fi

# Step 3: Repository Setup
print_header "Step 3: Repository Setup"
CURRENT_DIR=$(basename "$PWD")

if [ "$CURRENT_DIR" = "agent-gym-sft" ]; then
    print_status "Already in agent-gym-sft repository"
    WORK_DIR="$PWD"
    print_status "Branch: $(git branch --show-current)"
else
    print_warning "Not in agent-gym-sft directory"

    # Clean up any existing agent-gym-sft directories
    if [ -d "agent-gym-sft" ]; then
        print_warning "Removing existing agent-gym-sft directory"
        rm -rf agent-gym-sft
    fi

    echo "📥 Cloning agent-gym-sft (dev-0 branch)..."
    git clone -b dev-0 https://github.com/HateBunnyPlzzz/agent-gym-sft.git

    if [ -d "agent-gym-sft" ]; then
        cd agent-gym-sft
        WORK_DIR="$PWD"
        print_status "Repository cloned successfully"
        print_status "Working directory: $WORK_DIR"
        print_status "Branch: $(git branch --show-current)"
    else
        print_error "Failed to clone repository"
        exit 1
    fi
fi

# Step 4: AgentGym-RL Setup
print_header "Step 4: AgentGym-RL Framework"

# Check if AgentGym-RL already exists
if [ -d "AgentGym-RL" ]; then
    print_status "AgentGym-RL directory already exists"
else
    echo "📥 Cloning AgentGym-RL..."
    git clone --recursive https://github.com/WooooDyy/AgentGym-RL.git
    print_status "AgentGym-RL cloned successfully"
fi

# Verify AgentGym-RL structure
if [ -d "AgentGym-RL/AgentGym-RL" ]; then
    print_status "AgentGym-RL package structure verified"
else
    print_error "AgentGym-RL package structure not found"
    echo "Available directories:"
    find AgentGym-RL -maxdepth 2 -type d
    exit 1
fi

# Step 5: RL Training Environment
print_header "Step 5: RL Training Environment"

# Create virtual environment
echo "🐍 Creating Python 3.10 virtual environment..."
uv venv python=3.10
source .venv/bin/activate
print_status "Virtual environment created and activated"

# Install PyTorch with CUDA
echo "🔥 Installing PyTorch 2.4.0 with CUDA 12.4..."
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
print_status "PyTorch installed successfully"

# Install Flash Attention (optional)
echo "⚡ Installing Flash Attention (optional)..."
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
if uv pip install $FLASH_ATTENTION_NAME; then
    print_status "Flash Attention installed successfully"
else
    print_warning "Flash Attention installation failed (optional - continuing)"
fi
rm -f $FLASH_ATTENTION_NAME

# Install AgentGym-RL package
echo "🤖 Installing AgentGym-RL RL training package..."
cd AgentGym-RL/AgentGym-RL
uv pip install -e .
cd ../../
print_status "AgentGym-RL package installed"

# Install RL dependencies
echo "📚 Installing RL dependencies..."
uv pip install transformers==4.51.3
uv pip install accelerate peft bitsandbytes
uv pip install wandb tensorboard
uv pip install gymnasium datasets numpy scipy matplotlib
print_status "RL dependencies installed"

# Step 6: AgentGym Environments
print_header "Step 6: AgentGym Environments"

# Create separate environment for environments
echo "🐍 Creating environment for AgentGym environments..."
uv venv python=3.10
source .venv/bin/activate

# Install PyTorch for environments
echo "🔥 Installing PyTorch for environments..."
uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install AgentGym environments
echo "🌍 Installing AgentGym environments..."
cd AgentGym-RL/AgentGym/agentenv
uv pip install -e .
cd ../../../../
print_status "AgentGym environments installed"

# Install environment dependencies
echo "📚 Installing environment dependencies..."
uv pip install transformers==4.51.3
uv pip install accelerate peft datasets bitsandbytes
print_status "Environment dependencies installed"

# Step 7: Verification
print_header "Step 7: Installation Verification"

echo "🔍 Testing RL Training environment..."
source .venv/bin/activate  # rl-training environment

python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ CUDA not available')
"

python -c "
try:
    import agentgym_rl
    print('✅ AgentGym-RL imported successfully')
except ImportError as e:
    print(f'❌ AgentGym-RL import failed: {e}')
"

# Step 8: Final Instructions
print_header "Setup Complete! 🎉"

echo ""
echo "🎯 AgentGym-SFT setup completed successfully!"
echo ""
echo "📂 Project Structure:"
echo "   $WORK_DIR/"
echo "   ├── AgentGym-RL/           # RL training framework"
echo "   │   └── AgentGym-RL/      # RL package"
echo "   └── .venv/                # RL training environment"
echo ""
echo "🔧 Environment Commands:"
echo ""
echo "🤖 RL Training:"
echo "   source $WORK_DIR/.venv/bin/activate"
echo "   cd $WORK_DIR/AgentGym-RL/AgentGym-RL"
echo "   python examples/train_grpo.py --config configs/qwen2.5-3b.yaml"
echo ""
echo "🌍 AgentGym Environments:"
echo "   source $WORK_DIR/.venv/bin/activate"
echo "   cd $WORK_DIR/AgentGym-RL/AgentGym/agentenv"
echo "   python -c \"import agentenv; print('Environments ready')\""
echo ""
echo "📋 Useful Commands:"
echo "   - Check GPU: nvidia-smi"
echo "   - Check models: ls $WORK_DIR/AgentGym-RL/AgentGym-RL/configs/"
echo "   - TensorBoard: tensorboard --logdir=logs"
echo ""
echo "🚀 Ready to train AgentGym-RL!"
print_status "Branch: dev-0 | GPU: A40 | Setup: Complete"