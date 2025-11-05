#!/bin/bash

# AgentGym-RL Complete Setup Script for RunPod A40
# One-command installation from scratch

set -e

echo "🚀 Setting up AgentGym-RL on RunPod A40"
echo "====================================="

# Install conda
if ! command -v conda &> /dev/null; then
    echo "📦 Installing conda..."
    cd /root
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    source ~/.bashrc
    echo "✅ Conda installed"
else
    echo "✅ Conda already available"
fi

# Accept conda Terms of Service
echo "📋 Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate agentgym-rl environment
echo "🏗️ Creating conda environment..."
conda create -n agentgym-rl python==3.10 -y
source ~/miniconda/etc/profile.d/conda.sh
conda activate agentgym-rl

# Install PyTorch
echo "📚 Installing PyTorch..."
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash attention (optional)
echo "⚡ Installing flash attention..."
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
pip3 install $FLASH_ATTENTION_NAME || echo "⚠️ Flash attention install failed - continuing without it"
rm -f $FLASH_ATTENTION_NAME

# Clone your repository and AgentGym-RL
echo "📥 Cloning repositories..."
cd /root
rm -rf agent-gym-sft AgentGym-RL
git clone --branch dev-0 https://github.com/HateBunnyPlzzz/agent-gym-sft.git
git clone --recursive https://github.com/WooooDyy/AgentGym-RL
cd AgentGym-RL

# Install AgentGym-RL
echo "🔧 Installing AgentGym-RL..."
pip3 install -e .

# Install AgentGym environments
echo "🌍 Installing AgentGym environments..."
cd AgentGym/agentenv
pip3 install -e .
pip3 install transformers==4.51.3

# Download dataset
echo "📊 Downloading AgentGym-RL-Data-ID dataset..."
huggingface-cli download AgentGym/AgentGym-RL-Data-ID --repo-type dataset

# Install BabyAI environment
echo "🍼 Installing BabyAI environment..."
cd agentenv-babyai
pip install -e .

echo ""
echo "✅ Setup complete!"
echo "=================="
echo "Next steps:"
echo "1. Activate environment: conda activate agentgym-rl"
echo "2. Copy training script: cp /root/agent-gym-sft/train_babyai_a40.sh /root/AgentGym-RL/"
echo "3. Start BabyAI server: cd /root/AgentGym-RL/AgentGym/agentenv-babyai && babyai --host 0.0.0.0 --port 36005"
echo "4. Run training: cd /root/AgentGym-RL && bash train_babyai_a40.sh"
echo ""
echo "Environment activated. Ready to start training!"