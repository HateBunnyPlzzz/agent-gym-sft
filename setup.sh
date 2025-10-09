#!/bin/bash

echo "🚀 Setting up Axolotl for AgentGym Multi-Environment Training..."

# Check if we're in the right directory
if [ ! -f "agentgym-multi-env.yml" ]; then
    echo "❌ Error: agentgym-multi-env.yml not found. Please run this script from the gpu-training-files directory."
    exit 1
fi

# Check GPU
echo "🔍 Checking GPU availability..."
nvidia-smi

# Update pip
echo "📦 Updating pip..."
pip install --upgrade pip

# Install Axolotl with DeepSpeed support
echo "📦 Installing Axolotl with DeepSpeed..."
pip install --no-build-isolation axolotl[deepspeed]

# Install additional required packages
echo "📦 Installing additional dependencies..."
pip install datasets transformers accelerate wandb

# Create a requirements file for reference
echo "📝 Creating requirements.txt..."
cat > requirements.txt << EOF
axolotl[deepspeed]>=0.4.0
datasets>=2.14.0
transformers>=4.36.0
accelerate>=0.24.0
wandb>=0.16.0
torch>=2.0.0
tokenizers>=0.15.0
peft>=0.7.0
bitsandbytes>=0.41.0
scipy>=1.10.0
numpy>=1.24.0
EOF

# Verify CUDA availability
echo "🔍 Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Verify Axolotl installation
echo "🔍 Verifying Axolotl installation..."
python -c "import axolotl; print('✅ Axolotl installed successfully')"

echo "✅ Setup complete! You can now run training with:"
echo "   ./train_multi_env.sh"
echo ""
echo "Or manually with:"
echo "   axolotl train agentgym-multi-env.yml"