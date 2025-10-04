#!/bin/bash
echo "🚀 Setting up BabyAI training environment with Unsloth..."

# Check GPU
echo "Checking GPU availability..."
nvidia-smi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify CUDA availability
echo "Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Verify Unsloth installation
echo "Verifying Unsloth installation..."
python -c "from unsloth import FastLanguageModel; print('✅ Unsloth installed successfully')"

echo "✅ Setup complete! Run 'python train_babyai_unsloth.py' for Unsloth-optimized training."