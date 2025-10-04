#!/bin/bash
echo "🚀 Setting up BabyAI training environment..."

# Check GPU
echo "Checking GPU availability..."
nvidia-smi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify CUDA availability
echo "Verifying CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

echo "✅ Setup complete! Run 'python train_babyai.py' to start training."