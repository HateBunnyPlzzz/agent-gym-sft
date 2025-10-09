#!/bin/bash

echo "🚀 Starting AgentGym Multi-Environment Training with Axolotl..."

# Check if required files exist
if [ ! -f "agentgym-multi-env.yml" ]; then
    echo "❌ Error: agentgym-multi-env.yml not found. Please run this script from the gpu-training-files directory."
    exit 1
fi

# Check if Axolotl is installed
python -c "import axolotl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Axolotl not found. Please run ./setup.sh first."
    exit 1
fi

# Display training configuration
echo "📋 Training Configuration:"
echo "   Config File: agentgym-multi-env.yml"
echo "   Model: Qwen/Qwen3-0.6B"
echo "   Method: QLoRA (4-bit quantization)"
echo "   Environments: BabyAI, AlfWorld, WebShop, SciWorld, TextCraft"
echo ""

# Check if GPU is available
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: No GPU detected. Training will be very slow without CUDA."
    echo "    Consider running on a GPU-enabled machine."
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create output directory if it doesn't exist
mkdir -p agentgym-multi-env-output

# Start training
echo "🎯 Starting training..."
echo "   This will train on all 5 AgentGym environments simultaneously."
echo "   Training logs will be displayed below."
echo ""

# Run Axolotl training
axolotl train agentgym-multi-env.yml

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "📁 Model saved in: ./agentgym-multi-env-output/"
    echo ""
    echo "🧪 To test the trained model, run:"
    echo "   ./test_model.sh"
    echo ""
    echo "📊 To view training logs and metrics, check the output directory."
else
    echo ""
    echo "❌ Training failed. Please check the error messages above."
    echo "   Common issues:"
    echo "   - Out of memory: Try reducing micro_batch_size or gradient_accumulation_steps"
    echo "   - Dataset not found: Check your internet connection and HuggingFace access"
    echo "   - CUDA errors: Verify GPU drivers and CUDA installation"
    exit 1
fi