#!/bin/bash

echo "🚀 Final AgentGym Multi-Environment Training"
echo "=========================================="

# Default model and parameters
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
MODEL_SAVE_NAME="${MODEL_SAVE_NAME:-qwen3-0.6b}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
EPOCHS="${EPOCHS:-3}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

echo "📋 Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Save Name: $MODEL_SAVE_NAME"
echo "   Batch Size: $BATCH_SIZE"
echo "   Grad Accumulation: $GRAD_ACCUM"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Epochs: $EPOCHS"
echo "   Max Sequence Length: $MAX_SEQ_LENGTH"
echo "   Max Samples: ${MAX_SAMPLES:-all}"

# Check GPU
echo "🔍 Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  nvidia-smi not found. GPU training may not be available."
else
    nvidia-smi
fi

# Install dependencies with UV
echo "📦 Installing dependencies with UV..."
if [ ! -d ".venv" ]; then
    uv sync
fi

# Prepare dataset if needed
if [ ! -f "agentgym_data/multi_env_train.json" ]; then
    echo "📝 Preparing AgentGym dataset..."
    uv run python prepare_agentgym_data.py
    if [ $? -ne 0 ]; then
        echo "❌ Dataset preparation failed!"
        exit 1
    fi
fi

# Dataset info
SAMPLES=$(uv run python -c 'import json; print(len(json.load(open("agentgym_data/multi_env_train.json")))')
echo "📊 Dataset ready: $SAMPLES samples"

# Build command
CMD="uv run python final_train.py \
    --model \"$MODEL_NAME\" \
    --model-name \"$MODEL_SAVE_NAME\" \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation $GRAD_ACCUM \
    --learning-rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --max-seq-length $MAX_SEQ_LENGTH"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

echo "🎯 Starting training..."
echo "   Command: $CMD"

# Execute training
eval $CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Model saved to: ./agentgym-$MODEL_SAVE_NAME-output/"
    echo ""
    echo "🧪 To test the model:"
    echo "   cd agentgym-$MODEL_SAVE_NAME-output"
    echo "   python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; tokenizer = AutoTokenizer.from_pretrained('.'); model = AutoModelForCausalLM.from_pretrained('.'); print('✅ Model loaded successfully!')\""
    echo ""
    echo "🔍 Environment Testing:"
    echo "   The trained model is ready for evaluation on:"
    echo "   - BabyAI (grid navigation)"
    echo "   - AlfWorld (household tasks)"
    echo "   - WebShop (e-commerce)"
    echo "   - SciWorld (science experiments)"
    echo "   - TextCraft (Minecraft crafting)"
else
    echo ""
    echo "❌ Training failed. Consider:"
    echo "   - Reducing batch size: --batch-size 1"
    echo "   - Reducing sequence length: --max-seq-length 512"
    echo "   - Using gradient accumulation: --gradient-accumulation 8"
    echo "   - Using fewer samples: --max-samples 1000"
    echo "   - Checking GPU memory availability"
fi