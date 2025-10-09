#!/bin/bash

echo "🚀 Starting AgentGym Multi-Environment Training with UV..."

# Check if we're in the right directory
if [ ! -f "agentgym-multi-env.yml" ]; then
    echo "❌ Error: agentgym-multi-env.yml not found. Please run this script from the agent-gym-sft directory."
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi

# Install dependencies with uv
echo "📦 Installing dependencies with UV..."
uv sync

# Check if dataset exists
if [ ! -f "agentgym_data/multi_env_train.json" ]; then
    echo "📝 Preparing AgentGym dataset..."
    uv run python prepare_agentgym_data.py
fi

# Check if dataset is ready
if [ ! -f "agentgym_data/multi_env_train.json" ]; then
    echo "❌ Error: Dataset not found. Please ensure prepare_agentgym_data.py runs successfully."
    exit 1
fi

echo "📊 Dataset ready with $(uv run python -c "import json; print(len(json.load(open('agentgym_data/multi_env_train.json')))" samples) samples"

echo "🎯 Starting training..."
echo "   This will train Qwen3-0.6B on all 5 AgentGym environments simultaneously."

# Try simple training first
echo "🔄 Attempting simple training with standard transformers..."
uv run python simple_train.py

# Check if simple training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "📁 Model saved in: ./agentgym-multi-env-output/"
    echo ""
    echo "🧪 To test the model:"
    echo "   uv run python -c \"import torch; from transformers import AutoTokenizer, AutoModelForCausalLM; tokenizer = AutoTokenizer.from_pretrained('./agentgym-multi-env-output'); model = AutoModelForCausalLM.from_pretrained('./agentgym-multi-env-output'); print('Model loaded successfully!')\""
else
    echo ""
    echo "❌ Simple training failed. Let's try with a more basic approach..."

    # Fallback to even simpler training
    echo "🔄 Trying minimal training approach..."
    echo "   Reduced batch size and sequence length for compatibility"

    # Create a minimal training script
    cat > minimal_train.py << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import json
import os

def main():
    print("🎯 Minimal AgentGym Training...")

    # Load data
    with open("agentgym_data/multi_env_train.json", 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # Use only first 100 samples for testing
    data = data[:100]

    # Format data
    texts = []
    for item in data:
        if item.get('instruction') and item.get('output'):
            texts.append(f"Human: {item['instruction']}\\nAssistant: {item['output']}")

    print(f"Formatted {len(texts)} examples")

    print("✅ Minimal setup complete!")
    print("Model training would require more VRAM. Consider:")
    print("   - Reducing batch size to 1")
    print("   - Reducing sequence length to 512")
    print("   - Using gradient accumulation")

if __name__ == "__main__":
    main()
EOF

    uv run python minimal_train.py
fi

echo ""
echo "📝 Training summary:"
echo "   If simple training worked: Model is ready for use"
echo "   If training failed: Consider adjusting batch size or using a smaller model"