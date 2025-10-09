#!/bin/bash

echo "🧪 Testing Trained AgentGym Multi-Environment Model..."

# Check if model output directory exists
if [ ! -d "agentgym-multi-env-output" ]; then
    echo "❌ Error: No trained model found in ./agentgym-multi-env-output/"
    echo "   Please run ./train_multi_env.sh first."
    exit 1
fi

# Check for adapter files
adapter_path="agentgym-multi-env-output"
if [ ! -f "$adapter_path/adapter_model.bin" ] && [ ! -f "$adapter_path/adapter_model.safetensors" ]; then
    echo "❌ Error: No adapter model found in output directory."
    echo "   Training may not have completed successfully."
    exit 1
fi

echo "🔍 Loading model and testing with sample prompts..."
echo ""

# Run inference with Axolotl
accelerate launch -m axolotl.cli.inference agentgym-multi-env.yml \
    --lora_model_dir="$adapter_path" \
    --gradio

echo ""
echo "✅ Model testing complete!"
echo "   If the Gradio interface didn't start automatically, you can run:"
echo "   accelerate launch -m axolotl.cli.inference agentgym-multi-env.yml --lora_model_dir=agentgym-multi-env-output --gradio"