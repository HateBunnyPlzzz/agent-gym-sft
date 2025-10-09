#!/bin/bash
# AgentGym Quick Start Training Script
# Choose one of the following training methods:

echo "🚀 AgentGym Quick Start Training"
echo "================================="

# Method 1: Train smallest model (0.6B) - Good for testing
echo "Method 1: Small Model Training (0.6B, ~1-2 hours on RTX 3090)"
echo "uv run python final_train.py --model 'Qwen/Qwen3-0.6B' --model-name 'qwen3-0.6b' --batch-size 2 --gradient-accumulation 4 --epochs 3 --max-seq-length 1024"
echo ""

# Method 2: GPU-optimized training (faster on modern GPUs)
echo "Method 2: GPU-Optimized Training (BF16/FP16, ~30-50% faster)"
echo "uv run python final_train_gpu.py --model 'Qwen/Qwen3-0.6B' --model-name 'qwen3-0.6b' --batch-size 4 --gradient-accumulation 2 --epochs 3 --max-seq-length 2048"
echo ""

# Method 3: Larger model training (1.8B, better performance)
echo "Method 3: Large Model Training (1.8B, ~3-4 hours on RTX 3090)"
echo "uv run python final_train.py --model 'Qwen/Qwen3-1.8B' --model-name 'qwen3-1.8b' --batch-size 1 --gradient-accumulation 8 --epochs 2 --max-seq-length 1024"
echo ""

# Method 4: Quick test (100 samples, 1 epoch)
echo "Method 4: Quick Test (100 samples, ~10 minutes)"
echo "uv run python final_train.py --model 'Qwen/Qwen3-0.6B' --max-samples 100 --epochs 1"
echo ""

# Method 5: Environment variables approach
echo "Method 5: Environment Variables (Easy for cloud services)"
echo "export MODEL_NAME='Qwen/Qwen3-0.6B'"
echo "export MODEL_SAVE_NAME='qwen3-0.6b'"
echo "export BATCH_SIZE=2"
echo "export GRAD_ACCUM=4"
echo "export LEARNING_RATE=2e-5"
echo "export EPOCHS=3"
echo "export MAX_SEQ_LENGTH=1024"
echo "./final_train.sh"
echo ""

echo "📋 Dataset Preparation (run once):"
echo "uv run python prepare_agentgym_data.py"
echo ""

echo "🧪 Model Testing (after training):"
echo "cd agentgym-qwen3-0.6b-output"
echo "python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; tokenizer = AutoTokenizer.from_pretrained('.'); model = AutoModelForCausalLM.from_pretrained('.'); print('✅ Model loaded successfully!')\""
echo ""

echo "📖 For detailed instructions, see: TRAINING_GUIDE.md"
echo "💡 Copy and paste any command above to start training!"