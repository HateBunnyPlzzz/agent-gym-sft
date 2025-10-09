# AgentGym Multi-Environment Training

🚀 **Training scripts for Qwen3 models on AgentGym environments**

## 🎯 What This Does

Trains Qwen3 models on **5 AgentGym environments simultaneously**:
- **BabyAI** (grid navigation): 810 samples (8.4%)
- **AlfWorld** (household tasks): 2,420 samples (25.1%)
- **SciWorld** (science experiments): 2,120 samples (22.0%)
- **TextCraft** (Minecraft crafting): 374 samples (3.9%)
- **WebShop** (e-commerce): 3,930 samples (40.7%)

**Total**: 9,654 high-quality instruction-following samples

## 📁 Files Overview

### Main Training Scripts
- `final_train.py` - **Universal trainer** (works on CPU/Mac/GPU)
- `final_train_gpu.py` - **GPU-optimized trainer** (BF16/FP16 support)
- `final_train.sh` - **Wrapper script** with environment variables

### Dataset Preparation
- `prepare_agentgym_data.py` - Downloads and formats AgentGym dataset
- `agentgym_data/multi_env_train.json` - Pre-processed training data

### Alternative Approaches
- `simple_train.py` - Basic training script (fallback option)
- `uv_train.sh` - UV-based training script
- `agentgym-multi-env.yml` - Axolotl configuration (if you want to use Axolotl)

## 🎛️ Model Selection

### Available Qwen3 Models
```bash
# Small and fast (recommended for testing)
MODEL="Qwen/Qwen3-0.6B"

# Medium models (good balance of performance/speed)
MODEL="Qwen/Qwen3-1.4B"
MODEL="Qwen/Qwen3-1.8B"

# Large models (best performance, requires more GPU memory)
MODEL="Qwen/Qwen3-4B"
MODEL="Qwen/Qwen3-8B"
MODEL="Qwen/Qwen3-14B"
```

## 🚀 Cloud GPU Training Commands

### Step 1: Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd agent-gym/gpu-training-files

# Install UV package manager (recommended)
pip install uv

# Install dependencies
uv sync

# Prepare dataset (only need to run once)
uv run python prepare_agentgym_data.py
```

### Step 2: Choose Training Method

#### Method A: Universal Training (Recommended)
```bash
# Use the wrapper script - easy customization
./final_train.sh

# Or run directly with custom parameters
uv run python final_train.py \
    --model "Qwen/Qwen3-0.6B" \
    --model-name "qwen3-0.6b" \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --max-seq-length 1024
```

#### Method B: GPU-Optimized Training (Faster on Cloud GPUs)
```bash
# For modern GPUs with BF16 support
uv run python final_train_gpu.py \
    --model "Qwen/Qwen3-0.6B" \
    --model-name "qwen3-0.6b" \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --learning-rate 2e-5 \
    --epochs 3 \
    --max-seq-length 2048
```

#### Method C: Environment Variables (Easy for Cloud Services)
```bash
# Set your model and parameters
export MODEL_NAME="Qwen/Qwen3-0.6B"
export MODEL_SAVE_NAME="qwen3-0.6b"
export BATCH_SIZE=2
export GRAD_ACCUM=4
export LEARNING_RATE=2e-5
export EPOCHS=3
export MAX_SEQ_LENGTH=1024

# Run training
./final_train.sh
```

### Step 3: Advanced Training Options

#### Train on Different Models
```bash
# Train on larger Qwen3 model
uv run python final_train.py \
    --model "Qwen/Qwen3-1.8B" \
    --model-name "qwen3-1.8b" \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --epochs 2

# Train on specific environments only (modify script)
# Edit prepare_agentgym_data.py to filter environments
```

#### Optimize for GPU Memory
```bash
# For limited GPU memory (< 8GB)
uv run python final_train.py \
    --model "Qwen/Qwen3-0.6B" \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --max-seq-length 512

# For large GPU memory (> 16GB)
uv run python final_train_gpu.py \
    --model "Qwen/Qwen3-1.8B" \
    --batch-size 8 \
    --gradient-accumulation 1 \
    --max-seq-length 4096
```

#### Quick Testing (Subset of Data)
```bash
# Test with 100 samples first
uv run python final_train.py \
    --model "Qwen/Qwen3-0.6B" \
    --max-samples 100 \
    --epochs 1
```

## 📊 Training Output

### Model Saving
- **Output Directory**: `./agentgym-{model-name}-output/`
- **Contents**:
  - Model weights (`pytorch_model.bin` or `model.safetensors`)
  - Tokenizer files
  - Configuration files
  - Training checkpoints

### Example Output
```
📁 Model saved to: ./agentgym-qwen3-0.6b-output/
🧪 Model is ready for evaluation on AgentGym environments!

🔍 Environment Testing:
   The trained model is ready for evaluation on:
   - BabyAI (grid navigation)
   - AlfWorld (household tasks)
   - WebShop (e-commerce)
   - SciWorld (science experiments)
   - TextCraft (Minecraft crafting)
```

## 🎛️ Training Parameters

### Batch Size and Memory
| Model Size | Recommended Batch Size | Gradient Accumulation | Approx. GPU Memory |
|------------|----------------------|----------------------|-------------------|
| 0.6B | 2-4 | 2-4 | 6-8GB |
| 1.4B | 1-2 | 4-8 | 8-12GB |
| 1.8B | 1 | 8 | 10-16GB |
| 4B+ | 1 | 8+ | 16GB+ |

### Learning Rates by Model Size
- **0.6B**: 2e-5 to 5e-5
- **1.4B**: 1e-5 to 3e-5
- **1.8B+**: 5e-6 to 2e-5

### Sequence Length Recommendations
- **Testing/Memory Constrained**: 512 or 1024
- **Standard Training**: 1024 or 2048
- **Maximum Performance**: 4096 (requires more memory)

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Out of Memory Errors
```bash
# Reduce batch size and increase gradient accumulation
--batch-size 1 --gradient-accumulation 8

# Reduce sequence length
--max-seq-length 512

# Use smaller model
--model "Qwen/Qwen3-0.6B"
```

#### Training Too Slow
```bash
# Use GPU-optimized script
uv run python final_train_gpu.py

# Increase batch size (if memory allows)
--batch-size 4 --gradient-accumulation 2
```

#### Precision Errors
```bash
# The scripts automatically handle precision:
# - GPUs with BF16 support: Uses BF16 (fastest)
# - GPUs with FP16 support: Uses FP16 (fast)
# - Older GPUs/CPU: Uses FP32 (compatible)
```

## 🧪 Testing Your Trained Model

### Quick Test
```bash
cd agentgym-{model-name}-output

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('.')
model = AutoModelForCausalLM.from_pretrained('.')
print('✅ Model loaded successfully!')
"
```

### Generate Responses
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('./agentgym-qwen3-0.6b-output')
model = AutoModelForCausalLM.from_pretrained('./agentgym-qwen3-0.6b-output')

instruction = "Go to the red ball"
input_text = "You are in a room with a red ball on the floor."

prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🎯 Next Steps

### Environment Evaluation
After training, you can evaluate your model on AgentGym environments:

1. **BabyAI**: Grid-world navigation tasks
2. **AlfWorld**: Household object manipulation
3. **WebShop**: E-commerce product search
4. **SciWorld**: Science experiment reasoning
5. **TextCraft**: Minecraft-style crafting

### GRPO Training (Advanced)
For reinforcement learning training, see the main AgentGym repository for GRPO implementation.

## 📞 Support

- **Dataset Issues**: Check `prepare_agentgym_data.py` output
- **Memory Issues**: Reduce batch size or sequence length
- **Training Errors**: Verify model name and GPU availability
- **Environment Setup**: Ensure UV sync completes successfully

---

**Happy Training! 🚀**