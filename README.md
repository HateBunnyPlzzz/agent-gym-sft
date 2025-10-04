# BabyAI Training on RTX 2000 Ada with Unsloth

**Optimized training setup for BabyAI SFT with Qwen3-0.6B using Unsloth on RunPod GPU instances.**

## 🚀 Why Unsloth?

- **2-5x faster training** than standard transformers
- **80% less memory usage** with 4-bit quantization
- **Larger batch sizes** on limited VRAM
- **Better performance** on RTX 2000 Ada (16GB)

## Quick Start

1. Clone this repository:
```bash
git clone https://github.com/HateBunnyPlzzz/agent-gym-sft.git
cd agent-gym-sft
```

2. Run setup script:
```bash
./setup.sh
```

3. Start Unsloth-optimized training:
```bash
python train_babyai_unsloth.py
```

## Configuration

- **Model**: Qwen3-0.6B + Unsloth + 4-bit quantization + LoRA
- **Dataset**: AgentGym/AgentTraj-L (filtered for BabyAI, 100 samples)
- **Batch Size**: 2 with gradient accumulation (effective batch = 8)
- **Memory Optimizations**: 4-bit quantization, 8-bit optimizer, gradient checkpointing
- **Sequence Length**: 2048 tokens (longer than standard)

## Files

- `train_babyai_unsloth.py` - **Unsloth-optimized training script (recommended)**
- `train_babyai.py` - Standard training script (fallback)
- `requirements.txt` - Python dependencies (includes Unsloth)
- `setup.sh` - Environment setup script
- `README.md` - This file

## Performance Benefits

### Standard Training vs Unsloth
- **Standard**: 1 sample/batch, 512 tokens, ~1.0GB VRAM
- **Unsloth**: 2 samples/batch, 2048 tokens, ~2.5GB VRAM
- **Speed**: 2-5x faster training time
- **Memory**: 80% less VRAM usage

## Expected Output

- Training logs showing rapid loss reduction
- LoRA adapters saved to `./babyai_qwen3_unsloth_output/`
- Merged model saved to `./babyai_qwen3_merged/`
- 2x faster inference generation

## Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training progress
ls -la babyai_qwen3_unsloth_output/
```

## Outputs

1. **LoRA Adapters**: `./babyai_qwen3_unsloth_output/` (lightweight, ~50MB)
2. **Merged Model**: `./babyai_qwen3_merged/` (ready for inference, ~1.5GB)

## Troubleshooting

If Unsloth installation fails, use the fallback:
```bash
python train_babyai.py
```