# BabyAI Training on RTX 2000 Ada

Simple training setup for BabyAI SFT with Qwen3-0.6B on RunPod GPU instances.

## Quick Start

1. Clone this repository:
```bash
git clone <repository-url>
cd gpu-training-files
```

2. Run setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Start training:
```bash
python train_babyai.py
```

## Configuration

- **Model**: Qwen3-0.6B (fits comfortably in 16GB VRAM)
- **Dataset**: AgentGym/AgentTraj-L (filtered for BabyAI, 100 samples)
- **Batch Size**: 1 with gradient accumulation (effective batch = 8)
- **Memory Optimizations**: FP16, gradient checkpointing

## Files

- `train_babyai.py` - Main training script
- `requirements.txt` - Python dependencies
- `setup.sh` - Environment setup script
- `README.md` - This file

## Expected Output

- Training logs showing loss reduction
- Model checkpoints saved every 50 steps
- Final model saved to `./babyai_qwen3_output/`
- Test generation at the end

## Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training progress
tail -f babyai_qwen3_output/trainer.log
```