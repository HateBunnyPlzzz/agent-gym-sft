# AgentGym Multi-Environment Training with Axolotl

**Complete training setup for all AgentGym environments (BabyAI, AlfWorld, WebShop, SciWorld, TextCraft) using Axolotl framework with Qwen3-0.6B.**

## 🎯 Why Axolotl for Multi-Environment Training?

- **Native multi-dataset support** - No custom Python code needed
- **YAML-based configuration** - Simple, declarative setup
- **Memory efficient QLoRA** - 4-bit quantization for all environments
- **Out-of-the-box training** - Works immediately after setup
- **Future-proof** - Supports DPO and other methods for GRPO later

## 🚀 Quick Start

### 1. Setup Environment
```bash
chmod +x setup.sh
./setup.sh
```

### 2. Start Training on All Environments
```bash
chmod +x train_multi_env.sh
./train_multi_env.sh
```

### 3. Test Trained Model
```bash
chmod +x test_model.sh
./test_model.sh
```

## 📁 Files Overview

- `agentgym-multi-env.yml` - **Axolotl configuration for all environments**
- `setup.sh` - **Installation script for Axolotl and dependencies**
- `train_multi_env.sh` - **Training launcher script**
- `test_model.sh` - **Model testing script with Gradio interface**
- `requirements.txt` - **Python dependencies list**
- `train_babyai_unsloth.py` - **Legacy Unsloth script (single environment)**
- `train_babyai.py` - **Legacy standard script (single environment)**

## 🎯 Supported Environments

1. **BabyAI**: Grid-world navigation tasks
2. **AlfWorld**: Household tasks environment
3. **WebShop**: E-commerce tasks
4. **SciWorld**: Science experiments
5. **TextCraft**: Text-based crafting

## ⚙️ Configuration Details

### Model Configuration
- **Base Model**: Qwen/Qwen3-0.6B
- **Training Method**: QLoRA (4-bit quantization)
- **Sequence Length**: 2048 tokens
- **Batch Size**: 2 (micro), 8 (effective with gradient accumulation)

### Training Parameters
- **Learning Rate**: 2e-5
- **Epochs**: 3
- **Optimizer**: paged_adamw_8bit
- **Scheduler**: Cosine
- **Warmup Steps**: 100

### Memory Optimization
- **4-bit Quantization**: Enabled
- **Gradient Checkpointing**: Enabled
- **Sample Packing**: Enabled
- **8-bit Optimizer**: Enabled

## 🔧 Customization

### Training on Specific Environments Only
Edit `agentgym-multi-env.yml` and comment out unwanted environments:

```yaml
datasets:
  # Only train on BabyAI and AlfWorld
  - path: AgentGym/AgentTraj-L
    filter_by_environment: babyai
    name: babyai
  - path: AgentGym/AgentTraj-L
    filter_by_environment: alfworld
    name: alfworld
  # Comment out others...
```

### Adjusting Batch Sizes
For different GPU memory sizes:

**RTX 3090 (24GB)**:
```yaml
micro_batch_size: 4
gradient_accumulation_steps: 2
```

**RTX 2000 Ada (16GB)**:
```yaml
micro_batch_size: 2
gradient_accumulation_steps: 4
```

**A100 (40GB)**:
```yaml
micro_batch_size: 8
gradient_accumulation_steps: 1
```

## 📊 Monitoring Training

### Wandb Integration
Training metrics are automatically logged to Weights & Biases:
- Project: `agentgym-multi-env-sft`
- Run name: `qwen3-0.6b-agentgym-multi-env`

### Local Logs
Check `agentgym-multi-env-output/` for:
- Training logs
- Model checkpoints
- Evaluation results

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce `micro_batch_size` to 1
   - Increase `gradient_accumulation_steps` to 8
   - Enable `gradient_checkpointing: true`

2. **CUDA Errors**:
   - Verify GPU drivers: `nvidia-smi`
   - Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Dataset Loading Issues**:
   - Check internet connection
   - Verify HuggingFace access: `huggingface-cli login`

4. **Slow Training**:
   - Enable `flash_attention: true` if supported
   - Use `sample_packing: true`
   - Increase batch size if memory allows

## 📈 Expected Performance

### Training Time (Approximate)
- **RTX 3090**: 2-3 hours for 3 epochs
- **RTX 2000 Ada**: 4-6 hours for 3 epochs
- **A100**: 1-2 hours for 3 epochs

### Model Quality
The trained model should be able to handle tasks from all 5 environments with improved performance compared to the base Qwen3-0.6B model.

## 🔄 Migration from Unsloth

**Benefits of switching to Axolotl:**
- ✅ **Multi-environment training** (5x environments simultaneously)
- ✅ **No custom Python code** needed for dataset handling
- ✅ **YAML configuration** (easier to modify and share)
- ✅ **Better dataset management** (built-in filtering and combining)
- ✅ **Future-proof** (supports DPO, GRPO, and other methods)

**Migration steps:**
1. Use `agentgym-multi-env.yml` instead of Python scripts
2. Run `./setup.sh` to install Axolotl
3. Use `./train_multi_env.sh` for training
4. All environments are trained automatically!

## 🔗 Advanced Usage

### Multi-GPU Training
```bash
accelerate launch --multi_gpu axolotl train agentgym-multi-env.yml
```

### Custom Dataset Integration
Add your own datasets to `agentgym-multi-env.yml`:
```yaml
datasets:
  - path: your-custom-dataset
    type: conversation
    ds_type: json
    conversation_field: conversations
    roles:
      user: human
      assistant: gpt
```

## 📝 Notes

- The configuration uses QLoRA for memory efficiency while maintaining good performance
- All 5 environments are trained simultaneously, allowing the model to learn multi-task capabilities
- The training automatically handles different data sizes across environments
- Model checkpoints are saved after each epoch for easy resumption

## 🤝 Contributing

To modify the training setup:
1. Edit `agentgym-multi-env.yml` for configuration changes
2. Modify scripts in this directory for workflow changes
3. Test changes with small datasets first