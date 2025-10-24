# AgentGym Development Context

**Current Development Status**: SFT training on AgentGym environments using local RTX 4090, targeting Qwen3-0.6B model. Framework decision: TRL recommended for better modularity.

## 🎯 Primary Objectives

1. **SFT Training**: Supervised fine-tuning on AgentGym trajectory datasets
2. **Online Learning**: GRPO implementation for environment-based RL
3. **Clean Implementation**: Lean codebase without unnecessary files
4. **Multi-Environment Progression**: SFT → GRPO across BabyAI, AlfWorld, WebShop, SciWorld, TextCraft

## 🔧 Qwen3-4B Best Practices (Alibaba Official)

### Model Specifications
- **Parameters**: 4B (4 billion parameters)
- **Architecture**: 36 layers, 32/8 GQA heads
- **Context Length**: 32,768 native, 131,072 with YaRN
- **VRAM Usage**: ~8GB (inference), ~12GB (LoRA training)
- **RTX 4090 Fit**: ✅ Perfect fit with 12GB headroom

### Training Parameters (SFT)
- **Learning Rate**: 5e-5 to 1e-4 for LoRA fine-tuning
- **LoRA Rank**: 32 (higher rank for 4B model)
- **LoRA Alpha**: 64
- **Target Modules**: `all-linear`
- **Batch Size**: 4 (can go up to 8 with gradient accumulation)
- **Epochs**: 3 for initial SFT training
- **Precision**: BF16 (auto-detect)

### Memory Optimization (RTX 4090)
- **Gradient Checkpointing**: Enabled
- **Device Mapping**: `device_map="auto"`
- **Torch Data Type**: `torch_dtype="auto"`
- **Expected VRAM**: ~12GB total with LoRA

### Chat Template & Special Tokens
```python
# Always use official chat template for Qwen3
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # For reasoning tasks
)
```

### AgentGym-Specific Considerations
- **Thinking Mode**: Use `enable_thinking=True` for complex reasoning
- **History Format**: Only include final outputs, not thinking content
- **Output Length**: Use up to 32,768 tokens for complex environment tasks
- **Repetition Control**: Set `presence_penalty=1.5` during generation

## 📝 Latest Discussion Summary

### Framework Decision: TRL vs Axolotl (2025-10-16)

**Recommendation: TRL for Better Modularity**

**Why TRL over Axolotl:**
- ✅ **Research flexibility**: Easy to experiment with different approaches
- ✅ **Cleaner codebase**: Simple Python scripts vs complex YAML
- ✅ **Better debugging**: Direct control over training loop
- ✅ **Future-proof**: Industry standard, more transferable skills
- ✅ **GRPO ready**: Built-in GRPOTrainer for Phase 2
- ✅ **Qwen3 compatible**: Works well with Alibaba's models

**AgentGym Installation**:
- `pip install agentenv` installs environment package for server deployment
- **SFT Phase**: No environment servers needed - uses pre-collected trajectories
- **GRPO Phase**: Environment servers required for real-time interaction

**Training Approach**:
1. **Phase 1 (SFT)**: Train on AgentTraj datasets using TRL
2. **Phase 2 (GRPO)**: Online learning with live environment servers
3. **Code Philosophy**: Minimal, clean implementation - no test files, auto-cleanup

### Environment Scoring & Structure
- **WebShop**: 0.0-1.0 (e-commerce tasks)
- **AlfWorld**: 0.0-1.0 (household tasks, binary success)
- **BabyAI**: 0.0+ (grid navigation, cumulative)
- **SciWorld**: variable range (science experiments)
- **TextCraft**: 0.0+ (text crafting, positive rewards)

## 📁 Current Dataset Structure

### AgentGym Environments (Downloaded - 77MB total)
- **alfworld_train.json**: 1,726 samples (household tasks, 16.5MB)
- **babyai_train.json**: 2,140 samples (grid navigation, 4.1MB)
- **webshop_train.json**: 3,124 samples (e-commerce, 29.8MB)
- **sciworld_train.json**: 2,842 samples (science experiments, 24.9MB)
- **textcraft_train.json**: 742 samples (text crafting, 2.1MB)

### Data Format
- **Structure**: JSON array with `conversations` field
- **Loss Fields**: `loss: true` = training targets, `loss: null` = inputs
- **Pattern**: Human instructions → GPT responses with Thought/Action format
- **Total**: 10,574 training examples across all environments

## 🔧 Active Training Components

### Environment Server Architecture
- **Communication**: HTTP API between GPU server and environment servers
- **Request Flow**: GPU → Environment Server → Model (via base_url) → Environment Server → GPU
- **Scoring**: Single score per conversation (0.0-1.0+ depending on environment)
- **Rollouts**: Batch requests with IDs, max rounds per sample

### Environment Score Ranges
- **WebShop**: 0.0 to 1.0 (normalized rewards)
- **AlfWorld**: 0.0 to 1.0 (binary success/failure)
- **BabyAI**: 0.0 to positive values (cumulative scoring)
- **SciWorld**: Large negative to large positive values
- **TextCraft**: 0.0 to positive values (binary/positive rewards)

## 🚀 Deployment Architecture Decision

### Current Options Under Evaluation

#### Option 1: Serverless Approach
- **Pros**: Pay-per-use, no infrastructure management
- **Cons**: Cold starts, potential latency issues, harder to maintain state
- **Status**: Implemented in `agent-gym-sft/serverless/`

#### Option 2: Dedicated GPU Pod (Recommended)
- **Pros**: Consistent performance, full control, easier state management
- **Cons**: Fixed cost, requires management
- **Implementation**: RunPod deployment with direct SSH access

### Code Synchronization Strategy

**For Dedicated Pod Approach:**
1. **Git Repository**: Clone main repository on pod
2. **Development Loop**: Local development → Git push → Pod pull
3. **Direct File Access**: SSH for direct file editing/sync
4. **Data Storage**: Use pod's persistent storage for models/datasets

**Implementation Plan:**
```bash
# On pod setup
git clone <repository-url>
cd agent-gym
pip install -r requirements.txt

# Development workflow
# Local: Make changes → git push
# Pod: git pull → restart training
```

## 📊 Training Pipeline Status

### Phase 1: SFT Training (Current Priority - FOCUS ON SFT ONLY)
- **Framework**: TRL (SFTTrainer) - chosen for modularity
- **Model**: Qwen3-4B (upgraded from 0.6B for meaningful results)
- **Hardware**: Local RTX 4090 (24GB VRAM, perfect fit)
- **Datasets**: 10,574 samples across 5 AgentGym environments
- **Training Method**: LoRA fine-tuning (rank=32, alpha=64)
- **Status**: Data downloaded, ready for TRL format conversion and training

### Phase 2: GRPO Implementation (Next)
- **Framework**: TRL (GRPOTrainer) - built-in GRPO support
- **Architecture**: GPU server + Environment server communication
- **Challenge**: Real-time model serving during training
- **Advantage**: Same framework as SFT, no migration needed

## 🔗 Key Integration Points

### Model Serving During GRPO
- **Requirement**: VLLM deployment for real-time inference
- **Endpoint**: `http://172.17.0.1:8001` (OpenAI-compatible)
- **Integration**: Environment server queries training model via base_url

### Environment Server Setup
```bash
# Example request from GPU to Environment server
curl -X POST http://127.0.0.1:8081/rollouts \
     -H "Content-Type: application/json" \
     -d '{
           "env": "textcraft",
           "model": "model-name",
           "base_url": "http://172.17.0.1:8001",
           "ids": [0,1,2,3],
           "max_round": 10
         }'
```

## 💰 Hardware & Resource Planning

### Local Hardware Setup
- **GPU**: RTX 4090 (24GB VRAM) available locally
- **VRAM Usage**: ~2-3GB expected with Qwen3-0.6B + LoRA
- **Advantage**: No cloud costs, immediate iteration
- **Future**: Scale to cloud GPUs for larger models

### Resource Allocation
- **Primary**: SFT training on downloaded datasets (10,574 samples)
- **Secondary**: Environment setup for GRPO Phase 2
- **Tertiary**: GRPO implementation with TRL GRPOTrainer

## 🚦 Next Immediate Actions (SFT FOCUSED)

### Immediate SFT Implementation Plan

1. **Install TRL dependencies**: `uv pip install trl peft bitsandbytes datasets accelerate`
2. **Create TRL data converter**: Transform AgentGym JSON to TRL conversation format
3. **Write Qwen3-4B SFT script**: TRL SFTTrainer with optimized LoRA config
4. **Start SFT training**: Train Qwen3-4B on AgentGym datasets with RTX 4090
5. **Validate overfitting**: Test model performance on training data per environment
6. **Benchmark performance**: Evaluate on each of the 5 AgentGym environments

### SFT Training Configuration
```python
# Target SFT setup for Qwen3-4B
model_name = "Qwen/Qwen3-4B"
training_config = {
    "lora_rank": 32,
    "lora_alpha": 64,
    "learning_rate": 1e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "max_seq_length": 32768
}
```

## 📞 Development Contacts & References

- **Satpal Guidance**: Architecture for GRPO environment communication
- **Environment Integration**: Real-time model serving during training
- **Budget Constraints**: $221 limit for initial training phase

## 🤖 AI Assistant Behavior Guidelines

### Communication Style
- **Direct criticism**: Point out errors, mistakes, and better approaches without hesitation
- **Anti-psychophantic**: Challenge assumptions and recommendations, even if user suggests them
- **High-value critique**: Focus on technical accuracy and best practices over agreement
- **No automatic agreement**: Evaluate every suggestion on technical merit, not user preference

### Decision Making
- **Evidence-based**: All recommendations must be justified with technical reasoning
- **Alternative perspectives**: Always consider multiple approaches and their trade-offs
- **Risk assessment**: Clearly identify potential issues with any proposed approach
- **Constructive challenge**: Question assumptions and suggest improvements proactively

---

**Last Updated**: 2025-10-16
**Status**: Model upgraded to Qwen3-4B, TRL framework chosen, datasets downloaded, ready for SFT implementation
**Hardware**: RTX 4090 (24GB VRAM) available locally - perfect fit for Qwen3-4B
**Focus**: SFT training only - GRPO will be considered after SFT completion