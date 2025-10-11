# AgentGym Development Context

**Current Development Status**: Training Qwen3 models on AgentGym environments with GRPO and SFT approaches.

## 🎯 Primary Objectives

1. **Train smallest Qwen3 model** from the family
2. **Decide deployment architecture**: Serverless vs dedicated GPU pod
3. **Implement GRPO training loop** with environment server integration
4. **Set up SFT baseline** with readily available datasets

## 📁 Current Codebase Structure

```
agent-gym/
├── AgentGym/                    # Main AgentGym environments (git submodule)
│   ├── agentenv-alfworld/       # Household tasks environment
│   ├── agentenv-babyai/         # Grid-world navigation
│   ├── agentenv-textcraft/      # Text-based crafting
│   ├── agentenv-sciworld/       # Science experiments
│   └── agentenv-webshop/        # E-commerce tasks
├── agent-gym-sft/               # SFT training implementation
│   └── serverless/              # Serverless training approach
├── simple_qwen_trainer.py       # Simplified Qwen trainer
├── runpod_training.py           # RunPod deployment script
└── configs/                     # Training configurations
```

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

### Phase 1: SFT Baseline (Current Priority)
- **Objective**: Establish performance baseline with existing datasets
- **Status**: Ready to implement
- **Datasets**: AgentGym SFT datasets + custom generation
- **Model**: Smallest Qwen3 variant (TBD exact size)

### Phase 2: GRPO Implementation (Next)
- **Objective**: Implement environment-based reinforcement learning
- **Architecture**: GPU server + Environment server communication
- **Challenge**: Real-time model serving during training

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

## 💰 Budget & Resource Planning

### Current Budget Constraints
- **Target**: $221 for initial training
- **GPU Options**: RTX 3090 (~320 hours) or A100 (~185 hours)
- **Priority**: Maximum training time within budget

### Resource Allocation
- **Primary**: SFT training on existing datasets
- **Secondary**: Environment setup for GRPO
- **Tertiary**: GRPO implementation and testing

## 🚦 Next Immediate Actions

1. **Finalize deployment architecture decision** (serverless vs dedicated pod)
2. **Set up SFT baseline training** with smallest Qwen3 model
3. **Prepare environment servers** for GRPO implementation
4. **Implement dataset generation pipeline** for continuous training

## 📞 Development Contacts & References

- **Satpal Guidance**: Architecture for GRPO environment communication
- **Environment Integration**: Real-time model serving during training
- **Budget Constraints**: $221 limit for initial training phase

---

**Last Updated**: 2025-10-05
**Status**: Architecture decision pending, SFT implementation ready