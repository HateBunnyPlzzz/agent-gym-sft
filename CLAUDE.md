# AgentGym-RL Development Context

**Current Status**: Setting up AgentGym-RL on WSL2 with RTX 4090 using existing vllm-venv

---

## 🎯 Project Goal

Get AgentGym-RL working for training. Nothing more.

---

## 🚨 Core Principles

1. **Make it work, don't optimize** - Functionality before performance
2. **Vanilla setup only** - Use AgentGym-RL as-is, no custom modifications
3. **No assumptions** - Always ask before implementing architectural decisions
4. **Simple debugging** - Use print() not logging frameworks
5. **Stop at success criteria** - Don't over-engineer

---

## 🖥️ Current Environment

### Hardware
- **GPU**: RTX 4090 (24GB VRAM)
- **OS**: WSL2
- **Driver**: NVIDIA 581.57 (CUDA 13.0 support)

### Software Stack
- **Python Env**: `vllm-venv` (already exists)
- **PyTorch**: 2.8.0+cu128 (CUDA 12.8 runtime included)
- **Package Manager**: `uv` (already installed)
- **CUDA Toolkit**: Not installed (not needed - PyTorch has CUDA runtime)

### Key Insight
✅ **No CUDA Toolkit needed** - PyTorch wheels include CUDA runtime
✅ **Flash-attention optional** - Can skip if installation fails
✅ **uv works fine** - Compatible with conda-based setups
✅ **vllm-venv works** - Reuse this environment

---

## ✅ Setup Phases

### Phase 1: Install AgentGym-RL (CURRENT)
```bash
# Activate existing environment
source vllm-venv/bin/activate

# Clone repositories (recursive for submodules)
git clone --recursive https://github.com/WooooDyy/AgentGym-RL
cd AgentGym-RL

# Install AgentGym-RL and dependencies
uv pip install -e .

# Install agentenv (required for environments)
cd AgentGym/agentenv
uv pip install -e .
cd ../..

# Install additional required packages
uv pip install transformers==4.51.3 accelerate peft datasets bitsandbytes

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import agentgym_rl; print('AgentGym-RL OK')"
python -c "import agentenv; print('AgentEnv OK')"
```

### Phase 2: Flash-Attention (Optional)
```bash
# Only attempt if Phase 1 succeeds
# Skip if this fails - it's optional for basic functionality
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
uv pip install $FLASH_ATTENTION_NAME || echo "Flash-attention install failed - skipping (optional)"
rm -f $FLASH_ATTENTION_NAME
```

### Phase 3: Basic Test
```bash
# Test that everything imports without errors
python -c "
import torch
import agentgym_rl
import agentenv
print('✅ All imports successful')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```

---

## 🎯 Success Criteria

### Minimum Working System
- [x] vllm-venv activated
- [ ] AgentGym-RL cloned with `--recursive`
- [ ] AgentGym-RL installed with `uv pip install -e .`
- [ ] agentenv installed from AgentGym/agentenv
- [ ] All required packages installed (transformers, accelerate, etc.)
- [ ] `import agentgym_rl` works without errors
- [ ] `import agentenv` works without errors
- [ ] CUDA is available in PyTorch

**STOP HERE when all checkboxes are checked. System is working.**

---

## ⚙️ Model Requirements (From README)

### Required Models
- **Qwen2.5-3B** (for testing)
- **Qwen2.5-7B** (for full training)

### Environment Support
- **WebArena** (web navigation)
- **Search-R1** (deep search)
- **TextCraft** (digital games)
- **BabyAI** (embodied tasks)
- **SciWorld** (scientific tasks)

---

## 🐛 Common Issues & Solutions

### Issue: UV vs Conda Compatibility
**Answer**: ✅ UV works perfectly fine. AgentGym-RL uses standard Python packages that UV can install.

### Issue: Flash-Attention on WSL2
**Answer**: ✅ Flash Attention 2.7.3 supports WSL2. Use pre-built wheels for CUDA 12.4 + PyTorch 2.4. If installation fails, skip it - it's optional.

### Issue: Missing --recursive flag
**Solution**: Always clone with `git clone --recursive` to get submodules including AgentGym.

### Issue: PyTorch version conflict
**Solution**: You have PyTorch 2.8.0+cu128. This should work fine with AgentGym-RL despite README mentioning 2.4.

---

## 📁 Project Structure

```
agent-gym-sft/
├── vllm-venv/              # Existing virtual environment (reuse this)
├── AgentGym-RL/            # RL training framework (clone here)
│   ├── AgentGym/           # Submodule with environments
│   │   └── agentenv/       # Environment package
│   ├── AgentGym-RL/        # Main package
│   └── examples/           # Training scripts
└── CLAUDE.md               # This file
```

---

## 🔑 Key Commands

### Environment Management
```bash
# Activate
source vllm-venv/bin/activate

# Install packages
uv pip install package-name

# Install from current directory
uv pip install -e .
```

### Testing Commands
```bash
# Test CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Test AgentGym-RL
python -c "import agentgym_rl; print('OK')"

# Test agentenv
python -c "import agentenv; print('OK')"
```

---

## 🚫 Don't Do These

- ❌ Install conda (UV is sufficient)
- ❌ Install CUDA Toolkit (PyTorch has CUDA runtime)
- ❌ Modify AgentGym-RL source code (use vanilla setup)
- ❌ Skip --recursive flag when cloning
- ❌ Install flash-attention from source (use pre-built wheels)

---

## 📝 Current Task

**Right now**: Phase 1 - Install AgentGym-RL with UV package manager

**Next**: Phase 2 - Optional flash-attention installation

**After that**: Phase 3 - Verify all imports work

---

## 💬 Claude Assistant Guidelines

1. **Challenge over-engineering** - Point out when user suggests unnecessary complexity
2. **Ask before assuming** - Don't implement major features without confirmation
3. **Debug incrementally** - Test one thing at a time
4. **Document failures** - Note what doesn't work and why
5. **Stop at success** - Don't add features beyond working system

---

**Last Updated**: 2025-11-04
**Environment**: vllm-venv on WSL2
**GPU**: RTX 4090 (24GB)
**Status**: Ready for Phase 1 - Install AgentGym-RL