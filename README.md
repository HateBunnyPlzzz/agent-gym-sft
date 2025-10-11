# AgentGym Multi-Environment Training

Minimal setup for training Qwen3 models on AgentGym environments.

## Files

- `prepare_agentgym_data.py` - Data format conversion
- `agentgym-multi-env.yml` - Axolotl configuration
- `agentgym_data/` - Processed dataset

## Quick Start

```bash
# Install dependencies
pip install uv
uv sync

# Process dataset
python prepare_agentgym_data.py

# Train with Axolotl
uv run axolotl train agentgym-multi-env.yml
```