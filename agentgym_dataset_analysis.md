# AgentGym Dataset Structure Analysis & Data Preparation

## Overview

This document provides a comprehensive analysis of the AgentGym/AgentTraj-L dataset structure from HuggingFace and explains how the data has been prepared for Axolotl training.

## Dataset Structure

### Source Dataset
- **Dataset**: AgentGym/AgentTraj-L on HuggingFace
- **Total Samples**: 14,485 training examples
- **Format**: JSON with `conversations` and `item_id` fields
- **Conversation Structure**: Multi-turn interactions between "human" and "gpt"

### Environment Distribution

| Environment | Samples | Percentage | Description |
|-------------|---------|------------|-------------|
| **alfworld** | 2,420 | 16.7% | Household tasks and object manipulation |
| **babyai** | 810 | 5.6% | Grid-world navigation tasks |
| **webshop** | 3,930 | 27.1% | E-commerce shopping tasks |
| **sciworld** | 2,120 | 14.6% | Science experiment simulations |
| **textcraft** | 374 | 2.6% | Minecraft text-based crafting |
| **tool** | 661 | 4.6% | Tool usage (weather, movie, todo) |
| **maze** | 215 | 1.5% | Maze navigation puzzles |
| **sqlgym** | 3,000 | 20.7% | SQL query tasks |
| **wordle** | 955 | 6.6% | Word puzzle game |
| **other** | 0 | 0.0% | Uncategorized |

### Target Environments for Training

For the current training phase, we've focused on the 5 core environments:
- **AlfWorld** (2,420 samples)
- **BabyAI** (810 samples)
- **WebShop** (3,930 samples)
- **SciWorld** (2,120 samples)
- **TextCraft** (374 samples)

**Total**: 9,654 samples

## Conversation Format Analysis

### Common Pattern
All environments follow a ReAct-style format with "Thought" and "Action" components:

```json
{
  "conversations": [
    {"from": "human", "value": "Environment instructions..."},
    {"from": "gpt", "value": "OK. I'll follow your instructions..."},
    {"from": "human", "value": "Current observation/state..."},
    {"from": "gpt", "value": "Thought: ...\nAction: ..."},
    ...
  ],
  "item_id": "environment_identifier"
}
```

### Environment-Specific Patterns

#### AlfWorld
- **Task**: Household object manipulation
- **Format**: Instructions → Environment state → Agent reasoning → Action
- **Example Actions**: `take alarmclock 1 from desk 1`, `use desklamp 1`

#### BabyAI
- **Task**: Grid-world navigation
- **Format**: Instructions → Goal description → Environment observation → Navigation action
- **Example Actions**: `go to red ball 1`

#### WebShop
- **Task**: E-commerce product search and purchase
- **Format**: Shopping instructions → Product search results → Agent reasoning → Purchase action
- **Example Actions**: `search[men's shorts...]`, `click[Buy Now]`

#### SciWorld
- **Task**: Science experiments
- **Format**: Experiment description → Observations → Scientific reasoning → Experimental action
- **Example Actions**: `open door to kitchen`, `wait1`

#### TextCraft
- **Task**: Minecraft crafting
- **Format**: Crafting recipes → Inventory status → Agent reasoning → Crafting action
- **Example Actions**: `craft 1 flint and steel using 1 iron ingot, 1 flint`

## Data Preparation Process

### Extraction & Formatting

The `prepare_agentgym_data.py` script performs the following steps:

1. **Load Dataset**: Downloads AgentTraj-L from HuggingFace
2. **Environment Categorization**: Classifies samples by `item_id` patterns
3. **Conversation Formatting**: Converts multi-turn conversations to instruction-following format
4. **Data Validation**: Ensures all samples have valid instruction-output pairs
5. **File Generation**: Creates individual and combined dataset files

### Formatting Strategy

For each conversation, we extract:

- **Instruction**: First human message (environment instructions)
- **Input**: Current state/observation (second human message + context)
- **Output**: Final GPT response (target action)

#### Environment-Specific Formatting

**WebShop**:
```json
{
  "instruction": "WebShop [SEP] Instruction: [SEP] Find me men's shorts...",
  "input": "Current search results and observations...",
  "output": "Thought: ...\nAction: click[Buy Now]"
}
```

**AlfWorld/BabyAI/SciWorld/TextCraft**:
```json
{
  "instruction": "You are an agent in [environment]...",
  "input": "Current state: You are in a room. You see...",
  "output": "Thought: ...\nAction: take object"
}
```

### Generated Files

The script creates the following files in `/agentgym_data/`:

| File | Samples | Description |
|------|---------|-------------|
| `alfworld_train.json` | 2,420 | AlfWorld environment data |
| `babyai_train.json` | 810 | BabyAI environment data |
| `webshop_train.json` | 3,930 | WebShop environment data |
| `sciworld_train.json` | 2,120 | SciWorld environment data |
| `textcraft_train.json` | 374 | TextCraft environment data |
| `multi_env_train.json` | 9,654 | Combined dataset for training |

### Data Quality Metrics

- **Success Rate**: 100% of samples successfully formatted
- **Validation**: All samples contain valid instruction-output pairs
- **Metadata**: Each sample includes `item_id` and `environment` fields for tracking

## Integration with Axolotl

### Configuration Updates

The `agentgym-multi-env.yml` has been updated to use the prepared datasets:

```yaml
datasets:
  - path: agentgym_data/multi_env_train.json
    type: json
    train_split: train
```

### Training Format

The prepared data follows Axolotl's expected format:
```json
{
  "instruction": "Environment-specific task instructions",
  "input": "Current state/observation context",
  "output": "Agent's reasoned action response"
}
```

## Training Recommendations

### Data Distribution Strategy
1. **Balanced Training**: Use the combined dataset for generalist agent training
2. **Environment-Specific Fine-tuning**: Use individual datasets for specialized performance
3. **Curriculum Learning**: Start with simpler environments (BabyAI) before complex ones (SciWorld)

### Quality Considerations
- **Thought Process**: All outputs include reasoning, promoting better generalization
- **Action Diversity**: Wide range of action types across environments
- **Context Length**: Varies significantly (4-110 turns), consider sequence length limits
- **Environment Coverage**: 5 core environments provide diverse task types

## Future Enhancements

### Additional Environments
The dataset structure supports easy addition of:
- **Tool environments** (weather, movie, todo): 661 samples
- **SQL environments**: 3,000 samples
- **Maze/Wordle puzzles**: 1,170 samples

### Advanced Formatting
- **Multi-turn Training**: Preserve conversation flow for sequential decision-making
- **Reward Integration**: Add environment-specific success signals
- **State Tracking**: Include intermediate states for better context modeling

---

**Last Updated**: 2025-10-09
**Status**: Data preparation completed, ready for training
**Next Steps**: Run `./train_multi_env.sh` to start SFT training