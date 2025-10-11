#!/usr/bin/env python3
"""
Improved script to prepare AgentGym datasets from HuggingFace AgentTraj-L for Axolotl training
Filters and formats data for each environment separately
"""

import json
from datasets import load_dataset
import os
from typing import List, Dict, Any
from tqdm import tqdm

def load_agentgym_data():
    """Load AgentTraj-L dataset from HuggingFace"""
    print("📥 Loading AgentGym/AgentTraj-L dataset from HuggingFace...")
    dataset = load_dataset('AgentGym/AgentTraj-L')
    print(f"✅ Loaded {len(dataset['train'])} total samples")
    return dataset

def categorize_by_environment(dataset):
    """Categorize samples by environment based on item_id patterns"""
    print("🔄 Categorizing samples by environment...")

    env_samples = {
        'alfworld': [],
        'babyai': [],
        'webshop': [],
        'sciworld': [],
        'textcraft': [],
        'tool': [],  # weather, movie, todo, academia, sheet
        'maze': [],  # lmrlgym maze
        'sqlgym': [],  # SQL queries
        'wordle': [],  # lmrlgym wordle
        'other': []
    }

    for i, item in enumerate(dataset['train']):
        item_id = item['item_id']

        if item_id.startswith('webshop_'):
            env_samples['webshop'].append(i)
        elif item_id.startswith('babyai_') or item_id.startswith('BabyAI-'):
            env_samples['babyai'].append(i)
        elif item_id.startswith('sciworld_'):
            env_samples['sciworld'].append(i)
        elif item_id.startswith('textcraft_'):
            env_samples['textcraft'].append(i)
        elif item_id.startswith(('weather_', 'movie_', 'todo_', 'academia_', 'sheet_')):
            env_samples['tool'].append(i)
        elif item_id.startswith('lmrlgym_maze_'):
            env_samples['maze'].append(i)
        elif item_id.startswith('sqlgym_'):
            env_samples['sqlgym'].append(i)
        elif item_id.startswith('lmrlgym_wordle_'):
            env_samples['wordle'].append(i)
        elif 'trial_' in item_id or item_id.startswith(('put', 'look', 'pick', 'go', 'open', 'close', 'clean', 'fill', 'heat', 'cool', 'slice')):
            env_samples['alfworld'].append(i)
        else:
            env_samples['other'].append(i)

    # Print environment statistics
    print("\n📊 Environment distribution:")
    total = len(dataset['train'])
    for env, indices in env_samples.items():
        count = len(indices)
        percentage = (count / total) * 100
        print(f"   - {env:10s}: {count:4d} samples ({percentage:5.1f}%)")

    return env_samples

def format_conversation_for_training(conversations: List[Dict[str, Any]], env: str) -> Dict[str, str]:
    """
    Format conversations into Axolotl alpaca format

    Args:
        conversations: List of conversation dictionaries with 'from' and 'value' keys
        env: Environment name for context-aware formatting

    Returns:
        Dictionary with 'instruction', 'input', and 'output' keys in alpaca format
    """
    human_msgs = [conv["value"] for conv in conversations if conv["from"] == "human"]
    gpt_msgs = [conv["value"] for conv in conversations if conv["from"] == "gpt"]

    if not human_msgs or not gpt_msgs:
        return None

    # Alpaca format expects specific structure
    if env == 'webshop':
        # For WebShop: instruction = task, input = context, output = action
        instruction = human_msgs[1] if len(human_msgs) >= 2 else human_msgs[0]
        input_text = " | ".join(human_msgs[2:-1]) if len(human_msgs) > 2 else ""
        output = gpt_msgs[-1]

    elif env in ['alfworld', 'babyai', 'sciworld', 'textcraft']:
        # For these environments
        instruction = human_msgs[0]  # Task/goal
        if len(human_msgs) > 1:
            input_text = human_msgs[1]  # Current state
            if len(human_msgs) > 2:
                input_text += " | " + " | ".join(human_msgs[2:-1])  # Additional context
        else:
            input_text = ""
        output = gpt_msgs[-1]

    else:
        # Default formatting
        instruction = human_msgs[0] if len(human_msgs) > 0 else ""
        input_text = " | ".join(human_msgs[1:-1]) if len(human_msgs) > 1 else ""
        output = gpt_msgs[-1] if len(gpt_msgs) > 0 else ""

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "item_id": "",  # Will be filled later
        "environment": env
    }

def prepare_environment_data(dataset, env_samples: Dict[str, List[int]], target_envs: List[str]):
    """Prepare training data for specific environments"""
    print(f"\n🔄 Preparing data for environments: {', '.join(target_envs)}")

    # Create directory for processed data
    os.makedirs("agentgym_data", exist_ok=True)

    results = {}

    for env in tqdm(target_envs, desc="Processing environments"):
        if env not in env_samples or not env_samples[env]:
            print(f"⚠️  No samples found for {env}, skipping...")
            continue

        print(f"\n📝 Processing {env} environment...")
        env_data = []

        indices = env_samples[env]
        print(f"   Found {len(indices)} samples for {env}")

        processed_count = 0
        for idx in indices:
            item = dataset['train'][idx]

            if "conversations" in item and item["conversations"]:
                formatted_item = format_conversation_for_training(
                    item["conversations"],
                    env
                )

                if formatted_item and formatted_item["instruction"] and formatted_item["output"]:
                    # Add metadata for tracking
                    formatted_item["item_id"] = item["item_id"]
                    formatted_item["environment"] = env
                    env_data.append(formatted_item)
                    processed_count += 1

        print(f"   ✅ Successfully processed {processed_count} samples for {env}")

        # Save to JSON file
        output_file = f"agentgym_data/{env}_train.json"
        with open(output_file, 'w') as f:
            json.dump(env_data, f, indent=2)

        print(f"   💾 Saved {env} data to {output_file}")
        results[env] = len(env_data)

    return results

def create_multi_environment_dataset(results: Dict[str, int]):
    """Create a combined dataset for multi-environment training"""
    print("\n🔄 Creating combined multi-environment dataset...")

    combined_data = []

    for env, count in results.items():
        if count > 0:
            input_file = f"agentgym_data/{env}_train.json"
            if os.path.exists(input_file):
                with open(input_file, 'r') as f:
                    env_data = json.load(f)
                    combined_data.extend(env_data)

    # Save combined dataset
    output_file = "agentgym_data/multi_env_train.json"
    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"💾 Combined dataset saved to {output_file}")
    print(f"   Total samples: {len(combined_data)}")

    return output_file

def update_axolotl_config(config_file: str = "agentgym-multi-env.yml"):
    """Update Axolotl configuration to use the prepared datasets"""
    print(f"\n📝 Updating Axolotl configuration: {config_file}")

    # Check if config exists
    if not os.path.exists(config_file):
        print(f"⚠️  Configuration file {config_file} not found, skipping update...")
        return

    # Read existing config
    with open(config_file, 'r') as f:
        config_content = f.read()

    # Update datasets section to use local files
    new_datasets = """
datasets:
  - path: agentgym_data/multi_env_train.json
    type: json
    train_split: train
"""

    # Replace datasets section (simple approach)
    lines = config_content.split('\n')
    new_lines = []
    skip_next = False

    for i, line in enumerate(lines):
        if line.strip().startswith('datasets:'):
            # Replace entire datasets section
            new_lines.append(new_datasets)
            skip_next = True
        elif skip_next and line.strip().startswith('  - path:'):
            continue  # Skip old dataset entries
        elif skip_next and line.strip() and not line.startswith('    '):
            skip_next = False
            new_lines.append(line)
        elif not skip_next:
            new_lines.append(line)

    # Write updated config
    with open(config_file, 'w') as f:
        f.write('\n'.join(new_lines))

    print("✅ Configuration updated to use local AgentGym datasets")

def main():
    """Main function to prepare AgentGym datasets"""
    print("🚀 AgentGym Data Preparation Started")
    print("=" * 50)

    # Load dataset from HuggingFace
    dataset = load_agentgym_data()

    # Categorize by environment
    env_samples = categorize_by_environment(dataset)

    # Define target environments for current training
    target_envs = ['alfworld', 'babyai', 'webshop', 'sciworld', 'textcraft']

    # Prepare data for target environments
    results = prepare_environment_data(dataset, env_samples, target_envs)

    # Create combined multi-environment dataset
    combined_file = create_multi_environment_dataset(results)

    # Update Axolotl configuration
    update_axolotl_config()

    print("\n🎉 AgentGym data preparation completed successfully!")
    print("\n📁 Created files:")
    for env in results.keys():
        print(f"   - agentgym_data/{env}_train.json")
    print(f"   - {combined_file}")

    print("\n📝 Dataset summary:")
    total_samples = sum(results.values())
    for env, count in results.items():
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"   - {env:10s}: {count:4d} samples ({percentage:5.1f}%)")
    print(f"   - {'TOTAL':10s}: {total_samples:4d} samples")

    print("\n🚀 Next steps:")
    print("   1. Review the prepared data files")
    print("   2. Run ./train_multi_env.sh to start training")
    print("   3. Monitor training progress and adjust as needed")

if __name__ == "__main__":
    main()