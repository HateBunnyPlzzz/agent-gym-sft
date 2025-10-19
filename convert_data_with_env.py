#!/usr/bin/env python3
"""
Enhanced AgentGym to TRL data converter with environment tracking
Converts AgentGym conversation format to TRL SFTTrainer format with environment labels
"""

import json
import glob
from typing import List, Dict

def convert_conversation_to_trl(conversations: List[Dict], env_name: str) -> List[Dict]:
    """Convert AgentGym conversation to TRL messages format with environment tracking"""
    training_examples = []
    context = []

    for msg in conversations:
        if msg['from'] == 'human':
            context.append(msg['value'])
        elif msg['from'] == 'gpt' and msg.get('loss') is True:
            # Create training example
            instruction = '\n'.join(context)
            output = msg['value']

            if instruction and output:
                training_examples.append({
                    "messages": [
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": output}
                    ],
                    "environment": env_name  # Add environment tracking
                })

            context.append(msg['value'])  # Add to context for next turn

    return training_examples

def main():
    """Convert all AgentGym datasets with environment tracking"""
    all_data = {}

    # Process each environment dataset separately
    for env_file in glob.glob('datasets/*_train.json'):
        env_name = env_file.split('/')[-1].replace('_train.json', '')
        print(f"Processing {env_name}...")

        env_data = []
        with open(env_file, 'r') as f:
            data = json.load(f)

        for item in data:
            if 'conversations' in item:
                examples = convert_conversation_to_trl(item['conversations'], env_name)
                env_data.extend(examples)

        all_data[env_name] = env_data
        print(f"  Added {len([x for x in data if 'conversations' in x])} trajectories")
        print(f"  Generated {len(env_data)} training examples")

    # Combine all data
    combined_data = []
    for env_name, examples in all_data.items():
        combined_data.extend(examples)

    # Save combined dataset
    with open('agentgym_trl_data_with_env.json', 'w') as f:
        json.dump(combined_data, f, indent=2)

    print(f"\n✅ Converted {len(combined_data)} training examples")
    print(f"💾 Saved to agentgym_trl_data_with_env.json")

    # Create balanced datasets
    print(f"\n🎯 Creating Balanced Datasets...")

    # Find minimum examples across environments
    min_examples = min(len(examples) for examples in all_data.values())
    print(f"   Minimum per environment: {min_examples:,}")

    # Create balanced datasets of different sizes
    dataset_sizes = [500, 1000, 2000, min_examples]

    for size in dataset_sizes:
        balanced_data = []
        for env_name, examples in all_data.items():
            # Take up to 'size' examples from each environment
            balanced_data.extend(examples[:size])

        filename = f'agentgym_balanced_{size}_per_env.json'
        with open(filename, 'w') as f:
            json.dump(balanced_data, f, indent=2)

        print(f"   💾 {filename}: {len(balanced_data):,} examples ({size:,} per env)")

    # Show environment breakdown
    print(f"\n📊 Environment Breakdown:")
    for env_name, examples in all_data.items():
        print(f"   {env_name:12s}: {len(examples):,} examples")

if __name__ == "__main__":
    main()