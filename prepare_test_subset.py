#!/usr/bin/env python3
"""
Prepare small test subset from AgentGym SFT trajectories dataset
for initial testing on RunPod with Axolotl
"""

import json
import random
from datasets import load_dataset
from pathlib import Path

def download_and_create_subset():
    """Download dataset and create 100-200 sample subset"""

    print("🔄 Downloading AgentGym SFT trajectories dataset...")

    # Download from HuggingFace
    dataset = load_dataset("bunnybhaiya/agentgym-sft-trajectories")

    print(f"📊 Dataset info:")
    print(f"   Total samples: {len(dataset['train'])}")
    print(f"   Dataset size: ~459MB")

    # Create subset of 150 samples for testing
    total_samples = len(dataset['train'])
    subset_size = min(150, total_samples)

    # Randomly sample across environments
    random.seed(42)  # For reproducibility
    indices = random.sample(range(total_samples), subset_size)

    subset_data = [dataset['train'][i] for i in indices]

    # Analyze environment distribution
    env_counts = {}
    for item in subset_data:
        env = item.get('environment', 'unknown')
        env_counts[env] = env_counts.get(env, 0) + 1

    print(f"🎯 Subset created with {len(subset_data)} samples")
    print(f"📈 Environment distribution: {env_counts}")

    # Save subset in Axolotl format
    axolotl_data = []
    for item in subset_data:
        # Convert to Axolotl conversation format (OpenAI style with role/content)
        messages = []

        # Check if it's already in messages format (OpenAI style)
        if 'messages' in item:
            messages = item['messages']
        elif 'conversations' in item:
            # Convert from conversations format to messages format
            for conv in item['conversations']:
                role = conv.get('from', 'user')
                if role == 'human':
                    role = 'user'
                elif role == 'gpt':
                    role = 'assistant'
                elif role == 'system':
                    role = 'system'

                messages.append({
                    "role": role,
                    "content": conv.get('value', '')
                })
        else:
            # Try to extract from raw data
            if 'text' in item:
                messages = [
                    {"role": "user", "content": item.get('instruction', '')},
                    {"role": "assistant", "content": item['text']}
                ]

        if messages:
            axolotl_data.append({
                "messages": messages,
                "environment": item.get('environment', 'unknown')
            })

    # Save as JSONL for Axolotl
    output_file = Path("agentgym_test_subset.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in axolotl_data:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Test subset saved to: {output_file}")
    print(f"   Formatted for Axolotl training")
    print(f"   Samples: {len(axolotl_data)}")

    # Show sample structure
    if axolotl_data:
        print(f"\n📝 Sample structure:")
        print(f"   Environment: {axolotl_data[0].get('environment', 'unknown')}")
        print(f"   Conversations: {len(axolotl_data[0]['conversations'])} turns")

        first_conv = axolotl_data[0]['conversations'][0]
        print(f"   First message type: {first_conv.get('from', 'unknown')}")
        print(f"   Content preview: {first_conv.get('value', '')[:100]}...")

    return output_file

if __name__ == "__main__":
    try:
        subset_file = download_and_create_subset()
        print(f"\n🚀 Ready for Axolotl training!")
        print(f"   Use this dataset file in your Axolotl config: {subset_file}")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Make sure you have internet connection and 'datasets' library installed")