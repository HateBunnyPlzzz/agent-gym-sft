#!/usr/bin/env python3
"""
Prepare medium subset (10,000 samples) from AgentGym SFT trajectories dataset
for 8B model training with stratified sampling across environments
"""

import json
import random
from datasets import load_dataset
from pathlib import Path
from collections import defaultdict

def create_stratified_subset():
    """Download dataset and create 10,000 sample subset with stratified sampling"""

    print("🔄 Downloading AgentGym SFT trajectories dataset...")

    # Download from HuggingFace
    dataset = load_dataset("bunnybhaiya/agentgym-sft-trajectories")

    print(f"📊 Dataset info:")
    print(f"   Total samples: {len(dataset['train'])}")
    print(f"   Dataset size: ~459MB")

    # Group samples by environment
    env_samples = defaultdict(list)
    total_samples = len(dataset['train'])

    print("🔍 Analyzing environment distribution...")
    for i, item in enumerate(dataset['train']):
        env = item.get('environment', 'unknown')
        env_samples[env].append(i)

    print(f"📈 Environment distribution in full dataset:")
    for env, indices in env_samples.items():
        print(f"   {env}: {len(indices)} samples")

    # Calculate target samples per environment (total: 10,000)
    target_total = 10000
    target_distribution = {
        'alfworld': int(target_total * 0.28),      # 2,800 samples (28%)
        'webshop': int(target_total * 0.24),       # 2,400 samples (24%)
        'babyai': int(target_total * 0.20),        # 2,000 samples (20%)
        'sciworld': int(target_total * 0.18),      # 1,800 samples (18%)
        'textcraft': int(target_total * 0.10),     # 1,000 samples (10%)
    }

    print(f"\n🎯 Target distribution for 10K sample subset:")
    for env, target_count in target_distribution.items():
        print(f"   {env}: {target_count} samples")

    # Sample from each environment
    subset_indices = []
    actual_distribution = defaultdict(int)

    random.seed(42)  # For reproducibility

    for env, target_count in target_distribution.items():
        available_indices = env_samples.get(env, [])

        if len(available_indices) < target_count:
            print(f"⚠️  Warning: {env} has only {len(available_indices)} samples, using all of them")
            selected = available_indices
        else:
            selected = random.sample(available_indices, target_count)

        subset_indices.extend(selected)
        actual_distribution[env] = len(selected)
        print(f"✅ {env}: Selected {len(selected)} samples")

    # Shuffle the final subset
    random.shuffle(subset_indices)

    # Create subset data
    subset_data = [dataset['train'][i] for i in subset_indices]

    print(f"\n🎯 10K subset created with {len(subset_data)} samples")
    print(f"📈 Actual distribution: {dict(actual_distribution)}")

    # Convert to Axolotl format (same as test subset)
    axolotl_data = []
    for item in subset_data:
        messages = []

        # Use existing messages format
        if 'messages' in item:
            messages = item['messages']

        if messages:
            axolotl_data.append({
                "messages": messages,
                "environment": item.get('environment', 'unknown')
            })

    # Quality filtering
    print(f"\n🔍 Applying quality filters...")

    # Filter by conversation length (remove very short or very long)
    filtered_data = []
    for item in axolotl_data:
        total_chars = sum(len(msg['content']) for msg in item['messages'])

        # Reasonable length for agent tasks (500 to 8000 characters)
        if 500 <= total_chars <= 8000:
            filtered_data.append(item)

    print(f"   Length filter: {len(axolotl_data)} -> {len(filtered_data)} samples")

    # Ensure we have exactly 10,000 samples (or as close as possible)
    if len(filtered_data) > target_total:
        filtered_data = filtered_data[:target_total]
    elif len(filtered_data) < target_total:
        print(f"⚠️  Warning: After filtering, we have {len(filtered_data)} samples (target was {target_total})")

    # Save as JSONL for Axolotl
    output_file = Path("agentgym_medium_8b_subset.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item) + '\n')

    print(f"\n✅ 8B model dataset saved to: {output_file}")
    print(f"   Formatted for Axolotl training")
    print(f"   Final samples: {len(filtered_data)}")

    # Show sample structure
    if filtered_data:
        print(f"\n📝 Sample structure:")
        sample = filtered_data[0]
        print(f"   Environment: {sample.get('environment', 'unknown')}")
        print(f"   Messages: {len(sample['messages'])} turns")

        first_msg = sample['messages'][0]
        print(f"   First message type: {first_msg.get('role', 'unknown')}")
        print(f"   Content preview: {first_msg.get('content', '')[:100]}...")

        # Show environment distribution in final dataset
        final_env_counts = defaultdict(int)
        for item in filtered_data:
            env = item.get('environment', 'unknown')
            final_env_counts[env] += 1

        print(f"\n📊 Final environment distribution:")
        for env, count in final_env_counts.items():
            percentage = (count / len(filtered_data)) * 100
            print(f"   {env}: {count} samples ({percentage:.1f}%)")

    return output_file

if __name__ == "__main__":
    try:
        subset_file = create_stratified_subset()
        print(f"\n🚀 Ready for 8B model training!")
        print(f"   Use this dataset file: {subset_file}")
        print(f"   Training command: axolotl train qwen3-8b-axolotl-config.yml")

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 Make sure you have internet connection and 'datasets' library installed")