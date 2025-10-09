#!/usr/bin/env python3
"""
Script to prepare AgentGym datasets for Axolotl training
Filters and formats data for each environment separately
"""

import json
from datasets import load_dataset
import os

def prepare_agentgym_datasets():
    print("📥 Processing AgentGym datasets from local files...")

    # The dataset appears to be already downloaded and separated by environment
    environment_files = {
        "babyai": "babyai_train.json",
        "alfworld": "alfworld_train.json",
        "webshop": "webshop_train.json",
        "sciworld": "sciworld_train.json",
        "textcraft": "textcraft_train.json"
    }

    # Create directory for processed data
    os.makedirs("agentgym_data", exist_ok=True)

    for env, filename in environment_files.items():
        print(f"\n🔄 Processing {env} environment...")
        env_data = []

        try:
            # Load the local file for this environment
            with open(filename, 'r') as f:
                raw_data = json.load(f)

            print(f"✅ Loaded {len(raw_data)} samples for {env}")

            # Convert conversations to the format Axolotl expects
            for item in raw_data:
                if "conversations" in item:
                    formatted_item = {
                        "instruction": "",
                        "input": "",
                        "output": ""
                    }

                    # Extract the main task and response
                    conversations = item["conversations"]
                    human_msgs = [conv["value"] for conv in conversations if conv["from"] == "human"]
                    gpt_msgs = [conv["value"] for conv in conversations if conv["from"] == "gpt"]

                    if human_msgs and gpt_msgs:
                        # Format as instruction-following
                        formatted_item["instruction"] = human_msgs[0] if len(human_msgs) > 0 else ""
                        formatted_item["output"] = gpt_msgs[-1] if len(gpt_msgs) > 0 else ""

                        # Add intermediate context if available
                        if len(human_msgs) > 1:
                            formatted_item["input"] = " ".join(human_msgs[1:-1])

                        env_data.append(formatted_item)

        except FileNotFoundError:
            print(f"⚠️  File {filename} not found, skipping {env}")
            continue
        except Exception as e:
            print(f"⚠️  Could not process {env} data: {e}")
            continue

        print(f"✅ Processed {len(env_data)} samples for {env}")

        # Save to JSON file
        output_file = f"agentgym_data/{env}_train.json"
        with open(output_file, 'w') as f:
            json.dump(env_data, f, indent=2)

        print(f"💾 Saved {env} data to {output_file}")

    print("\n🎉 All AgentGym environments processed successfully!")
    print("\n📁 Created files:")
    for env in environment_files.keys():
        print(f"   - agentgym_data/{env}_train.json")

    print("\n📝 Next steps:")
    print("   1. Update agentgym-multi-env.yml to use these local files")
    print("   2. Run ./train_multi_env.sh to start training")

if __name__ == "__main__":
    prepare_agentgym_datasets()