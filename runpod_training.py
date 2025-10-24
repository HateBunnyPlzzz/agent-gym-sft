#!/usr/bin/env python3
"""
RunPod Training Script for Qwen3-0.6B SFT using Axolotl
Setup and run training with minimal commands for RunPod deployment
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, show_output=True):
    """Run command with live output and better logging"""
    print(f"🔄 {description}...")
    print(f"📝 Running: {cmd}")
    print("-" * 50)

    try:
        if show_output:
            # Show live output
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, universal_newlines=True,
                                     bufsize=1)

            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())

            process.wait()

            if process.returncode == 0:
                print("-" * 50)
                print(f"✅ {description} completed successfully")
            else:
                print(f"❌ {description} failed with code {process.returncode}")
                sys.exit(1)
        else:
            # Silent execution for quick commands
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ {description} completed")

    except Exception as e:
        print(f"❌ Error in {description}: {e}")
        sys.exit(1)

def setup_environment():
    """Setup Axolotl environment using PyPI installation"""
    print("🚀 Setting up Axolotl training environment...")

    # Install dependencies
    commands = [
        ("pip install --upgrade pip setuptools wheel ninja", "Upgrading pip and tools"),
        ("pip install datasets accelerate transformers", "Installing core dependencies"),
        ("pip install --no-build-isolation axolotl[flash-attn,deepspeed]", "Installing Axolotl with PyPI"),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc)

def prepare_dataset():
    """Prepare test dataset from HuggingFace"""
    print("📊 Preparing test dataset...")

    # Check if dataset file exists
    dataset_file = Path("agentgym_test_subset.jsonl")
    if not dataset_file.exists():
        print("🔄 Downloading and creating test subset...")
        run_command("python prepare_test_subset.py", "Creating test subset")
    else:
        print(f"✅ Dataset file exists: {dataset_file}")

def run_training():
    """Run Axolotl training"""
    print("🎯 Starting Axolotl training...")

    config_file = "qwen3-0.6b-axolotl-config.yml"

    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)

    # Run training using Axolotl CLI
    training_cmd = f"axolotl train {config_file}"
    run_command(training_cmd, "Axolotl SFT training")

def main():
    """Main execution function"""
    print("🏃‍♂️ Qwen3-0.6B SFT Training on RunPod with Axolotl")
    print("=" * 60)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    try:
        # Step 1: Setup environment
        setup_environment()

        # Step 2: Prepare dataset
        prepare_dataset()

        # Step 3: Run training
        run_training()

        print("\n🎉 Training completed successfully!")
        print("📁 Check outputs in ./qwen3-agentgym-smoke-test/")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()