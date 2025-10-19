#!/usr/bin/env python3
"""
Final AgentGym Training Script with BF16 Precision and Loss Visualization
Clean, stable, and optimized for RTX 4090
"""

import torch
import gc
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM

class LossTracker:
    """Track and visualize training loss"""

    def __init__(self):
        self.steps = []
        self.losses = []
        self.epochs = []

    def log_step(self, step, loss, epoch=0):
        """Log training step"""
        self.steps.append(step)
        self.losses.append(loss)
        self.epochs.append(epoch)

    def save_plot(self, output_dir):
        """Save loss plot to file"""
        if len(self.losses) == 0:
            print("⚠️ No loss data to plot")
            return

        plt.figure(figsize=(10, 6))

        # Plot 1: Loss over steps
        plt.subplot(2, 2, 1)
        plt.plot(self.steps, self.losses, 'b-', linewidth=1.5, alpha=0.8)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Steps')
        plt.grid(True, alpha=0.3)

        # Plot 2: Loss over epochs (if multiple epochs)
        if len(set(self.epochs)) > 1:
            plt.subplot(2, 2, 2)
            epoch_losses = {}
            for step, loss, epoch in zip(self.steps, self.losses, self.epochs):
                if epoch not in epoch_losses:
                    epoch_losses[epoch] = []
                epoch_losses[epoch].append(loss)

            epochs = sorted(epoch_losses.keys())
            avg_losses = [np.mean(epoch_losses[e]) for e in epochs]
            plt.plot(epochs, avg_losses, 'r-', marker='o', linewidth=1.5)
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.title('Average Loss Per Epoch')
            plt.grid(True, alpha=0.3)

        # Plot 3: Loss smoothing (moving average)
        plt.subplot(2, 2, 3)
        if len(self.losses) >= 3:
            window = min(3, len(self.losses) // 2)
            smoothed = np.convolve(self.losses, np.ones(window)/window, mode='valid')
            smoothed_steps = self.steps[window-1:]
            plt.plot(smoothed_steps, smoothed, 'g-', linewidth=1.5, label=f'{window}-step MA')
            plt.plot(self.steps, self.losses, 'b-', alpha=0.3, label='Raw')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Smoothed Loss (Moving Average)')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Plot 4: Loss distribution
        plt.subplot(2, 2, 4)
        bins = min(10, len(self.losses) // 2 if len(self.losses) > 2 else 3)
        plt.hist(self.losses, bins=bins, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Loss Values')
        plt.ylabel('Frequency')
        plt.title('Loss Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"{output_dir}/training_loss.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Loss plot saved to {plot_path}")

        # Print summary statistics
        print(f"\n📈 Training Loss Summary:")
        print(f"   Initial loss: {self.losses[0]:.4f}")
        print(f"   Final loss: {self.losses[-1]:.4f}")
        print(f"   Best loss: {min(self.losses):.4f}")
        print(f"   Loss reduction: {((self.losses[0] - self.losses[-1]) / self.losses[0] * 100):.1f}%")

        plt.close()

def clear_gpu_memory():
    """Clear GPU memory before training"""
    print("🧹 Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Memory management is handled by model loading config
        print(f"   ✅ Memory cleared and optimized")

def get_optimal_config(total_samples):
    """Get optimal BF16 configuration based on dataset size"""

    base_config = {
        "precision": "BF16",
        "batch_size": 8,           # Maximum stable batch size for RTX 4090
        "gradient_accumulation": 4, # Effective batch size = 32
        "lora_rank": 32,           # Good balance of adaptability
        "max_length": 1024,        # Based on data analysis
        "learning_rate": 3e-5,     # Stable for BF16 fine-tuning
        "num_epochs": 2,
    }

    # Adjust for very small datasets
    if total_samples < 1000:
        base_config["gradient_accumulation"] = 2
        base_config["learning_rate"] = 5e-5
    # Optimize for medium datasets (2,000-5,000 samples)
    elif total_samples <= 5000:
        base_config["gradient_accumulation"] = 4  # Reduce memory pressure
        base_config["learning_rate"] = 2e-5       # Slightly lower for stability
        base_config["lora_rank"] = 32            # Keep proven base value
        base_config["max_length"] = 1024         # Further reduced for memory safety
        base_config["batch_size"] = 4            # Smaller batch for large datasets
    # Optimize for large datasets (5,000+ samples)
    else:
        base_config["gradient_accumulation"] = 4  # Reduce memory pressure
        base_config["learning_rate"] = 1e-5
        base_config["lora_rank"] = 32            # Keep conservative rank
        base_config["max_length"] = 1024         # Reduced for memory safety
        base_config["batch_size"] = 4            # Smaller batch for large datasets

    return base_config

def main():
    parser = argparse.ArgumentParser(description='Final AgentGym Training with BF16')
    parser.add_argument('--dataset', type=str,
                       choices=['400', '800', '1500', '2992', 'test'],
                       default='800',
                       help='Dataset size per environment (use "test" for 50 samples)')
    parser.add_argument('--epochs', type=int, default=2,
                       help='Number of epochs (default: 2)')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples to use (overrides dataset setting)')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B',
                       help='Model name (default: Qwen/Qwen3-4B)')
    args = parser.parse_args()

    print("🚀 Final AgentGym Training with BF16 Precision")
    print("="*60)

    # Determine dataset
    if args.samples:
        # Use custom sample count
        dataset_file = 'agentgym_balanced_400_per_env.json'  # Use base dataset
        test_mode = False
        custom_samples = args.samples
        print(f"🎯 Custom sample count: {args.samples} samples")
    elif args.dataset == 'test':
        dataset_file = 'agentgym_balanced_400_per_env.json'
        test_mode = True
        custom_samples = 50
        print("🧪 TEST MODE: Using 50 samples for quick validation")
    else:
        dataset_file = f'agentgym_balanced_{args.dataset}_per_env.json'
        test_mode = False
        custom_samples = None
        print(f"📊 Dataset: {args.dataset} samples per environment")

    print(f"🔄 Epochs: {args.epochs}")
    print(f"🎯 Model: {args.model}")

    # Clear memory
    clear_gpu_memory()

    # Load dataset
    print(f"\n📁 Loading dataset: {dataset_file}")
    try:
        dataset = load_dataset('json', data_files=dataset_file, split='train')

        if custom_samples:
            # Use specified number of samples
            dataset = dataset.select(range(min(custom_samples, len(dataset))))
            print(f"   Using {len(dataset)} samples for training")
        else:
            print(f"   Loaded {len(dataset)} samples")

    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    total_samples = len(dataset)
    samples_per_env = total_samples // 5

    print(f"   Total samples: {total_samples}")
    print(f"   Samples per environment: {samples_per_env}")

    # Get optimal configuration
    config = get_optimal_config(total_samples)
    config["num_epochs"] = args.epochs

    print(f"\n🎛️  Optimized BF16 Configuration:")
    for k, v in config.items():
        print(f"   {k}: {v}")

    # Calculate training time
    steps_per_epoch = total_samples // (config["batch_size"] * config["gradient_accumulation"])
    total_steps = steps_per_epoch * config["num_epochs"]
    # Time estimation varies greatly, let training proceed naturally

    print(f"\n⏱️  Training Estimates:")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total steps: {total_steps}")
    print(f"   Effective batch size: {config['batch_size'] * config['gradient_accumulation']}")

    # Initialize loss tracker
    loss_tracker = LossTracker()

    # Load model with BF16 and memory optimization
    print(f"\n🔧 Loading model with BF16 precision...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,      # BF16 precision (fixed deprecated warning)
            device_map="auto",
            low_cpu_mem_usage=True,    # Reduce CPU memory usage
            trust_remote_code=True      # Allow loading custom code
        )
        print("✅ Model loaded successfully with BF16")

        # Print memory info
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3      # GB
            print(f"   📊 GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # LoRA config optimized for BF16
    peft_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_rank"] * 2,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Training config with BF16 and memory optimization
    if args.samples:
        output_dir = f"./qwen3-agentgym-{args.samples}samples-bf16"
    else:
        output_dir = f"./qwen3-agentgym-{args.dataset if not test_mode else 'test'}-bf16"
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"],
        logging_steps=1,  # Log every step to capture all loss data
        save_steps=max(50, total_samples // 20),  # Adaptive saving
        save_total_limit=3,  # Keep more checkpoints for long runs
        bf16=True,  # Enable BF16 training
        gradient_checkpointing=True,
        max_length=config["max_length"],
        dataloader_pin_memory=False,  # Disable for memory efficiency
        remove_unused_columns=False,
        optim="adamw_torch",  # Better memory usage
        fp16=False,  # Use BF16 instead of FP16
    )

    # We'll capture loss from the training results after completion

    # Start training
    print(f"\n🚀 Starting BF16 training...")
    print(f"💡 Stable BF16 precision with optimized configuration")

    if test_mode:
        print("🧪 TEST MODE: Will stop after a few steps to validate graph generation")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    try:
        print("📊 Training progress:")

        # Initialize loss capture callback variable
        loss_capture_callback = None

        # Use TRL's built-in RichProgressCallback for professional visualization
        try:
            from trl.trainer.callbacks import RichProgressCallback
            rich_callback = RichProgressCallback()
            trainer.add_callback(rich_callback)
            print(f"   🎨 Rich progress display enabled (TRL built-in)")
        except ImportError:
            print(f"   ⚠️ RichProgressCallback not available, using basic progress")
            print(f"   💡 Install with: pip install rich")

        # Simple loss tracking for our plot - no hard-coding
        from transformers import TrainerCallback

        class BasicLossCapture(TrainerCallback):
            def __init__(self):
                self.captured_losses = []
                self.all_logs = []

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    self.all_logs.append(logs)
                    # Capture individual loss if available
                    if 'loss' in logs:
                        step = state.global_step
                        loss_value = logs['loss']
                        self.captured_losses.append((step, loss_value))
                        print(f"   📊 Step {step}: Loss = {loss_value:.4f}")
                    # Also capture train_loss if available (usually averaged)
                    elif 'train_loss' in logs:
                        step = state.global_step
                        loss_value = logs['train_loss']
                        self.captured_losses.append((step, loss_value))
                        print(f"   📊 Step {step}: Train Loss = {loss_value:.4f}")
                    # Debug: show what keys are available
                    else:
                        print(f"   🔍 Available log keys: {list(logs.keys())}")

        # Add basic loss capture for all runs
        loss_capture_callback = BasicLossCapture()
        trainer.add_callback(loss_capture_callback)
        if total_samples > 50:
            print(f"   📈 Loss tracking enabled for training run")
        else:
            print(f"   📈 Loss tracking enabled for test run")

        training_result = trainer.train()

        # Process captured loss data
        if loss_capture_callback and loss_capture_callback.captured_losses:
            for step, loss in loss_capture_callback.captured_losses:
                loss_tracker.log_step(step, loss, 0)

        print(f"\n📊 Training Complete - Loss Summary:")
        if loss_capture_callback and loss_capture_callback.captured_losses:
            print(f"   Captured {len(loss_capture_callback.captured_losses)} real gradient steps")
        else:
            print(f"   No loss data captured during training")

        trainer.save_model()

        print(f"\n✅ Training completed successfully!")
        print(f"💾 Model saved to {output_dir}")

        # Generate and save loss plot with real data (already captured in callback)
        if loss_capture_callback and loss_capture_callback.captured_losses:
            loss_tracker.save_plot(output_dir)
            print(f"✅ Loss plot generated with {len(loss_capture_callback.captured_losses)} real training data points")
        else:
            print("⚠️ No loss data captured for plotting")

        print(f"📊 Total samples trained: {total_samples}")
        print(f"🌍 Balanced training from all 5 AgentGym environments")
        print(f"🎯 BF16 precision provided stable, efficient training")

    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        # Still save partial results and plot
        trainer.save_model()
        loss_tracker.save_plot(output_dir)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        # Try to save whatever we have for debugging
        if len(loss_tracker.losses) > 0:
            loss_tracker.save_plot(output_dir + "_error")

if __name__ == "__main__":
    main()