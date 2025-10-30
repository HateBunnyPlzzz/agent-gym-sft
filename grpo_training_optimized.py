#!/usr/bin/env python3
"""
Optimized GRPO Training for RTX 4090
Balanced performance for AgentGym environments
"""

import json
import random
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
import time
from datetime import datetime

def random_reward_function(completions, **kwargs):
    """Random reward function for testing"""
    return [random.uniform(0.0, 1.0) for _ in completions]

class OptimizedTrainingLogger:
    """Enhanced logger optimized for faster training"""

    def __init__(self, save_interval=25):
        self.save_interval = save_interval
        self.steps = []
        self.losses = []
        self.rewards = []
        self.entropies = []
        self.gradient_norms = []
        self.learning_rates = []
        self.epoch_times = []
        self.start_time = None

    def start_training(self):
        """Initialize training timer"""
        self.start_time = time.time()
        print(f"🚀 Optimized training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def log_step(self, step, metrics):
        """Log training step metrics"""
        current_time = time.time()

        self.steps.append(step)
        self.losses.append(metrics.get('loss', 0))
        self.rewards.append(metrics.get('reward', 0))
        self.entropies.append(metrics.get('entropy', 0))
        self.gradient_norms.append(metrics.get('grad_norm', 0))
        self.learning_rates.append(metrics.get('learning_rate', 0))

        if self.start_time:
            self.epoch_times.append(current_time - self.start_time)

    def print_progress(self, step, total_steps, metrics):
        """Print training progress"""
        progress = (step / total_steps) * 100
        loss = metrics.get('loss', 0)
        reward = metrics.get('reward', 0)
        entropy = metrics.get('entropy', 0)

        # Calculate ETA (optimized for faster training)
        if len(self.epoch_times) > 1:
            recent_times = np.diff(self.epoch_times[-5:])  # Last 5 steps
            avg_time_per_step = np.mean(recent_times)
            remaining_steps = total_steps - step
            eta_seconds = remaining_steps * avg_time_per_step
            eta = f"{eta_seconds/60:.1f}m"
        else:
            eta = "calculating..."

        print(f"Step {step:4d}/{total_steps} ({progress:5.1f}%) | "
              f"Loss: {loss:7.4f} | Reward: {reward:5.3f} | "
              f"Entropy: {entropy:5.3f} | ETA: {eta}")

    def create_comprehensive_plots(self, save_path="./grpo_training_metrics_optimized.png"):
        """Create comprehensive training metrics visualization"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('GRPO Training Metrics - Optimized RTX 4090 Configuration', fontsize=16, fontweight='bold')

        # Convert to numpy arrays
        steps = np.array(self.steps)
        losses = np.array(self.losses)
        rewards = np.array(self.rewards)
        entropies = np.array(self.entropies)
        grad_norms = np.array(self.gradient_norms)
        learning_rates = np.array(self.learning_rates)

        def plot_with_moving_avg(ax, data, title, ylabel, window_size=15, color='blue'):
            """Plot data with moving average and confidence bands"""
            if len(data) < window_size:
                window_size = len(data)

            # Calculate moving statistics
            moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            moving_std = []

            for i in range(len(moving_avg)):
                window_data = data[i:i+window_size]
                moving_std.append(np.std(window_data))
            moving_std = np.array(moving_std)

            ma_steps = steps[window_size-1:]

            # Plot raw data
            ax.plot(steps, data, alpha=0.3, color=color, linewidth=0.5, label='Raw data')

            # Plot moving average
            ax.plot(ma_steps, moving_avg, color=color, linewidth=2.5,
                   label=f'Moving avg (window={window_size})')

            # Add confidence bands
            ax.fill_between(ma_steps, moving_avg - moving_std, moving_avg + moving_std,
                           alpha=0.2, color=color, label='±1 std')

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Create subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Loss plot (larger, top row)
        ax1 = fig.add_subplot(gs[0, :])
        plot_with_moving_avg(ax1, losses, 'Training Loss', 'Loss', window_size=20, color='red')

        # Reward plot (larger, middle row)
        ax2 = fig.add_subplot(gs[1, :])
        plot_with_moving_avg(ax2, rewards, 'Average Reward', 'Reward', window_size=20, color='green')

        # Three smaller plots in bottom row
        ax3 = fig.add_subplot(gs[2, 0])
        plot_with_moving_avg(ax3, entropies, 'Policy Entropy', 'Entropy', window_size=20, color='blue')

        ax4 = fig.add_subplot(gs[2, 1])
        plot_with_moving_avg(ax4, grad_norms, 'Gradient Norm', 'Grad Norm', window_size=20, color='orange')

        ax5 = fig.add_subplot(gs[2, 2])
        ax5.plot(steps, learning_rates, color='purple', linewidth=2)
        ax5.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Training Steps')
        ax5.set_ylabel('Learning Rate')
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Optimized training metrics saved to: {save_path}")
        plt.show()

    def print_training_summary(self):
        """Print training summary statistics"""
        if not self.start_time:
            return

        total_time = time.time() - self.start_time

        print("\n" + "="*60)
        print("🎯 OPTIMIZED TRAINING SUMMARY")
        print("="*60)
        print(f"⏱️  Total training time: {total_time/60:.1f} minutes")
        print(f"📈 Total steps: {len(self.steps)}")
        print(f"⚡ Average time per step: {total_time/len(self.steps):.2f} seconds")
        print(f"")
        print(f"📊 Final Metrics:")
        print(f"  - Final Loss: {self.losses[-1]:.4f}")
        print(f"  - Final Reward: {self.rewards[-1]:.4f}")
        print(f"  - Final Entropy: {self.entropies[-1]:.4f}")
        print(f"  - Final Grad Norm: {self.gradient_norms[-1]:.4f}")
        print(f"")
        print(f"📈 Training Progress:")
        if len(self.losses) > 10:
            initial_avg = np.mean(self.losses[:10])
            final_avg = np.mean(self.losses[-10:])
            loss_improvement = ((initial_avg - final_avg) / abs(initial_avg)) * 100
            print(f"  - Loss improvement: {loss_improvement:+.1f}%")

            initial_reward = np.mean(self.rewards[:10])
            final_reward = np.mean(self.rewards[-10:])
            reward_improvement = ((final_reward - initial_reward) / abs(initial_reward)) * 100 if initial_reward != 0 else 0
            print(f"  - Reward improvement: {reward_improvement:+.1f}%")
        print("="*60)

def load_prompts_from_dataset(dataset_path, num_samples=None):
    """Extract prompts from GRPO dataset"""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    prompts = []
    environments = []

    for item in data:
        # Extract the prompt content
        prompt_content = item['prompt'][0]['content']
        prompts.append(prompt_content)
        environments.append(item['environment'])

    if num_samples:
        prompts = prompts[:num_samples]
        environments = environments[:num_samples]

    print(f"📂 Loaded {len(prompts)} prompts from {dataset_path}")

    # Show environment distribution
    env_counts = {}
    for env in environments:
        env_counts[env] = env_counts.get(env, 0) + 1

    print("📈 Environment distribution:")
    for env, count in sorted(env_counts.items()):
        percentage = (count / len(environments)) * 100
        print(f"  - {env}: {count} samples ({percentage:.1f}%)")

    return prompts

def main():
    print("🚀 Optimized GRPO Training - RTX 4090 Performance")
    print("=" * 60)

    # Configuration
    DATASET_PATH = "grpo_dataset_1000.json"
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    OUTPUT_DIR = "./grpo-training-optimized"
    NUM_SAMPLES = 300  # Use all samples

    # OPTIMIZED Training parameters for RTX 4090
    TRAINING_EPOCHS = 3
    BATCH_SIZE = 4          # Fixed for generation compatibility
    GRADIENT_ACCUMULATION = 4  # ↑ Better gradient estimates
    GENERATIONS_PER_PROMPT = 4   # ↑ Optimal for GRPO + RTX 4090 speed (was 2, now 8 for effective GRPO)
    MAX_COMPLETION_LENGTH = 500  # ↑ Safer for AgentGym (was 300)
    LEARNING_RATE = 5e-5      # ↑ Faster convergence (was 3e-5)

    # Calculate training steps
    steps_per_epoch = NUM_SAMPLES // (BATCH_SIZE * GRADIENT_ACCUMULATION)
    total_steps = steps_per_epoch * TRAINING_EPOCHS

    print(f"⚙️ OPTIMIZED Training Configuration:")
    print(f"  - Dataset: {NUM_SAMPLES} samples")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - Epochs: {TRAINING_EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE} (compatible with generations)")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION} (better gradients)")
    print(f"  - Generations per prompt: {GENERATIONS_PER_PROMPT} (ultra-fast optimization)")
    print(f"  - Max completion length: {MAX_COMPLETION_LENGTH} (safe for AgentGym)")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Total training steps: {total_steps}")
    print(f"")
    print(f"🚀 OPTIMIZED Performance + GRPO Effectiveness:")
    print(f"  - 8 generations: Optimal for GRPO advantage calculation")
    print(f"  - Better gradient estimates (4 vs 2 accum)")
    print(f"  - Safer for AgentGym (500 vs 300 tokens)")
    print(f"  - Higher learning rate for faster convergence")
    print(f"  - RTX 4090 optimized: Max speed without sacrificing GRPO quality")

    # Initialize logger
    logger = OptimizedTrainingLogger(save_interval=25)
    logger.start_training()

    # Load dataset
    prompts = load_prompts_from_dataset(DATASET_PATH, NUM_SAMPLES)
    dataset = Dataset.from_dict({"prompt": prompts})

    # OPTIMIZED GRPO Configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_generations=GENERATIONS_PER_PROMPT,
        generation_batch_size=GENERATIONS_PER_PROMPT,
        max_completion_length=MAX_COMPLETION_LENGTH,
        max_prompt_length=1024,
        temperature=0.9,
        top_k=50,
        beta=0.05,  # KL regularization coefficient
        logging_steps=5,  # More frequent logging for optimization
        save_steps=100,
        num_train_epochs=TRAINING_EPOCHS,
        # Memory optimization
        gradient_checkpointing=True,
        bf16=True,
        report_to="none",
        # Optimizer settings
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        # Performance optimizations
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # Custom progress callback using TrainerCallback
    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def __init__(self, logger, total_steps):
            self.logger = logger
            self.total_steps = total_steps

        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            if logs:
                step = state.global_step
                self.logger.log_step(step, logs)

                # Print progress every 25 steps (more frequent for optimization)
                if step % 25 == 0 or step == self.total_steps:
                    self.logger.print_progress(step, self.total_steps, logs)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=random_reward_function,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ProgressCallback(logger, total_steps)]
    )

    print(f"\n🎯 Starting optimized GRPO training...")
    print(f"📊 Using random rewards for pipeline validation")
    print(f"📈 Training metrics will be logged in real-time...")
    print(f"⚡ Optimized for RTX 4090 performance")

    # Start training
    trainer.train()

    # Print final summary
    logger.print_training_summary()

    # Create comprehensive plots
    print(f"\n📊 Generating optimized training metrics visualization...")
    logger.create_comprehensive_plots("./grpo_training_metrics_optimized.png")

    print(f"\n🎉 Optimized GRPO training completed successfully!")
    print(f"📁 Model and logs saved to: {OUTPUT_DIR}")
    print(f"🚀 Ready for environment reward integration!")

if __name__ == "__main__":
    main()