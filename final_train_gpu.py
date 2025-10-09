#!/usr/bin/env python3
"""
GPU-Optimized AgentGym Multi-Environment Training Script
This version is optimized for cloud GPU training with BF16/FP16 support
"""

import torch
import json
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import os

def load_and_format_dataset():
    """Load and format the AgentGym dataset"""
    print("📥 Loading AgentGym dataset...")

    if not os.path.exists("agentgym_data/multi_env_train.json"):
        print("❌ Dataset not found. Please run 'uv run python prepare_agentgym_data.py' first.")
        return None, None

    with open("agentgym_data/multi_env_train.json", 'r') as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} samples from AgentGym")

    # Environment distribution
    env_counts = {}
    for item in data:
        env = item.get('environment', 'unknown')
        env_counts[env] = env_counts.get(env, 0) + 1

    print("📊 Environment distribution:")
    for env, count in sorted(env_counts.items()):
        percentage = (count / len(data)) * 100
        print(f"   - {env:10s}: {count:4d} samples ({percentage:5.1f}%)")

    # Format data for training
    def format_text(sample):
        """Format sample as instruction-following text"""
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')

        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    texts = [format_text(item) for item in data if item.get('instruction') and item.get('output')]
    print(f"✅ Formatted {len(texts)} training examples")

    return data, texts

def create_training_config(args):
    """Create training configuration"""
    return {
        'batch_size': args.batch_size,
        'grad_accum': args.gradient_accumulation,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'max_seq_length': args.max_seq_length,
        'warmup_steps': args.warmup_steps,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
        'logging_steps': args.logging_steps,
    }

def main():
    parser = argparse.ArgumentParser(description='AgentGym Multi-Environment Training')

    # Model configuration
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-0.6B',
                       help='Model to train (default: Qwen/Qwen3-0.6B)')
    parser.add_argument('--model-name', type=str, default='qwen3-0.6b',
                       help='Model name for saving (default: qwen3-0.6b)')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size per device (default: 4 for GPU)')
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                       help='Gradient accumulation steps (default: 2 for GPU)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate (default: 2e-5)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length (default: 2048 for GPU)')
    parser.add_argument('--warmup-steps', type=int, default=100,
                       help='Warmup steps (default: 100)')
    parser.add_argument('--save-steps', type=int, default=100,
                       help='Save checkpoint every N steps (default: 100)')
    parser.add_argument('--eval-steps', type=int, default=100,
                       help='Evaluate every N steps (default: 100)')
    parser.add_argument('--logging-steps', type=int, default=10,
                       help='Log every N steps (default: 10)')

    # Dataset options
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use (default: all)')

    args = parser.parse_args()

    print("🚀 AgentGym Multi-Environment GPU Training")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Model Name: {args.model_name}")

    config = create_training_config(args)
    print(f"Training Config:")
    print(f"  Batch Size: {config['batch_size']} x {config['grad_accum']} = {config['batch_size'] * config['grad_accum']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Max Sequence Length: {config['max_seq_length']}")

    # Load and format dataset
    data, texts = load_and_format_dataset()
    if data is None:
        return

    # Limit samples if specified
    if args.max_samples:
        texts = texts[:args.max_samples]
        print(f"📝 Using {len(texts)} samples (limited by --max-samples)")

    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with GPU optimization
    print("📥 Loading model...")
    try:
        # Try BF16 first for modern GPUs
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            precision = "BF16"
            print("✅ Model loaded with BF16 precision")
        except Exception:
            # Fallback to FP16
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            precision = "FP16"
            print("✅ Model loaded with FP16 precision")

        model.gradient_checkpointing_enable()
        print(f"✅ Model loaded with {precision} precision and gradient checkpointing")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Tokenize data
    print("📝 Tokenizing data...")
    def tokenize_function(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=config['max_seq_length'],
            return_tensors="pt"
        )

    encodings = tokenize_function(texts)

    # Split train/eval
    split_idx = int(0.9 * len(texts))
    train_encodings = {k: v[:split_idx] for k, v in encodings.items()}
    eval_encodings = {k: v[split_idx:] for k, v in encodings.items()}

    print(f"   Train: {len(train_encodings['input_ids'])} samples")
    print(f"   Eval:  {len(eval_encodings['input_ids'])} samples")

    # Create dataset class
    class AgentGymDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings["input_ids"]
            self.attention_mask = encodings["attention_mask"]

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.input_ids[idx].clone()
            }

    train_dataset = AgentGymDataset(train_encodings)
    eval_dataset = AgentGymDataset(eval_encodings)

    # Training arguments with GPU optimization
    output_dir = f"./agentgym-{args.model_name}-output"

    # Set precision based on model dtype
    if precision == "BF16":
        precision_args = {"bf16": True}
    else:  # FP16
        precision_args = {"fp16": True}

    print(f"🎯 Using {precision} precision for training")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['grad_accum'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        dataloader_pin_memory=True,  # GPU optimization
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        **precision_args
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print(f"\n🎯 Starting training...")
    print(f"   Output directory: {output_dir}")
    print(f"   Training on {len(train_dataset)} samples")
    print(f"   Validating on {len(eval_dataset)} samples")
    print(f"   Precision: {precision}")

    # Start training
    trainer.train()

    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Training completed! Model saved to {output_dir}")

    # Test model
    print("\n🧪 Testing trained model...")
    test_prompts = [
        ("Go to the red ball", ""),
        ("Pick up the blue key", ""),
        ("Open the door", "")
    ]

    model.eval()
    for i, (instruction, input_text) in enumerate(test_prompts[:2]):  # Test only first 2
        if input_text:
            test_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            test_text = f"### Instruction:\n{instruction}\n\n### Response:\n"

        inputs = tokenizer(test_text, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nTest {i+1}: {instruction}")
        print(f"Response: {response}")

    print(f"\n🎉 Training complete!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"🧪 Model is ready for evaluation on AgentGym environments!")

if __name__ == "__main__":
    main()