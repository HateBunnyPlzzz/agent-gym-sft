#!/usr/bin/env python3
"""
Simple training script for AgentGym multi-environment SFT
Uses standard transformers without Axolotl dependencies
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import json
import os

def main():
    print("🚀 Starting AgentGym Multi-Environment Training with Standard Transformers...")

    # Configuration
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    BATCH_SIZE = 2
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    MAX_SEQ_LENGTH = 1024  # Reduced for memory efficiency

    print(f"Model: {MODEL_NAME}")
    print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max sequence length: {MAX_SEQ_LENGTH}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Load the prepared dataset
    print("Loading AgentGym dataset...")
    if os.path.exists("agentgym_data/multi_env_train.json"):
        print("Loading from local file...")
        with open("agentgym_data/multi_env_train.json", 'r') as f:
            data = json.load(f)
    else:
        print("❌ Dataset file not found. Run 'uv run python prepare_agentgym_data.py' first.")
        return

    print(f"Loaded {len(data)} samples from AgentGym dataset")

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
    print(f"Formatted {len(texts)} training examples")

    # Tokenize data
    def tokenize_function(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

    print("Tokenizing data...")
    encodings = tokenize_function(texts)

    # Split train/eval
    split_idx = int(0.9 * len(texts))
    train_encodings = {k: v[:split_idx] for k, v in encodings.items()}
    eval_encodings = {k: v[split_idx:] for k, v in encodings.items()}

    print(f"Train: {len(train_encodings['input_ids'])}, Eval: {len(eval_encodings['input_ids'])}")

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

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./agentgym-multi-env-output",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
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

    print("Training configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")

    # Start training
    print("\n🎯 Starting training...")
    trainer.train()

    # Save model
    trainer.save_model()
    tokenizer.save_pretrained("./agentgym-multi-env-output")
    print("✅ Training completed! Model saved to ./agentgym-multi-env-output")

    # Test model
    print("\n🧪 Testing trained model...")
    test_instruction = "Go to the red ball"
    test_input = "You are in a room with a red ball on the floor."

    test_text = f"### Instruction:\n{test_instruction}\n\n### Input:\n{test_input}\n\n### Response:\n"
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
    print(f"Test response:\n{response}")

if __name__ == "__main__":
    main()