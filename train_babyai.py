#!/usr/bin/env python3
"""
BabyAI training script for Qwen3-0.6B on RTX 2000 Ada (16GB)
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

def main():
    print("🚀 Starting BabyAI SFT training with Qwen3-0.6B...")

    # Configuration
    MODEL_NAME = "Qwen/Qwen3-0.6B"
    MAX_SAMPLES = 100
    MAX_SEQ_LENGTH = 512
    BATCH_SIZE = 1
    GRAD_ACCUM = 8
    LEARNING_RATE = 2e-4
    EPOCHS = 3

    # Load model and tokenizer
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"Model loaded: {model.num_parameters():,} parameters")

    # Load dataset
    print("Loading AgentTraj-L dataset...")
    try:
        dataset = load_dataset("AgentGym/AgentTraj-L", split="train")
        babyai_data = []
        for item in dataset:
            if item.get("environment") == "babyai":
                babyai_data.append(item)
                if len(babyai_data) >= MAX_SAMPLES:
                    break
        print(f"Found {len(babyai_data)} BabyAI samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Creating sample BabyAI data...")
        babyai_data = [
            {
                "conversations": [
                    {"from": "human", "value": "Go to the red ball"},
                    {"from": "gpt", "value": "I need to navigate to the red ball. I'll move forward and turn to reach it."}
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Pick up the blue key"},
                    {"from": "gpt", "value": "I'll pick up the blue key by moving to its location and interacting with it."}
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Open the door"},
                    {"from": "gpt", "value": "I need to find the key first, then go to the door and use the key to open it."}
                ]
            }
        ]

    # Format data
    def format_conversation(convs):
        text = ""
        for conv in convs:
            if conv["from"] == "human":
                text += f"User: {conv['value']}\n"
            elif conv["from"] == "gpt":
                text += f"Assistant: {conv['value']}\n"
        return text.strip()

    training_texts = [format_conversation(item["conversations"]) for item in babyai_data]

    # Split train/eval
    split_idx = int(0.9 * len(training_texts))
    train_texts = training_texts[:split_idx]
    eval_texts = training_texts[split_idx:]
    print(f"Train: {len(train_texts)}, Eval: {len(eval_texts)}")

    # Tokenize
    def tokenize_function(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

    train_encodings = tokenize_function(train_texts)
    eval_encodings = tokenize_function(eval_texts)

    # Dataset class
    class BabyAIDataset(torch.utils.data.Dataset):
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

    train_dataset = BabyAIDataset(train_encodings)
    eval_dataset = BabyAIDataset(eval_encodings)

    # Training args
    training_args = TrainingArguments(
        output_dir="./babyai_qwen3_output",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
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
    tokenizer.save_pretrained("./babyai_qwen3_output")
    print("✅ Training completed! Model saved to ./babyai_qwen3_output")

    # Test model
    print("\n🧪 Testing trained model...")
    test_prompt = "User: Go to the red ball\nAssistant:"
    inputs = tokenizer(test_prompt, return_tensors="pt", padding=True).to(model.device)

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