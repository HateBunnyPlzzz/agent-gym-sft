#!/usr/bin/env python3
"""
BabyAI training script with Unsloth optimization for Qwen3-0.6B on RTX 2000 Ada (16GB)
"""

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

def main():
    print("🚀 Starting BabyAI SFT training with Qwen3-0.6B + Unsloth...")

    # Configuration
    MODEL_NAME = "unsloth/Qwen3-0.6B"
    MAX_SAMPLES = 100
    MAX_SEQ_LENGTH = 2048  # Unsloth allows longer sequences
    BATCH_SIZE = 2  # Unsloth allows larger batch sizes
    GRAD_ACCUM = 4
    LEARNING_RATE = 2e-4
    EPOCHS = 3

    # Load model and tokenizer with Unsloth
    print(f"Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
    )

    print(f"Model loaded with 4-bit quantization")
    print(f"Original parameters: {model.num_parameters():,}")

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    print("LoRA adapters added")
    print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")

    # Load dataset
    print("Loading AgentTraj-L dataset...")
    babyai_data = []

    # Try to load from local file first
    import os
    if os.path.exists("babyai_train.json"):
        print("Loading from local babyai_train.json...")
        import json
        try:
            with open("babyai_train.json", 'r') as f:
                data = json.load(f)
                # Extract first MAX_SAMPLES examples
                babyai_data = data[:MAX_SAMPLES]
                print(f"Found {len(babyai_data)} BabyAI samples from local file")
        except Exception as e:
            print(f"Failed to load local file: {e}")

    # Fallback to HuggingFace dataset
    if not babyai_data:
        try:
            dataset = load_dataset("AgentGym/AgentTraj-L", split="train")
            for item in dataset:
                if item.get("environment") == "babyai":
                    babyai_data.append(item)
                    if len(babyai_data) >= MAX_SAMPLES:
                        break
            print(f"Found {len(babyai_data)} BabyAI samples from HuggingFace")
        except Exception as e:
            print(f"Failed to load dataset: {e}")

    # If still no data, create sample data
    if not babyai_data:
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
            },
            {
                "conversations": [
                    {"from": "human", "value": "Go to the green box"},
                    {"from": "gpt", "value": "I'll navigate to the green box by moving in that direction."}
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "Pick up the yellow ball"},
                    {"from": "gpt", "value": "I'll pick up the yellow ball by going to its location and grabbing it."}
                ]
            }
        ]
        print(f"Created {len(babyai_data)} sample BabyAI examples")

    # Format data for chat template
    def format_conversation(convs):
        formatted_text = ""
        for conv in convs:
            if conv["from"] == "human":
                formatted_text += f"<|im_start|>user\n{conv['value']}<|im_end|>\n"
            elif conv["from"] == "gpt":
                formatted_text += f"<|im_start|>assistant\n{conv['value']}<|im_end|>\n"
        return formatted_text.strip()

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

    # Training args optimized for Unsloth
    training_args = TrainingArguments(
        output_dir="./babyai_qwen3_unsloth_output",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
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
    print(f"  Model: {MODEL_NAME} + Unsloth + 4-bit + LoRA")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"  Memory optimization: 4-bit + 8-bit optimizer")

    # Start training
    print("\n🎯 Starting Unsloth-optimized training...")
    trainer.train()

    # Save model (LoRA adapters)
    trainer.save_model()
    tokenizer.save_pretrained("./babyai_qwen3_unsloth_output")
    print("✅ Training completed! LoRA adapters saved to ./babyai_qwen3_unsloth_output")

    # Save model for inference
    model.save_pretrained_merged("./babyai_qwen3_merged", tokenizer, save_method="merged_16bit")
    print("✅ Merged model saved to ./babyai_qwen3_merged")

    # Test model
    print("\n🧪 Testing trained model...")
    FastLanguageModel.for_inference(model)  # Enable 2x faster inference

    test_prompt = "<|im_start|>user\nGo to the red ball<|im_end|>\n<|im_start|>assistant\n"
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