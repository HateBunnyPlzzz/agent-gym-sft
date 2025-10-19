#!/usr/bin/env python3
"""
Clean minimal inference testing script for trained AgentGym model
Tests the fine-tuned Qwen3-4B model on AgentGym-style prompts
"""

import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_trained_model(model_path, base_model="Qwen/Qwen3-4B"):
    """Load the fine-tuned model with LoRA weights"""
    import os

    print(f"🔧 DETAILED MODEL LOADING PROCESS")
    print("=" * 50)

    # Check if model path exists
    print(f"📁 Checking model path: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model path does NOT exist: {model_path}")
        raise FileNotFoundError(f"Model path not found: {model_path}")
    print(f"✅ Model path exists!")

    # List files in model directory
    print(f"\n📋 Files in trained model directory:")
    for file in os.listdir(model_path):
        file_path = os.path.join(model_path, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   📄 {file} ({size_mb:.1f} MB)")
        else:
            print(f"   📁 {file}/ (directory)")

    # Check for required LoRA files
    required_files = ["adapter_model.safetensors", "adapter_config.json"]
    missing_files = []
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ Found required file: {file}")
        else:
            print(f"❌ Missing required file: {file}")
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(f"Missing required LoRA files: {missing_files}")

    print(f"\n🔧 Step 1: Loading tokenizer from base model: {base_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        print(f"✅ Tokenizer loaded successfully!")
        print(f"   - Vocabulary size: {tokenizer.vocab_size}")
        print(f"   - Model max length: {tokenizer.model_max_length}")
        print(f"   - Pad token: {tokenizer.pad_token}")
        print(f"   - EOS token: {tokenizer.eos_token}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"⚠️  Set pad_token to eos_token: {tokenizer.eos_token}")

    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        raise

    print(f"\n🔧 Step 2: Loading base model: {base_model}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"✅ Base model loaded successfully!")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Device map: {model.hf_device_map}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")

    except Exception as e:
        print(f"❌ Failed to load base model: {e}")
        raise

    print(f"\n🔧 Step 3: Loading LoRA weights from: {model_path}")
    try:
        # Check adapter config
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            print(f"📋 LoRA Configuration:")
            print(f"   - Rank (r): {adapter_config.get('r', 'N/A')}")
            print(f"   - Alpha: {adapter_config.get('lora_alpha', 'N/A')}")
            print(f"   - Target modules: {adapter_config.get('target_modules', 'N/A')}")
            print(f"   - Task type: {adapter_config.get('task_type', 'N/A')}")

        model = PeftModel.from_pretrained(model, model_path)
        print(f"✅ LoRA weights loaded successfully!")

        # Verify LoRA is applied
        if hasattr(model, 'peft_config'):
            print(f"📋 Active LoRA adapters: {list(model.peft_config.keys())}")

        model.eval()
        print(f"✅ Model set to evaluation mode!")

    except Exception as e:
        print(f"❌ Failed to load LoRA weights: {e}")
        raise

    print(f"\n🎯 MODEL LOADING SUMMARY:")
    print(f"   - Base model: {base_model}")
    print(f"   - LoRA weights: {model_path}")
    print(f"   - Model is ready for inference!")

    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=1024, temperature=0.7):
    """Generate response from the model with detailed logging"""
    print(f"🎯 GENERATION PROCESS")
    print("-" * 30)

    # Format using Qwen3 chat template
    messages = [
        {"role": "user", "content": prompt}
    ]

    print(f"📝 Input prompt: {prompt}")
    print(f"🌡️  Temperature: {temperature}")
    print(f"📏 Max length: {max_length}")

    # Apply chat template
    print(f"\n🔧 Step 1: Applying chat template...")
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"✅ Chat template applied!")
        print(f"📄 Full formatted text (first 200 chars): {text[:200]}...")

    except Exception as e:
        print(f"❌ Failed to apply chat template: {e}")
        raise

    # Tokenize
    print(f"\n🔧 Step 2: Tokenizing input...")
    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        print(f"✅ Tokenization successful!")
        print(f"   - Input shape: {input_ids.shape}")
        print(f"   - Device: {input_ids.device}")
        print(f"   - Input tokens (first 10): {input_ids[0][:10].tolist()}")
        print(f"   - Input text length: {len(input_ids[0])} tokens")

    except Exception as e:
        print(f"❌ Failed to tokenize: {e}")
        raise

    # Generate
    print(f"\n🔧 Step 3: Generating response...")
    print(f"   - Generation parameters:")
    print(f"     * max_length: {max_length}")
    print(f"     * temperature: {temperature}")
    print(f"     * do_sample: True")
    print(f"     * pad_token_id: {tokenizer.eos_token_id}")
    print(f"     * eos_token_id: {tokenizer.eos_token_id}")

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        print(f"✅ Generation completed!")
        print(f"   - Output shape: {outputs.shape}")
        print(f"   - New tokens generated: {outputs.shape[1] - input_ids.shape[1]}")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise

    # Decode response
    print(f"\n🔧 Step 4: Decoding response...")
    try:
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Decoding successful!")
        print(f"📄 Full response (first 300 chars): {full_response[:300]}...")

        # Extract only the assistant's response
        if "assistant" in full_response:
            assistant_response = full_response.split("assistant")[-1].strip()
            print(f"✅ Extracted assistant response")
        else:
            assistant_response = full_response.strip()
            print(f"⚠️  No 'assistant' tag found, using full response")

        print(f"📤 Final response length: {len(assistant_response)} characters")
        print(f"📤 Final response (first 200 chars): {assistant_response[:200]}...")

        return assistant_response

    except Exception as e:
        print(f"❌ Decoding failed: {e}")
        raise

def test_agentgym_samples():
    """Test with typical AgentGym-style prompts"""

    test_samples = [
        {
            "environment": "AlfWorld",
            "prompt": "I need to put a clean apple in the microwave. The kitchen is messy with items scattered around. What should I do first?",
            "expected_keywords": ["find", "apple", "clean", "microwave"]
        },
        {
            "environment": "BabyAI",
            "prompt": "You are in a grid world. Go to the red key, pick it up, then go to the green door and open it. What's your plan?",
            "expected_keywords": ["key", "door", "navigate", "pickup"]
        },
        {
            "environment": "WebShop",
            "prompt": "I want to buy a wireless mouse under $30 with good reviews. Can you help me search for this?",
            "expected_keywords": ["search", "mouse", "price", "reviews"]
        },
        {
            "environment": "SciWorld",
            "prompt": "I need to test how different amounts of fertilizer affect plant growth. Design an experiment for me.",
            "expected_keywords": ["experiment", "fertilizer", "plants", "growth", "measure"]
        },
        {
            "environment": "TextCraft",
            "prompt": "Help me craft a simple wooden pickaxe in this text-based crafting game.",
            "expected_keywords": ["craft", "wood", "pickaxe", "materials"]
        }
    ]

    return test_samples

def main():
    parser = argparse.ArgumentParser(description='Test trained AgentGym model')
    parser.add_argument('--model-path', type=str,
                       default='./qwen3-agentgym-110samples-bf16',
                       help='Path to trained model directory')
    parser.add_argument('--base-model', type=str,
                       default='Qwen/Qwen3-4B',
                       help='Base model name')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum generation length')

    args = parser.parse_args()

    print("🚀 AgentGym Model Inference Testing")
    print("=" * 50)

    # Load model
    try:
        model, tokenizer = load_trained_model(args.model_path, args.base_model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Get test samples
    test_samples = test_agentgym_samples()

    print(f"\n🧪 Testing {len(test_samples)} AgentGym environments...")
    print(f"🎛️  Temperature: {args.temperature}")
    print(f"📏 Max Length: {args.max_length}")
    print()

    # Test each sample
    for i, sample in enumerate(test_samples, 1):
        print(f"🌍 Test {i}/{len(test_samples)}: {sample['environment']}")
        print("-" * 40)
        print(f"Prompt: {sample['prompt']}")
        print()

        try:
            # Generate response
            response = generate_response(
                model, tokenizer, sample['prompt'],
                max_length=args.max_length,
                temperature=args.temperature
            )

            print(f"Response: {response}")
            print()

            # Check for expected keywords
            found_keywords = [kw for kw in sample['expected_keywords']
                            if kw.lower() in response.lower()]

            print(f"Expected keywords found: {found_keywords}/{sample['expected_keywords']}")
            print(f"Coverage: {len(found_keywords)/len(sample['expected_keywords'])*100:.1f}%")
            print()

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            print()

        print("=" * 60)
        print()

    print("✅ Testing completed!")
    print("💡 Tips:")
    print("   - Lower temperature (0.1-0.3) for more focused responses")
    print("   - Higher temperature (0.8-1.0) for more creative responses")
    print("   - Adjust max_length based on task complexity")

if __name__ == "__main__":
    main()