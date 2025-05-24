import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
try:
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    # Set pad token to EOS token explicitly
    tokenizer.pad_token = tokenizer.eos_token
    print("Loaded GPT-2 model and tokenizer")
except Exception as e:
    print(f"Error loading GPT-2: {e}")
    print("Ensure 'transformers' is installed: pip install transformers")
    sys.exit(1)

# Example prompts
prompts = [
    "It is a truth universally acknowledged",
    "In a quiet village, there lived",
    "Love and friendship are"
]

# Generate text for each prompt
try:
    with open("generated_text_gpt2.txt", "w", encoding="utf-8") as f:
        for prompt in prompts:
            # Encode with attention mask and padding
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=50,
                return_attention_mask=True
            ).to(device)
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=200,
                do_sample=True,
                temperature=0.6,  # Lower for coherence
                top_k=40,
                top_p=0.95,
                no_repeat_ngram_size=2,  # Prevent repetitive phrases
                pad_token_id=tokenizer.eos_token_id
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generated_text}")
            f.write(f"Prompt: {prompt}\nGenerated: {generated_text}\n\n")
    print("Saved generated text to generated_text_gpt2.txt")
except Exception as e:
    print(f"Error during generation: {e}")
