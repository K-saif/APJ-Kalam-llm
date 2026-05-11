import os
import torch
from unsloth import FastLanguageModel

# =========================================================
# ENV SETTINGS
# =========================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Disable torch dynamo/compilation to avoid conflicts
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.disable()

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

print("Loading trained model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="kalam_gemma3_4b_final_model",
    max_seq_length=768,
    dtype=None,
    load_in_4bit=True,
)

# Set model to inference mode
FastLanguageModel.for_inference(model)

print("Model loaded successfully!")

# =========================================================
# INFERENCE FUNCTION
# =========================================================

def generate_response(user_input, system_prompt="", max_new_tokens=256, temperature=0.7, top_p=0.9):
    """
    Generate a response from the model
    
    Args:
        user_input: User's question/prompt
        system_prompt: Optional system prompt to guide behavior
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling parameter
    
    Returns:
        Generated response text
    """
    
    # Format the input using the same format as training
    prompt = ""
    
    if system_prompt:
        prompt += f"<|system|>\n{system_prompt}\n"
    
    prompt += f"<|user|>\n{user_input}\n<|assistant|>\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with inference mode
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    return response

# =========================================================
# INTERACTIVE TESTING
# =========================================================

def interactive_test():
    """Interactive chat interface"""
    print("\n" + "="*60)
    print("MODEL TESTING - INTERACTIVE MODE")
    print("="*60)
    print("Type your questions below. Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not user_input:
            continue
        
        print("\nGenerating response...")
        response = generate_response(user_input)
        print(f"Model: {response}\n")

# =========================================================
# BATCH TESTING
# =========================================================

def batch_test(test_prompts):
    """Test model with a list of prompts"""
    print("\n" + "="*60)
    print("MODEL TESTING - BATCH MODE")
    print("="*60 + "\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"[Test {i}]")
        print(f"Prompt: {prompt}")
        response = generate_response(prompt)
        print(f"Response: {response}")
        print("-" * 60)

# =========================================================
# EXAMPLE TEST CASES
# =========================================================

if __name__ == "__main__":
    # Example test prompts - modify these based on your model's training data
    test_prompts = [
        "Hello, how are you?",
        "What is machine learning?",
        "Explain artificial intelligence in simple terms",
    ]
    
    print("\nChoose testing mode:")
    print("1. Interactive (chat-like interface)")
    print("2. Batch (predefined prompts)")
    print("3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        interactive_test()
    
    if choice in ['2', '3']:
        batch_test(test_prompts)
