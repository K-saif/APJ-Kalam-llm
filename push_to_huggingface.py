from unsloth import FastLanguageModel

# Load your final trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="C:/Users/khans/Documents/APJ-Abdul-Kalam/kalam_final_model",
    max_seq_length=1024,
    load_in_4bit=True,
)

# Push adapter + tokenizer to Hugging Face
model.push_to_hub("K-saif/apj-kalam-instruct")
tokenizer.push_to_hub("K-saif/apj-kalam-instruct")

print("✅ Model uploaded successfully!")