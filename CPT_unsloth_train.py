from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ========================= CONFIG =========================
model_name = "unsloth/Qwen2.5-3B-bnb-4bit"     # ← Change to 7B only if you have headroom

max_seq_length = 2048          # Keep low (2048 or even 1024) to save VRAM
dtype = None                   # Auto detection
load_in_4bit = True

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                          # LoRA rank - keep low
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",   # Very important for low VRAM
    random_state=3407,
    use_rslora=False,
)

# Optional: Train embed_tokens and lm_head too (helps style absorption)
# model = FastLanguageModel.get_peft_model(..., modules_to_save=["embed_tokens", "lm_head"])

# Load dataset
dataset = load_dataset("json", data_files="kalam_cpt.jsonl", split="train")

# Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,                    # Important for raw text
    args=TrainingArguments(
        per_device_train_batch_size=1,      # Must be 1 on 8GB
        gradient_accumulation_steps=16,     # Effective batch size = 16
        warmup_steps=10,
        max_steps=300,                      # Start small! (you can continue later)
        learning_rate=2e-4,                 # Good for CPT
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="kalam_cpt_output",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
    ),
)

print("Starting Continued Pre-Training...")
trainer.train()

# Save the LoRA adapter
model.save_pretrained("kalam_cpt_lora")
tokenizer.save_pretrained("kalam_cpt_lora")