import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ========================= UPDATED CONFIG =========================
model_name = "unsloth/Qwen2.5-1.5B-bnb-4bit"

max_seq_length = 1024          # Safe for 8GB
dtype = None
load_in_4bit = True

# Load base model + previous CPT LoRA (if you want to continue from previous training)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="kalam_cpt_lora",      # ← Load your previous CPT adapter here
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Apply / update LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    modules_to_save=["embed_tokens", "lm_head"],   # ← Recommended now (helps style)
)

# Load new dataset (650 paragraphs)
dataset = load_dataset("json", data_files="kalam_cpt.jsonl", split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,      # effective batch ~16
        warmup_steps=20,
        num_train_epochs=3,                  # ← Changed: 3 epochs instead of 300 steps
        # max_steps=300,                     # Comment this out when using epochs
        learning_rate=1e-4,                  # Slightly lower than before
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="kalam_cpt_output_v2",
        optim="adamw_8bit",
        weight_decay=0.02,                   # Slightly higher to reduce overfitting
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
    ),
)

print("Starting Continued Pre-Training on 650 paragraphs...")
trainer.train()

model.save_pretrained("kalam_cpt_lora_v2")
tokenizer.save_pretrained("kalam_cpt_lora_v2")