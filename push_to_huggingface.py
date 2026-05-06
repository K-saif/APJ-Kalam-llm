from unsloth import FastLanguageModel
from datasets import load_dataset
from huggingface_hub import HfApi
import os

# =====================================================
# FLAGS
# =====================================================

UPLOAD_MODEL = False
UPLOAD_DATASET = True
UPLOAD_EXTRA_FILES = True

# =====================================================
# MODEL CONFIG
# =====================================================

MODEL_PATH = "C:/Users/khans/Documents/APJ-Abdul-Kalam/kalam_final_model"
MODEL_REPO = "K-saif/apj-kalam-instruct"

MAX_SEQ_LENGTH = 1024

# =====================================================
# DATASET CONFIG
# =====================================================

DATASET_FILE = "C:/Users/khans/Documents/APJ-Abdul-Kalam/Datasets/kalam_sft_final_fixed.jsonl"
DATASET_REPO = "K-saif/kalam-instruct-dataset"

# =====================================================
# EXTRA FILES TO UPLOAD
# =====================================================

EXTRA_FILES = [
    "C:/Users/khans/Documents/APJ-Abdul-Kalam/Datasets/kalam_cpt.jsonl",
]

# =====================================================
# HF API
# =====================================================

api = HfApi()

# =====================================================
# UPLOAD MODEL
# =====================================================

if UPLOAD_MODEL:

    print("Loading model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    print("Uploading model...")

    model.push_to_hub(MODEL_REPO)
    tokenizer.push_to_hub(MODEL_REPO)

    print(f"✅ Model uploaded -> {MODEL_REPO}")

# =====================================================
# UPLOAD DATASET
# =====================================================

if UPLOAD_DATASET:

    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files=DATASET_FILE,
        split="train"
    )

    print("Uploading dataset parquet...")

    dataset.push_to_hub(DATASET_REPO)

    print(f"✅ Dataset uploaded -> {DATASET_REPO}")

# =====================================================
# UPLOAD EXTRA RAW FILES
# =====================================================

if UPLOAD_EXTRA_FILES:

    print("Uploading extra raw files...")

    for file_path in EXTRA_FILES:

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue

        filename = os.path.basename(file_path)

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"raw_files/{filename}",
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )

        print(f"✅ Uploaded raw file: {filename}")

# =====================================================
# DONE
# =====================================================

print("🎉 All uploads completed!")