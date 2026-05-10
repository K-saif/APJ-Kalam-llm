# APJ Abdul Kalam - LLM Fine-tuning

This project focuses on fine-tuning Large Language Models (LLMs) to specialize in content related to **APJ Abdul Kalam**, including his inspirational teachings, personal experiences, and wisdom. The repository implements both **Continued Pre-Training (CPT)** and **Supervised Fine-Tuning (SFT)** approaches using the Unsloth library for efficient training on limited hardware.

## 📋 Project Overview

The goal is to create a specialized LLM that can:
- Generate responses about APJ Abdul Kalam's life, lessons, and philosophy
- Answer questions based on his famous speeches and writings
- Provide inspirational insights from his perspective
- Maintain context through conversation

## 🚀 Quick Start

### Get Pre-trained Model

The easiest way to use this project is to download the pre-trained model from Hugging Face:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
model = AutoModelForCausalLM.from_pretrained("K-saif/apj-kalam-instruct")
tokenizer = AutoTokenizer.from_pretrained("K-saif/apj-kalam-instruct")

# Generate responses
inputs = tokenizer("Who was APJ Abdul Kalam?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

**Resources:**
- 🤗 **Model**: [K-saif/apj-kalam-instruct](https://huggingface.co/K-saif/apj-kalam-instruct)
- 📊 **Dataset**: [K-saif/apj-kalam-instruct-dataset](https://huggingface.co/datasets/K-saif/apj-kalam-instruct-dataset)

### Train Your Own Model

#### Prerequisites

```bash
pip install unsloth torch transformers datasets trl peft
```


#### Training Steps

**1. Continued Pre-Training (CPT)**
```bash
python CPT_unsloth_train.py
```
Trains the model on raw Kalam-related text to specialize the base model.

**2. Supervised Fine-Tuning (SFT)**
```bash
python SFT_fine_tune.py
```
Fine-tunes the CPT model on chat-format Q&A pairs.


**3. Push to Hugging Face (Optional)**
```bash
python push_to_huggingface.py
```

## 📊 Data Format

### CPT Data Format (`.jsonl`)
```json
{"text": "Full paragraph of APJ Abdul Kalam content..."}
```

### SFT Data Format (`.jsonl`)
```json
{
  "messages": [
    {"role": "user", "content": "Question about APJ Abdul Kalam"},
    {"role": "assistant", "content": "Answer from Kalam's perspective"}
  ]
}
```

### Q&A Pairs Format (`.json`)
```json
[
    {"question": "...", "answer": "..."}
]
```

## 🗂️ Repository Structure

### Core Training Scripts

- **`CPT_unsloth_train.py`** - Main CPT (Continued Pre-Training) script using Qwen 2.5 model with LoRA adapters
- **`SFT_fine_tune.py`** - Supervised Fine-Tuning script for chat-format instruction data
- **`push_to_huggingface.py`** - Utility to push trained models to Hugging Face Hub

### Datasets (`Datasets/`)

Contains diverse training data organized into categories:

- **`kalam_cpt.jsonl`** - Raw text data for CPT pre-training (645 paragraphs)
- **`kalam_qa_pairs.json`** - Question-Answer pairs dataset
- **`kalam_sft_final_fixed.jsonl`** - Finalized SFT data

### Utilities (`utils/`)

Helper scripts for data processing:

- **`generate_qa.py`** - Generate Q&A pairs from raw text
- **`qa_to_chatML.py`** - Convert Q&A to ChatML format
- **`txt_to_jsonl.py`** - Convert text files to JSONL format


## 🛠️ Configuration

Key parameters in training scripts:

- **`max_seq_length`**: 1024 tokens (adjust based on available VRAM)
- **`load_in_4bit`**: Quantization for memory efficiency
- **`r` (LoRA rank)**: 16 (increase for more capacity)
- **`lora_alpha`**: 16 (scaling factor)
- **`per_device_train_batch_size`**: Start with 1 for 8GB VRAM

## 🔗 References

- **Unsloth Documentation**: https://github.com/unslothai/unsloth
- **Hugging Face Hub**: https://huggingface.co

**About APJ Abdul Kalam:**
- Former President of India (1931-2015)
- Renowned scientist and author
- Known as the "Missile Man of India"


## Future Updates

### Planned Features

- **Multi-language Support**: Expand model to handle content in Hindi, Tamil, Telugu, and other Indian languages
- **Enhanced Q&A Generation**: Implement advanced techniques for automatic Q&A pair generation from raw text
- **Web API**: Deploy model as REST API for easy integration with web applications
- **Evaluation Metrics & Pipeline**: Add comprehensive benchmarking against standard LLM evaluation datasets

### Model Improvements

- **Larger Model Variants**: Train versions using Llama3, Gemma4, and other base models
- **Context Window Expansion**: Extend max sequence length for longer conversations

### Dataset Enhancements

- **Data Augmentation**: Implement synthetic data generation techniques
- **Quality Filtering**: Implement advanced filtering for data quality assurance

### Infrastructure & Deployment

- **Docker Support**: Add containerization for easy deployment
- **Model Quantization**: Implement INT8/FP8 quantization for faster inference
- **Monitoring & Analytics**: Track model performance metrics and user interactions


## 🤝 Contributing

To add more Kalam-related content:

1. Add raw text to appropriate `Datasets/` subfolder
2. Convert to JSONL format using `utils/txt_to_jsonl.py`
3. Extract Q&A pairs using the extractor agent
4. Re-train or fine-tune models as needed
