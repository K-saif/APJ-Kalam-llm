import torch
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# Load context JSON
with open("wings-of-fire.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# Extract all context strings
full_context = "\n".join([entry["text"] for entry in data if "text" in entry])

# Unique paragraph extractor
def extract_unique_paragraphs(text, threshold=0.85):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    unique = []
    vectorizer = TfidfVectorizer().fit(paragraphs)
    tfidf_matrix = vectorizer.transform(paragraphs)

    for i, para in enumerate(paragraphs):
        sim = (tfidf_matrix[i] @ tfidf_matrix[:i].T).toarray()
        if sim.shape[1] == 0 or sim.max() < threshold:
            unique.append(para)
    return unique

# Login to Hugging Face Hub
login(token="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", add_to_git_credential=False)

# # -----------------------------
# # Load saved context and embeddings
# # -----------------------------
# with open("context_paragraphs.pkl", "rb") as f:
#     context_paragraphs = pickle.load(f)

# context_embeddings = np.load("context_embeddings.npy")

# # -----------------------------
# # Set up retriever (Nearest Neighbors)
# # -----------------------------
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# nn = NearestNeighbors(n_neighbors=3, metric="cosine")
# nn.fit(context_embeddings)

# -----------------------------
# Load the LLaMA 2 Chat model
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model_id = "meta-llama/Llama-2-7b-chat-hf"
model_id = "meta-llama/Meta-Llama-3-8B"  # or another variant
# model_id = "/home/oem/Music/temp1/outputs/checkpoint-650"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# -----------------------------
# Prompt Template
# -----------------------------
def build_prompt(paragraph):
    return f"""
    You are helping generate question-answer pairs to fine-tune a model that speaks in the voice of Dr. APJ Abdul Kalam.
    Given a paragraph from his book "My Journey", create 10 question-answer pairs. Each question should be either:
    - a meaningful interview-style question OR
    - a natural question someone curious might ask in conversation.
    Each answer should reflect Dr. Kalam’s original thoughts, wisdom, and tone based on the paragraph.
    Avoid copying exact lines from the paragraph — instead, paraphrase in Kalam's reflective style. Keep answers informative, inspirational, and in first-person as if Dr. Kalam is speaking.
    Paragraph: {paragraph}

"""

# -----------------------------
# Inference Function
# -----------------------------
def generate_answer(paragraph):
    prompt = build_prompt(paragraph)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.replace(prompt.strip(), "").strip()

# -----------------------------
# Main Process
# -----------------------------
if __name__ == "__main__":
    qa_pairs = []
    unique_paragraphs = extract_unique_paragraphs(full_context)
    for idx, para in enumerate(unique_paragraphs):
        print(f"\n📖 Processing Paragraph {idx+1}/{len(unique_paragraphs)}")
        result = generate_answer(para)
        print("📚 Generated QA Pairs:\n", result)
        qa_pairs.append({"context": para, "result": result})

    # Save to a JSON file
    output_file = "generated_qa_results_llama.json"
    with open(output_file, "w") as f:
        json.dump(qa_pairs, f, indent=4)

    print(f"\n✅ Saved {len(qa_pairs)} QA pairs to {output_file}")
