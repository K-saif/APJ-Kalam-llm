import torch
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login



import json
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load context JSON
with open("qa_context_data.json", "r") as f:
    data = json.load(f)

# Extract all context strings
full_context = "\n".join([entry["context"] for entry in data if "context" in entry])

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

# -----------------------------
# Step 1: Load saved context and embeddings
# -----------------------------
with open("context_paragraphs.pkl", "rb") as f:
    context_paragraphs = pickle.load(f)

context_embeddings = np.load("context_embeddings.npy")

# -----------------------------
# Step 2: Set up retriever (Nearest Neighbors)
# -----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
nn = NearestNeighbors(n_neighbors=3, metric="cosine")
nn.fit(context_embeddings)

# -----------------------------
# Step 3: Load the Gemma 2B IT model with 4-bit quantization
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# -----------------------------
# Step 4: Inference function
# -----------------------------
def generate_answer(paragraph):
    prompt = f"""
    <start_of_turn>user
    consider You are Dr. A.P.J. Abdul Kalam, former President of India. Only based on the paragraph below, generate 10 thoughtful question-answer pairs by considering someone asking question to you and you are answering as Dr kalam. Always answer in a calm, wise, and encouraging tone — just as Dr. Kalam would.




    Paragraph:
    \"\"\"{paragraph}\"\"\"

    <end_of_turn>
    <start_of_turn>model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.convert_tokens_to_ids("<end_of_turn>")
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.replace(prompt, "").strip()
# consider You are Dr. A.P.J. Abdul Kalam, former President of India. Based on the paragraph below, generate 10 thoughtful question-answer pairs by considering someone asking question to you and you are answering as Dr kalam. Your answers should reflect your values: humility, scientific spirit, faith, simplicity, and the ability to inspire youth. Always answer in a calm, wise, and encouraging tone — just as Dr. Kalam would.


'''
You are helping generate question-answer pairs to fine-tune a model that speaks in the voice of Dr. APJ Abdul Kalam.

Given a paragraph from his book "My Journey", create 3–5 question-answer pairs. Each question should be either:
- a meaningful interview-style question OR
- a natural question someone curious might ask in conversation.

Each answer should reflect Dr. Kalam’s original thoughts, wisdom, and tone based on the paragraph.

Avoid copying exact lines from the paragraph — instead, paraphrase in Kalam's reflective style. Keep answers informative, inspirational, and in first-person as if Dr. Kalam is speaking.

Paragraph:
[INSERT PARAGRAPH HERE]

Output Format:
Q1:
A1:
Q2:
A2:
Q3:
A3:
...'''
    

# -----------------------------
# Step 5: Interactive mode
# -----------------------------
if __name__ == "__main__":
    # paragraph = """For as far back as I can remember, my father Jainulabdeen's day began at 4 a.m. He would be up before anyone else in the household. After saying his prayers in the breaking light of the day, he would go on a long walk to visit his coconut grove. We lived in Rameswaram, a small temple town on an island in Tamil Nadu. This being on the east coast of India, dawn would break early, and our day's schedule followed the rhythm of the rising and setting of the sun and the sea waves. The sound of the sea was a constant presence in our lives. Storms and cyclones blew by with regularity during the tumultuous monsoon months. We lived in our ancestral home, a fairly large house made of limestone and brick, built some time in the nineteenth century. It was never luxurious, but was filled with love. My father had a boat-building business. Additionally, we also owned a small coconut grove some four miles away from our house. That was where my father would be headed for in the early morning hours. His walking circuit was well established and he rarely deviated from it. First he would step out into Mosque Street, where our house was located. It was a small, predominantly Muslim locality not too far from the Shiva temple that has made our town famous for centuries. He would then walk through the narrow lanes of the town, on to the more open roads leading to the coconut groves, and finally wind his way through the groves to his patch of land."""
    # while True:
    qa_pairs= []
    unique_paragraphs = extract_unique_paragraphs(full_context)
    for para in unique_paragraphs:
        result = generate_answer(para)
        print("📚 Generated QA Pairs:\n", result)
        qa_pairs.append({"context":para,"result":result})


    # Save to a JSON file
    output_file = "generated_qa_results2.json"
    with open(output_file, "w") as f:
        json.dump(qa_pairs, f, indent=4)

    print(f"\n✅ Saved {len(qa_pairs)} QA pairs to {output_file}")
