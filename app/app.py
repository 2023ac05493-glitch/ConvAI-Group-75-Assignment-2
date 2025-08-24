import os
import time
import streamlit as st
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import json
import re
from nltk.corpus import stopwords
import torch.nn.functional as F

# from fine_tuning import generate_with_confidence 
# === Build absolute paths =====
BASE_DIR   = os.path.dirname(__file__)
FT_PATH    = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "fine_tuned_model"))
RAG_DIR    = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "rag_model"))

# === Load Fine-tuned model ===
ft_tokenizer = AutoTokenizer.from_pretrained(FT_PATH, local_files_only=True)
ft_model     = AutoModelForCausalLM.from_pretrained(FT_PATH, local_files_only=True)

# === Load RAG components ===
# Load chunk texts
with open(os.path.join(RAG_DIR,"chunks.json"), "r") as f:
    chunks_list = json.load(f)
# Load FAISS
faiss_index = faiss.read_index(os.path.join(RAG_DIR, "faiss_index.bin"))


# Load embedding model
# embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
 
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=device   # force CPU or GPU
)
 
 

# Fix meta tensors to avoid NotImplementedError during .to(device)
import torch
for name, param in embedder.named_parameters():
    if param.device.type == 'meta':
        param.data = torch.empty_like(param)

for name, buffer in embedder.named_buffers():
    if buffer.device.type == 'meta':
        buffer.data = torch.empty_like(buffer)


# Build BM25
bm25_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenized_chunks = [bm25_tokenizer.tokenize(c.lower()) for c in chunks_list]
normalized = [[t.lstrip("Ġ") for t in toks] for toks in tokenized_chunks]
bm25_model = BM25Okapi(normalized)

def generate_with_confidence(prompt, model, tokenizer, max_new_tokens=50):
    model.eval()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    
    # Generate output with scores
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract generated tokens (excluding the prompt)
    generated_tokens = outputs.sequences[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Get token-level probabilities
    scores = outputs.scores  # List of logits per token
    token_confidences = []
    
    for i, (score, token_id) in enumerate(zip(scores, generated_tokens)):
        try:
            # Get the probability distribution for this token position
            probs = F.softmax(score, dim=-1)
            # Get the probability of the generated token
            token_prob = probs[0, token_id].item()
            token_confidences.append(token_prob)
        except IndexError:
            # If there's an index error, skip this token
            token_confidences.append(0.0)
            continue

    avg_confidence = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0

    return {
        "generated_text": generated_text.strip(),
        "token_confidences": token_confidences,
        "avg_confidence": avg_confidence
    }

# Hybrid functions
def preprocess_query(q):
    sw = set(stopwords.words("english"))
    q = re.sub(r"[^a-zA-Z0-9\s]","", q.lower())
    toks = [t.lstrip("Ġ") for t in bm25_tokenizer.tokenize(q)]
    toks = [t for t in toks if t not in sw and len(t) > 2]
    return " ".join(toks)

def dense_ret(q, top_k=5):
    emb = embedder.encode([q]).astype(np.float32)
    dist, idxs = faiss_index.search(emb, top_k)
    return [(chunks_list[i], 1/(1+dist[0][j])) for j,i in enumerate(idxs[0])]

def sparse_ret(q, top_k=5):
    toks = [t.lstrip("Ġ") for t in bm25_tokenizer.tokenize(q)]
    scores = bm25_model.get_scores(toks)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(chunks_list[i], scores[i]) for i in top_idx]

def hybrid_retrieve(query, top_k=5, alpha=0.7):
    q = preprocess_query(query)
    d = dense_ret(q, top_k)
    s = sparse_ret(q, top_k)
    combined = {}
    # normalize
    if d:
        m = max([sc for _,sc in d])
        d = [(t,sc/m) for t,sc in d]
    if s:
        m = max([sc for _,sc in s])
        s = [(t,sc/m) for t,sc in s]
    for t,sc in d:
        combined[t] = alpha*sc
    for t,sc in s:
        combined[t] = combined.get(t,0) + (1-alpha)*sc
    top = sorted(combined.items(), key=lambda x:x[1], reverse=True)[:top_k]
    # return list of (text, score)
    return top

def answer_with_rag(question):

    retrieved = hybrid_retrieve(question, top_k=5)

    context = " ".join([t for t, _ in retrieved])
    prompt = f"Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"

    inputs = ft_tokenizer.encode(prompt, return_tensors="pt")
    out = ft_model.generate(inputs, max_new_tokens=80)
    raw_text = ft_tokenizer.decode(out[0], skip_special_tokens=True)

    # Clean the output to extract only the final answer line
    if "Answer:" in raw_text:
        answer = raw_text.split("Answer:")[-1].split('\n')[0].strip()
    else:
        answer = raw_text.strip()

    confidence = sum([score for _, score in retrieved]) / len(retrieved) if retrieved else 0.0
    return answer, confidence

def answer_with_finetuned(question):
    # Use your existing function that returns answer + avg confidence
    result = generate_with_confidence(question, ft_model, ft_tokenizer)
    return result["generated_text"], result["avg_confidence"]

 
# =============== Streamlit App ==================
st.title("Financial QA Assistant")

mode = st.selectbox("Select Method", ["Fine-Tuned","RAG"])
question = st.text_input("Ask a financial question")

if st.button("Generate Answer") and len(question) > 0:
    start = time.time()
    if mode == "Fine-Tuned":
        answer, confidence = answer_with_finetuned(question)
    else:
        answer, confidence = answer_with_rag(question)
    elapsed = time.time() - start

    st.markdown(f"**Answer ({mode}):**  {answer}")
    st.markdown(f"**Confidence Score:**  {confidence:.3f}")
    st.markdown(f"**Response Time:**  {elapsed:.2f}s")
    


