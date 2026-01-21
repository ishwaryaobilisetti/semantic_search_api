import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configuration
DATA_DIR = "data"
MODELS_DIR = "models/sentence_transformer"
DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.json")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def generate_synthetic_data(num_docs=1000):
    """Generates synthetic documents for testing."""
    print(f"Generating {num_docs} synthetic documents...")
    documents = []
    base_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models require data for training.",
        "Deep learning is a subset of machine learning.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to see and interpret images.",
        "Reinforcement learning learns from trial and error.",
        "Supervised learning uses labeled datasets.",
        "Unsupervised learning finds patterns in unlabeled data.",
        "Python is a popular programming language for data science."
    ]
    
    for i in range(num_docs):
        base_text = base_texts[i % len(base_texts)]
        documents.append({
            "id": f"doc_{i}",
            "text": f"{base_text} - Variation {i}"
        })
    return documents

def generate_embeddings_and_index():
    # 1. Load Model
    print(f"Loading model...")
    if os.path.exists(MODELS_DIR) and len(os.listdir(MODELS_DIR)) > 0:
        print(f"Loading from local path: {MODELS_DIR}")
        model = SentenceTransformer(MODELS_DIR)
    else:
        print(f"Downloading from Hub: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        model.save(MODELS_DIR)

    # 2. Get Data
    if os.path.exists(DOCUMENTS_FILE):
        print(f"Loading documents from {DOCUMENTS_FILE}...")
        with open(DOCUMENTS_FILE, 'r') as f:
            documents = json.load(f)
    else:
        documents = generate_synthetic_data()
        print(f"Saving documents to {DOCUMENTS_FILE}...")
        with open(DOCUMENTS_FILE, 'w') as f:
            json.dump(documents, f, indent=2)

    # 3. Generate Embeddings
    texts = [doc["text"] for doc in documents]
    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # 4. Create Faiss Index
    dimension = embeddings.shape[1]
    print(f"Creating Faiss index with dimension: {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"Index contains {index.ntotal} vectors.")

    # 5. Save Index
    print(f"Saving Faiss index to {FAISS_INDEX_FILE}...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    print("Done.")

if __name__ == "__main__":
    generate_embeddings_and_index()
