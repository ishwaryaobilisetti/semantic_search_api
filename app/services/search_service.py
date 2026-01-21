import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class SearchService:
    _instance = None
    
    def __init__(self):
        self.model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.data_path = os.getenv("DATA_PATH", "data/documents.json")
        self.index_path = os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
        self.model_path = os.getenv("MODEL_PATH", "models/sentence_transformer")
        
        self.model = None
        self.index = None
        self.documents_list = []
        
        self._load_resources()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_resources(self):
        print(f"Loading model...")
        try:
             self.model = SentenceTransformer(self.model_path)
        except:
             self.model = SentenceTransformer(self.model_name)

        print(f"Loading documents...")
        if not os.path.exists(self.data_path):
             raise FileNotFoundError(f"Documents file not found at {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            self.documents_list = json.load(f)

        print(f"Loading index...")
        if not os.path.exists(self.index_path):
             raise FileNotFoundError(f"Index not found at {self.index_path}")
        
        self.index = faiss.read_index(self.index_path)

    def search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        query_vector = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents_list):
                doc = self.documents_list[idx]
                results.append({
                    "id": doc["id"],
                    "text_snippet": doc["text"][:200], # Snippet
                    "score": float(distances[0][i])
                })
        return results

def get_search_service():
    return SearchService.get_instance()
