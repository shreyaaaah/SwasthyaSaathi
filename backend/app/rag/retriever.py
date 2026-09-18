import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
from app.config import settings

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None

class FAISSRetriever:
    def __init__(self, vector_store_dir: str = None, model_name: str = None):
        self.vector_store_dir = vector_store_dir or settings.VECTOR_STORE_DIR
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.index = None
        self.metadata = []
        self.model = None
        self.fallback_embedder = None
        self._load_store()

    def _load_store(self):
        index_path = os.path.join(self.vector_store_dir, "index.faiss")
        metadata_path = os.path.join(self.vector_store_dir, "metadata.json")
        vectorizer_path = os.path.join(self.vector_store_dir, "vectorizer.pkl")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"Loading FAISS index from {index_path}...")
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            
            if os.path.exists(vectorizer_path):
                print("Loading saved local vectorizer fallback...")
                try:
                    with open(vectorizer_path, "rb") as f:
                        self.fallback_embedder = pickle.load(f)
                except Exception as e:
                    print(f"Error loading vectorizer.pkl: {e}")

            if self.fallback_embedder is None:
                try:
                    print(f"Loading embedding model {self.model_name}...")
                    self.model = SentenceTransformer(self.model_name)
                except Exception as e:
                    print(f"Could not load SentenceTransformer ({e}). Initializing TF-IDF embedder...")
                    from app.rag.ingest import FallbackEmbedder
                    self.fallback_embedder = FallbackEmbedder(dim=self.index.d)
                    all_texts = [m.get("chunk_text", "") for m in self.metadata]
                    self.fallback_embedder.fit_encode(all_texts)

            print(f"Vector store ready. {self.index.ntotal} vectors loaded.")
        else:
            print(f"Vector store not found at {self.vector_store_dir}. Please run ingestion script.")

    def reload(self):
        self._load_store()

    def search(self, query: str, top_k: int = 4, min_score: float = 0.10) -> List[Dict[str, Any]]:
        if self.index is None:
            self._load_store()
            if self.index is None:
                return []

        # Encode query
        if self.model is not None:
            try:
                query_vector = self.model.encode([query], convert_to_numpy=True)
            except Exception:
                if self.fallback_embedder is not None:
                    query_vector = self.fallback_embedder.encode([query])
                else:
                    return []
        elif self.fallback_embedder is not None:
            query_vector = self.fallback_embedder.encode([query])
        else:
            return []

        faiss.normalize_L2(query_vector)

        top_k = min(top_k, self.index.ntotal)
        if top_k <= 0:
            return []

        scores, indices = self.index.search(query_vector.astype(np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            float_score = float(score)
            if float_score >= min_score:
                item = dict(self.metadata[idx])
                item["score"] = round(float_score, 4)
                results.append(item)

        return results

retriever_instance = None

def get_retriever() -> FAISSRetriever:
    global retriever_instance
    if retriever_instance is None:
        retriever_instance = FAISSRetriever()
    return retriever_instance
