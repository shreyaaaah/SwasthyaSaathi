import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CHUNKS_FILE = os.path.join(BASE_DIR, "data", "advisory_chunks.json")

VECTOR_STORE_DIRS = [
    os.path.join(BASE_DIR, "data", "vector_store"),
    os.path.join(BASE_DIR, "backend", "data", "vector_store")
]

MODEL_NAME = "all-MiniLM-L6-v2"

class FallbackEmbedder:
    """Fallback TF-IDF + Hashing embedder when HuggingFace / online transformer is unreachable."""
    def __init__(self, dim=384):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=dim, stop_words="english")
        self.is_fitted = False

    def fit_encode(self, texts: List[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts).toarray()
        if matrix.shape[1] < self.dim:
            pad_width = self.dim - matrix.shape[1]
            matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode='constant')
        self.is_fitted = True
        return matrix.astype(np.float32)

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if not self.is_fitted:
            return self.fit_encode(texts)
        matrix = self.vectorizer.transform(texts).toarray()
        if matrix.shape[1] < self.dim:
            pad_width = self.dim - matrix.shape[1]
            matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode='constant')
        return matrix.astype(np.float32)

def main():
    if not os.path.exists(INPUT_CHUNKS_FILE):
        print(f"Error: Input chunks file not found at {INPUT_CHUNKS_FILE}")
        return

    with open(INPUT_CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("Error: Chunks list is empty.")
        return

    print(f"Loaded {len(chunks)} chunks from {INPUT_CHUNKS_FILE}")
    texts_to_embed = [c["chunk_text"] for c in chunks]

    use_fallback = False
    model = None
    try:
        print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(texts_to_embed, show_progress_bar=False, convert_to_numpy=True)
    except Exception as e:
        print(f"SentenceTransformer load failed ({e}). Using FallbackEmbedder...")
        use_fallback = True
        embedder = FallbackEmbedder(dim=384)
        embeddings = embedder.fit_encode(texts_to_embed)

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    for target_dir in VECTOR_STORE_DIRS:
        os.makedirs(target_dir, exist_ok=True)
        
        # Save both faiss_index.bin and index.faiss
        faiss_bin_path = os.path.join(target_dir, "faiss_index.bin")
        index_faiss_path = os.path.join(target_dir, "index.faiss")
        faiss.write_index(index, faiss_bin_path)
        faiss.write_index(index, index_faiss_path)

        # Save both chunk_metadata.json and metadata.json
        chunk_meta_path = os.path.join(target_dir, "chunk_metadata.json")
        metadata_json_path = os.path.join(target_dir, "metadata.json")

        with open(chunk_meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        if use_fallback:
            vectorizer_path = os.path.join(target_dir, "vectorizer.pkl")
            with open(vectorizer_path, "wb") as f:
                pickle.dump(embedder, f)

        print(f"Successfully written FAISS index & metadata to: {target_dir}")

    print("=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"Total Vectors Indexed: {index.ntotal}")
    print(f"Embedding Dimension: {dimension}")
    print(f"Fallback Mode: {use_fallback}")

if __name__ == "__main__":
    main()
