import os
import json
import re
import pickle
from typing import List, Dict, Any
import numpy as np

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None

# Default paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RAW_DOCS_DIR = os.path.join(BASE_DIR, "data", "raw_docs")
DEFAULT_VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")
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

def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from .txt, .md, or .pdf files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        if pypdf is None:
            raise ImportError("pypdf package is required for processing PDF files.")
        reader = pypdf.PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        return "\n".join(text_parts)
    return ""

def chunk_document(doc_name: str, content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    chunks = []
    lines = content.split("\n")
    
    current_section = "General Overview"
    current_buffer = ""
    
    for line in lines:
        stripped = line.strip()
        if stripped.isupper() and len(stripped) > 5 or stripped.startswith("SECTION") or stripped.startswith("#"):
            if stripped.startswith("DOCUMENT:"):
                doc_name = stripped.replace("DOCUMENT:", "").strip()
            current_section = re.sub(r"^[#\s]+", "", stripped)
        
        current_buffer += line + "\n"
        
        if len(current_buffer) >= chunk_size:
            chunk_text = current_buffer.strip()
            if chunk_text:
                chunks.append({
                    "doc_name": doc_name,
                    "section": current_section,
                    "chunk_text": chunk_text
                })
            overlap_buffer = current_buffer[-chunk_overlap:] if len(current_buffer) > chunk_overlap else ""
            current_buffer = overlap_buffer

    if current_buffer.strip():
        chunks.append({
            "doc_name": doc_name,
            "section": current_section,
            "chunk_text": current_buffer.strip()
        })

    return chunks

def build_and_save_index(raw_docs_dir: str = None, vector_store_dir: str = None) -> Dict[str, Any]:
    raw_docs_dir = raw_docs_dir or DEFAULT_RAW_DOCS_DIR
    vector_store_dir = vector_store_dir or DEFAULT_VECTOR_STORE_DIR
    
    os.makedirs(vector_store_dir, exist_ok=True)
    os.makedirs(raw_docs_dir, exist_ok=True)

    all_chunks = []
    doc_files = [f for f in os.listdir(raw_docs_dir) if f.endswith((".txt", ".md", ".pdf"))]
    
    if not doc_files:
        print(f"No documents found in {raw_docs_dir}")
        return {"status": "warning", "message": "No documents found to ingest", "documents_indexed": 0, "chunks_created": 0}

    for file_name in doc_files:
        file_path = os.path.join(raw_docs_dir, file_name)
        text = extract_text_from_file(file_path)
        if text.strip():
            doc_chunks = chunk_document(doc_name=file_name, content=text)
            all_chunks.extend(doc_chunks)

    if not all_chunks:
        return {"status": "warning", "message": "No text content extracted", "documents_indexed": 0, "chunks_created": 0}

    print(f"Extracted {len(all_chunks)} chunks across {len(doc_files)} documents.")
    texts_to_embed = [c["chunk_text"] for c in all_chunks]

    model = None
    use_fallback = False
    
    try:
        print(f"Attempting to load SentenceTransformer: {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME, local_files_only=False)
        embeddings = model.encode(texts_to_embed, show_progress_bar=False, convert_to_numpy=True)
    except Exception as e:
        print(f"Could not load HuggingFace online model ({e}). Using local TF-IDF fallback embedder...")
        use_fallback = True
        embedder = FallbackEmbedder(dim=384)
        embeddings = embedder.fit_encode(texts_to_embed)
        vectorizer_path = os.path.join(vector_store_dir, "vectorizer.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(embedder, f)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    # Save FAISS index
    index_path = os.path.join(vector_store_dir, "index.faiss")
    faiss.write_index(index, index_path)

    # Save metadata sidecar
    metadata_sidecar = []
    for idx, chunk in enumerate(all_chunks):
        metadata_sidecar.append({
            "chunk_id": idx,
            "doc_name": chunk["doc_name"],
            "section": chunk["section"],
            "chunk_text": chunk["chunk_text"]
        })

    metadata_path = os.path.join(vector_store_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_sidecar, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS index to: {index_path} (Fallback: {use_fallback})")
    print(f"Saved Metadata sidecar to: {metadata_path}")

    return {
        "status": "success",
        "documents_indexed": len(doc_files),
        "chunks_created": len(all_chunks),
        "vector_store_path": vector_store_dir,
        "is_fallback_embedding": use_fallback
    }

if __name__ == "__main__":
    result = build_and_save_index()
    print("Ingestion Result:", result)
