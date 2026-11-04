import os
import json
import re
from typing import List, Dict, Any
import numpy as np
from app.config import settings
from app.rag.source_registry import validate_source_registry

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    from fastembed import TextEmbedding
    import faiss
except ImportError:
    TextEmbedding = None
    faiss = None

# Default paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RAW_DOCS_DIR = os.path.join(BASE_DIR, "data", "raw_docs")
DEFAULT_VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")
MODEL_NAME = settings.EMBEDDING_MODEL_NAME

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
    registry = validate_source_registry(raw_docs_dir)

    all_chunks = []
    doc_files = sorted(registry)
    
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

    if TextEmbedding is None or faiss is None:
        raise RuntimeError("fastembed and faiss-cpu are required to build the vector store")
    model_name = MODEL_NAME if "/" in MODEL_NAME else f"sentence-transformers/{MODEL_NAME}"
    model = TextEmbedding(model_name=model_name)
    embeddings = np.asarray(list(model.embed(texts_to_embed)), dtype=np.float32)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    # Save FAISS index
    for index_name in ("faiss_index.bin", "index.faiss"):
        faiss.write_index(index, os.path.join(vector_store_dir, index_name))

    # Save metadata sidecar
    metadata_sidecar = []
    for idx, chunk in enumerate(all_chunks):
        metadata_sidecar.append({
            "chunk_id": idx,
            "doc_name": chunk["doc_name"],
            "section": chunk["section"],
            "chunk_text": chunk["chunk_text"]
        })

    for metadata_name in ("chunk_metadata.json", "metadata.json"):
        with open(os.path.join(vector_store_dir, metadata_name), "w", encoding="utf-8") as f:
            json.dump(metadata_sidecar, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS index and metadata to: {vector_store_dir}")

    return {
        "status": "success",
        "documents_indexed": len(doc_files),
        "chunks_created": len(all_chunks),
        "vector_store_path": vector_store_dir,
        "is_fallback_embedding": False
    }

if __name__ == "__main__":
    result = build_and_save_index()
    print("Ingestion Result:", result)
