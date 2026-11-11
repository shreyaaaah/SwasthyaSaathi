import os
import json
import re
import time
from typing import List, Dict, Any
import numpy as np
from app.config import settings
from app.rag.source_registry import validate_source_registry

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

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

_PDF_UNICODE_GLYPH = re.compile(r"/uni([0-9a-fA-F]{4,6})")


def _decode_pdf_unicode_glyphs(text: str) -> str:
    """Repair pypdf's /uniXXXX output for PDFs with broken ToUnicode maps."""
    matches = list(_PDF_UNICODE_GLYPH.finditer(text))
    if len(matches) < 8:
        return text
    return _PDF_UNICODE_GLYPH.sub(lambda match: chr(int(match.group(1), 16)), text)

def extract_text_from_file(file_path: str) -> str:
    """Extract plain text from .txt, .md, or .pdf files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext == ".pdf":
        if fitz is not None:
            try:
                with fitz.open(file_path) as document:
                    pages = [page.get_text("text") for page in document]
                extracted = "\n".join(text for text in pages if text)
                if extracted.strip():
                    return extracted
            except Exception as exc:
                print(f"PyMuPDF could not extract {os.path.basename(file_path)} ({exc}); falling back to pypdf.")
        if pypdf is None:
            raise ImportError("PyMuPDF or pypdf is required for processing PDF files.")
        reader = pypdf.PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(_decode_pdf_unicode_glyphs(t))
        return "\n".join(text_parts)
    return ""

def chunk_document(doc_name: str, content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """Split a document into bounded, word-aligned chunks within each section."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size - 1")

    chunks: List[Dict[str, Any]] = []
    current_section = "General Overview"
    section_lines: list[str] = []

    def flush_section() -> None:
        text = " ".join(part.strip() for part in section_lines if part.strip())
        text = re.sub(r"\s+", " ", text).strip()
        for chunk_text in _split_into_chunks(text, chunk_size, chunk_overlap):
            chunks.append({
                "doc_name": doc_name,
                "section": current_section,
                "chunk_text": chunk_text,
            })

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_heading = (
            stripped.startswith("#")
            or stripped.upper().startswith("SECTION ")
            or (stripped.isupper() and sum(char.isalpha() for char in stripped) >= 5 and len(stripped) <= 160)
        )
        if is_heading:
            # Flush before changing section so a chunk never receives the next
            # heading's label while still containing the previous section.
            flush_section()
            current_section = re.sub(r"^[#\s]+", "", stripped)
            section_lines = [stripped]
        else:
            section_lines.append(stripped)

    flush_section()
    return chunks


def _split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Prefer sentence and word boundaries while strictly bounding chunk size."""
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            # Prefer a sentence break in the latter half of the window.
            sentence_breaks = [
                match.end()
                for match in re.finditer(r"[.!?](?:[\"'”’)]*)?(?:\s+|$)", text[start:hard_end])
                if match.end() >= chunk_size // 2
            ]
            if sentence_breaks:
                end = start + sentence_breaks[-1]
            else:
                whitespace = text.rfind(" ", start, hard_end)
                if whitespace > start:
                    end = whitespace

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(start + 1, end - chunk_overlap)
        # Move overlap to the next whole word when its initial offset landed
        # inside one. This may shorten the overlap but never split a token.
        while next_start < end and next_start > 0 and not text[next_start - 1].isspace() and not text[next_start].isspace():
            next_start += 1
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        start = next_start

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

    for idx, file_name in enumerate(doc_files, 1):
        file_path = os.path.join(raw_docs_dir, file_name)
        try:
            text = extract_text_from_file(file_path)
            if text.strip():
                doc_chunks = chunk_document(doc_name=file_name, content=text)
                all_chunks.extend(doc_chunks)
                print(f"[{idx}/{len(doc_files)}] Processed '{file_name}': {len(doc_chunks)} chunks (total so far: {len(all_chunks)})")
            else:
                print(f"[{idx}/{len(doc_files)}] Warning: '{file_name}' yielded empty text.")
        except Exception as exc:
            raise RuntimeError(f"Error extracting or chunking document '{file_name}': {exc}") from exc

    if not all_chunks:
        return {"status": "warning", "message": "No text content extracted", "documents_indexed": 0, "chunks_created": 0}

    print(f"Extracted {len(all_chunks)} chunks across {len(doc_files)} documents.")
    texts_to_embed = [c["chunk_text"] for c in all_chunks]

    if TextEmbedding is None or faiss is None:
        raise RuntimeError("fastembed and faiss-cpu are required to build the vector store")
    model_name = MODEL_NAME if "/" in MODEL_NAME else f"sentence-transformers/{MODEL_NAME}"
    print(f"Initializing FastEmbed model: '{model_name}'...")
    try:
        model = TextEmbedding(model_name=model_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to load FastEmbed model '{model_name}': {exc}") from exc

    batch_size = 256
    total_chunks = len(texts_to_embed)
    print(f"Generating embeddings for {total_chunks} chunks using FastEmbed '{model_name}' (batch_size={batch_size})...")

    embedded_vectors = []
    start_time = time.perf_counter()
    last_log_time = start_time
    try:
        for idx, vec in enumerate(model.embed(texts_to_embed, batch_size=batch_size), 1):
            embedded_vectors.append(vec)
            now = time.perf_counter()
            if idx % 2500 == 0 or idx == total_chunks or (now - last_log_time >= 10.0):
                elapsed = now - start_time
                speed = idx / elapsed if elapsed > 0 else 0
                eta = (total_chunks - idx) / speed if speed > 0 else 0
                print(
                    f"  [EMBED PROGRESS] {idx}/{total_chunks} chunks ({idx/total_chunks*100:.1f}%) | "
                    f"Elapsed: {elapsed:.1f}s | Speed: {speed:.1f} chunks/s | ETA: {eta:.1f}s"
                )
                last_log_time = now
    except Exception as exc:
        failed_idx = len(embedded_vectors)
        failed_doc = all_chunks[failed_idx]["doc_name"] if failed_idx < total_chunks else "unknown"
        raise RuntimeError(
            f"Embedding stalled or failed at chunk {failed_idx}/{total_chunks} (doc: '{failed_doc}'): {exc}"
        ) from exc

    embeddings = np.asarray(embedded_vectors, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(all_chunks) or embeddings.shape[1] <= 0:
        raise RuntimeError(
            f"Embedding output shape {embeddings.shape} does not match {len(all_chunks)} input chunks"
        )
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding model returned non-finite values")

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3):
        raise RuntimeError("Embedding model returned zero or invalid vectors after normalization")
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))

    # Save metadata sidecar
    metadata_sidecar = []
    for idx, chunk in enumerate(all_chunks):
        metadata_sidecar.append({
            "chunk_id": idx,
            "doc_name": chunk["doc_name"],
            "section": chunk["section"],
            "chunk_text": chunk["chunk_text"]
        })

    # Write complete temporary artifacts first so a failed rebuild leaves the
    # last known-good index and metadata available to the service.
    pending_files = []
    try:
        for index_name in ("faiss_index.bin", "index.faiss"):
            destination = os.path.join(vector_store_dir, index_name)
            temporary = destination + ".tmp"
            pending_files.append((temporary, destination))
            faiss.write_index(index, temporary)

        for metadata_name in ("chunk_metadata.json", "metadata.json"):
            destination = os.path.join(vector_store_dir, metadata_name)
            temporary = destination + ".tmp"
            pending_files.append((temporary, destination))
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(metadata_sidecar, f, indent=2, ensure_ascii=False)

        for temporary, destination in pending_files:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in pending_files:
            if os.path.exists(temporary):
                os.remove(temporary)

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
    if result.get("status") != "success" or int(result.get("chunks_created", 0)) <= 0:
        raise SystemExit("Ingestion did not create a valid non-empty index; failing the build.")
