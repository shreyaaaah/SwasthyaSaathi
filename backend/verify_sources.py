"""Validate local source hashes; optionally make a best-effort HTTP check."""

import argparse
import json
from pathlib import Path

import requests
import faiss

from app.config import settings
from app.rag.source_registry import SourceRegistryError, validate_source_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-urls", action="store_true",
        help="also check registered URLs with HTTP GET (may be slow or blocked)",
    )
    parser.add_argument(
        "--check-index", action="store_true",
        help="verify the committed FAISS index includes every registered source document",
    )
    args = parser.parse_args()
    raw_docs_dir = Path(settings.RAW_DOCS_DIR)
    try:
        entries = validate_source_registry(raw_docs_dir)
    except SourceRegistryError as error:
        print(f"FAIL: {error}")
        return 1

    print(f"PASS: {len(entries)} registered documents match their local SHA-256 hashes.")
    if args.check_index:
        vector_store_dir = Path(settings.VECTOR_STORE_DIR)
        index_path = vector_store_dir / "faiss_index.bin"
        metadata_path = vector_store_dir / "chunk_metadata.json"
        try:
            index = faiss.read_index(str(index_path))
            chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"FAIL: unable to read vector store: {error}")
            return 3
        if index.ntotal != len(chunks):
            print(f"FAIL: FAISS has {index.ntotal} vectors but metadata has {len(chunks)} rows.")
            return 3
        indexed_documents = {chunk.get("doc_name") for chunk in chunks}
        missing_documents = sorted(set(entries) - indexed_documents)
        if missing_documents:
            print("FAIL: index is stale; missing registered documents: " + ", ".join(missing_documents))
            print("Rebuild it with: python -m app.rag.ingest")
            return 3
        print(f"PASS: the vector store contains {index.ntotal} vectors across all {len(entries)} registered documents.")
    if args.check_urls:
        failed = []
        for filename, entry in entries.items():
            try:
                response = requests.get(
                    entry["url"], timeout=12, allow_redirects=True,
                    headers={"User-Agent": "SwasthyaSaathi-source-check/1.0"},
                )
                print(f"{response.status_code}\t{filename}\t{response.url}")
                if not response.ok:
                    failed.append(filename)
            except requests.RequestException as error:
                print(f"FETCH_ERROR\t{filename}\t{error}")
                failed.append(filename)
        if failed:
            print(f"REVIEW: {len(failed)} URL checks failed: {', '.join(failed)}")
            return 2
        print("PASS: all registered URLs returned a successful HTTP status.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
