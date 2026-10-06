import os
import json
import re

backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
advisories_dir = os.path.join(backend_dir, 'data', 'advisories')
if not os.path.exists(advisories_dir):
    advisories_dir = os.path.join(os.path.dirname(__file__), 'data', 'advisories')

sources_md_path = os.path.join(backend_dir, 'data', 'raw_docs', 'SOURCES.md')
if not os.path.exists(sources_md_path):
    sources_md_path = os.path.join(os.path.dirname(__file__), 'data', 'raw_docs', 'SOURCES.md')

chunk_meta_path = os.path.join(backend_dir, 'data', 'vector_store', 'chunk_metadata.json')
if not os.path.exists(chunk_meta_path):
    chunk_meta_path = os.path.join(os.path.dirname(__file__), 'data', 'vector_store', 'chunk_metadata.json')

faiss_bin_path = os.path.join(backend_dir, 'data', 'vector_store', 'faiss_index.bin')
if not os.path.exists(faiss_bin_path):
    faiss_bin_path = os.path.join(os.path.dirname(__file__), 'data', 'vector_store', 'faiss_index.bin')

print(f"Advisories dir: {advisories_dir}")
print(f"SOURCES.md path: {sources_md_path}")
print(f"chunk_metadata.json path: {chunk_meta_path}")
print(f"faiss_index.bin path: {faiss_bin_path}")

# 1. Disk files in advisories/
disk_files = set(os.listdir(advisories_dir)) if os.path.exists(advisories_dir) else set()
print(f"\nFiles on disk in data/advisories/ ({len(disk_files)} files):")
for f in sorted(disk_files):
    print(f"  - {f}")

# 2. Files in SOURCES.md
sources_md_files = set()
if os.path.exists(sources_md_path):
    with open(sources_md_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Find filenames mentioned in markdown, e.g. `filename` or - filename
        matches = re.findall(r'[\`\*]*([\w\-\.]+\.(?:pdf|txt))[\`\*]*', content)
        sources_md_files = set(matches)

print(f"\nFiles in SOURCES.md ({len(sources_md_files)} files):")
for f in sorted(sources_md_files):
    print(f"  - {f}")

# 3. Unique doc_name / source_file in chunk_metadata.json
vector_store_files = set()
if os.path.exists(chunk_meta_path):
    with open(chunk_meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for chunk in data:
            doc = chunk.get("doc_name") or chunk.get("source_file") or chunk.get("file_name")
            if doc:
                vector_store_files.add(doc)

print(f"\nUnique doc_name/source_file in chunk_metadata.json ({len(vector_store_files)} files):")
for f in sorted(vector_store_files):
    print(f"  - {f}")

# 4. Diff
not_in_sources_md = vector_store_files - sources_md_files
not_on_disk = vector_store_files - disk_files

print("\n=== DIFF RESULTS ===")
print(f"Filenames in vector store but NOT in SOURCES.md: {sorted(list(not_in_sources_md))}")
print(f"Filenames in vector store that NO LONGER EXIST on disk in data/advisories/: {sorted(list(not_on_disk))}")

# 5. Check specific 3 deleted files
target_deleted = [
    "fever_flu_diarrhea_selfcare.txt",
    "cardiac_emergency_guidelines.txt",
    "hypertension_diabetes_management.txt"
]

print("\n=== SPECIFIC DELETED FILES CHECK ===")
for target in target_deleted:
    in_meta = target in vector_store_files
    print(f"  - '{target}' in chunk_metadata.json: {in_meta}")

if os.path.exists(faiss_bin_path):
    with open(faiss_bin_path, 'rb') as f:
        bin_content = f.read()
        for target in target_deleted:
            in_bin = target.encode('utf-8') in bin_content
            print(f"  - '{target}' in faiss_index.bin raw bytes: {in_bin}")
