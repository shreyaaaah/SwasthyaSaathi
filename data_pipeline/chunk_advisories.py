import os
import json
import re
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVISORIES_DIR = os.path.join(BASE_DIR, "data", "advisories")
# Also check backend/data/raw_docs in case pipeline is run from within backend directory
BACKEND_RAW_DOCS_DIR = os.path.join(BASE_DIR, "backend", "data", "raw_docs")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "advisory_chunks.json")
BACKEND_OUTPUT_FILE = os.path.join(BASE_DIR, "backend", "data", "advisory_chunks.json")

TOPIC_MAP = {
    "air_pollution_advisory.txt": "air_pollution",
    "cardiac_emergency_guidelines.txt": "cardiac_emergency",
    "dengue_clinical_guidelines.txt": "dengue",
    "fever_flu_diarrhea_selfcare.txt": "fever_flu_diarrhea",
    "flood_health_guidelines.txt": "flood_health",
    "hypertension_diabetes_management.txt": "hypertension_diabetes",
    "maternal_child_health_guidelines.txt": "maternal_child_health",
    "tb_public_faqs.txt": "tuberculosis"
}

def chunk_text(doc_name: str, topic: str, content: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    chunks = []
    lines = content.split("\n")
    
    current_section = "General Overview"
    current_buffer = ""
    
    for line in lines:
        stripped = line.strip()
        if stripped.isupper() and len(stripped) > 5 or stripped.startswith("SECTION") or stripped.startswith("#") or stripped.startswith("GUIDELINES") or stripped.startswith("FAQ"):
            if stripped.startswith("DOCUMENT:"):
                doc_name = stripped.replace("DOCUMENT:", "").strip()
            current_section = re.sub(r"^[#\s]+", "", stripped)
        
        current_buffer += line + "\n"
        
        if len(current_buffer) >= chunk_size:
            chunk_str = current_buffer.strip()
            if chunk_str:
                chunks.append({
                    "doc_name": doc_name,
                    "topic": topic,
                    "section": current_section,
                    "chunk_text": chunk_str
                })
            overlap_buffer = current_buffer[-chunk_overlap:] if len(current_buffer) > chunk_overlap else ""
            current_buffer = overlap_buffer

    if current_buffer.strip():
        chunks.append({
            "doc_name": doc_name,
            "topic": topic,
            "section": current_section,
            "chunk_text": current_buffer.strip()
        })

    return chunks

def main():
    # Prefer ADVISORIES_DIR; fall back to BACKEND_RAW_DOCS_DIR
    active_dir = ADVISORIES_DIR if os.path.exists(ADVISORIES_DIR) else BACKEND_RAW_DOCS_DIR
    if not os.path.exists(active_dir):
        print(f"Error: Advisories directory not found at {ADVISORIES_DIR} or {BACKEND_RAW_DOCS_DIR}")
        return
    print(f"Reading advisories from: {active_dir}")

    files = [f for f in os.listdir(active_dir) if f.endswith(".txt")]
    all_chunks = []
    chunk_counter = 0

    for file_name in sorted(files):
        topic = TOPIC_MAP.get(file_name, file_name.replace(".txt", ""))
        file_path = os.path.join(active_dir, file_name)
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        file_chunks = chunk_text(file_name, topic, content)
        for c in file_chunks:
            c["chunk_id"] = chunk_counter
            all_chunks.append(c)
            chunk_counter += 1

    # Save outputs
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(BACKEND_OUTPUT_FILE), exist_ok=True)
    with open(BACKEND_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Calculate summary
    topic_counts = {}
    sample_per_topic = {}
    for c in all_chunks:
        t = c["topic"]
        topic_counts[t] = topic_counts.get(t, 0) + 1
        if t not in sample_per_topic:
            sample_per_topic[t] = c

    print("=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"Total Chunks Overall: {len(all_chunks)}\n")
    print("Chunks per Topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  - {topic}: {count} chunks")

    print("\n" + "=" * 60)
    print("SAMPLE CHUNKS PER TOPIC")
    print("=" * 60)
    for topic, sample in sorted(sample_per_topic.items()):
        print(f"\n--- TOPIC: {topic} (Doc: {sample['doc_name']}, Section: {sample['section']}) ---")
        print(sample['chunk_text'])

if __name__ == "__main__":
    main()
