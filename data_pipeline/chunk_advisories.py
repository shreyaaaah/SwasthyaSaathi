"""
chunk_advisories.py — Extractor and chunker for .txt and .pdf WHO/MoHFW public health advisories.
Supports both pdfplumber and PyMuPDF (fitz) for PDF parsing, plain read for .txt.
Ignores non-PDF/non-TXT files (like .json or folders).
"""
import os
import json
import re
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVISORIES_DIR = os.path.join(BASE_DIR, "data", "advisories")
BACKEND_RAW_DOCS_DIR = os.path.join(BASE_DIR, "backend", "data", "raw_docs")

OUTPUT_FILE = os.path.join(BASE_DIR, "data", "advisory_chunks.json")
BACKEND_OUTPUT_FILE = os.path.join(BASE_DIR, "backend", "data", "advisory_chunks.json")

TOPIC_MAP = {
    "hiv.pdf": "hiv",
    "healthy diet who.pdf": "healthy_diet",
    "healthydiet.pdf": "healthy_diet",
    "obesity and overweight who.pdf": "obesity",
    "activeworkoutwho.pdf": "physical_activity",
    "whophysicalguideline.pdf": "physical_activity",
    "air_pollution_advisory.txt": "air_pollution",
    "airquality.pdf": "air_quality",
    "anixiety.pdf": "anxiety",
    "cardiac_emergency_guidelines.txt": "cardiac_emergency",
    "corona.pdf": "covid",
    "dengue_clinical_guidelines.txt": "dengue",
    "depression and anxiety.pdf": "depression_anxiety",
    "diabetes_hypertension_guidelines.txt": "hypertension_diabetes",
    "hypertension_diabetes_management.txt": "hypertension_diabetes",
    "fat2.pdf": "nutrition_fat",
    "fatloss.pdf": "weight_loss",
    "fever_flu_diarrhea_selfcare.txt": "fever_flu_diarrhea",
    "flood_health_guidelines.txt": "flood_health",
    "heartdieases.pdf": "heart_disease",
    "hepatitis b.pdf": "hepatitis_b",
    "infection.pdf": "infection_control",
    "low hameoglobin.pdf": "low_hemoglobin_anemia",
    "malaria.pdf": "malaria",
    "malaria_guidelines.txt": "malaria",
    "maternal_child_health_guidelines.txt": "maternal_child_health",
    "mengitis.pdf": "meningitis",
    "mental_health_depression_guidelines.txt": "mental_health",
    "mentalhealth at work.pdf": "mental_health_workplace",
    "phuenomia and diahria.pdf": "pneumonia_diarrhea",
    "selfcare.pdf": "self_care",
    "selfcare (1).pdf": "self_care",
    "standard-treatment-guidelines.pdf": "standard_treatment",
    "stress.pdf": "stress",
    "tb_public_faqs.txt": "tuberculosis",
    "tuberclosis.pdf": "tuberculosis"
}

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from PDF using PyMuPDF (fitz) or pdfplumber."""
    try:
        import fitz  # PyMuPDF is extremely fast
        doc = fitz.open(pdf_path)
        pages_text = [page.get_text() for page in doc if page.get_text()]
        text = "\n\n".join(pages_text)
        if len(text.strip()) > 100:
            return text
    except Exception as e:
        print(f"    (fitz fallback to pdfplumber: {e})")

    try:
        import pdfplumber
        pages_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages_text.append(t)
        return "\n\n".join(pages_text)
    except Exception as e:
        print(f"    Error reading PDF with pdfplumber: {e}")
        return ""

def chunk_text(doc_name: str, topic: str, content: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    chunks = []
    content = content.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    
    current_section = "General Overview"
    current_buffer = ""
    
    for p in paragraphs:
        # Check if line looks like a header
        if len(p) < 100 and (p.isupper() or p.startswith("SECTION") or p.startswith("#") or p.startswith("FAQ") or p.endswith(":")):
            current_section = re.sub(r"^[#\s]+", "", p).strip()
            continue
            
        if len(current_buffer) + len(p) < chunk_size:
            current_buffer += ("\n\n" if current_buffer else "") + p
        else:
            if len(current_buffer.strip()) >= 50:
                chunks.append({
                    "doc_name": doc_name,
                    "topic": topic,
                    "section": current_section,
                    "chunk_text": current_buffer.strip()
                })
            words = current_buffer.split()
            overlap_words = words[-30:] if len(words) > 30 else words
            current_buffer = " ".join(overlap_words) + "\n\n" + p
            
    if len(current_buffer.strip()) >= 50:
        chunks.append({
            "doc_name": doc_name,
            "topic": topic,
            "section": current_section,
            "chunk_text": current_buffer.strip()
        })
        
    return chunks

def main():
    active_dir = ADVISORIES_DIR if os.path.exists(ADVISORIES_DIR) else BACKEND_RAW_DOCS_DIR
    if not os.path.exists(active_dir):
        print(f"Error: Advisories directory not found at {active_dir}")
        return

    print(f"Processing advisories directory: {active_dir}")
    all_files = sorted(os.listdir(active_dir))
    
    pdf_files = []
    txt_files = []
    excluded_files = []

    for f in all_files:
        full_path = os.path.join(active_dir, f)
        if os.path.isdir(full_path):
            excluded_files.append((f, "DIRECTORY"))
        elif f.lower().endswith(".pdf"):
            pdf_files.append(f)
        elif f.lower().endswith(".txt"):
            txt_files.append(f)
        else:
            excluded_files.append((f, f"NON-PDF/TXT FILE ({os.path.splitext(f)[1]})"))

    print(f"\nFound {len(pdf_files)} PDF files, {len(txt_files)} TXT files, {len(excluded_files)} EXCLUDED items.")
    if excluded_files:
        print("Excluded items:")
        for name, reason in excluded_files:
            print(f"  - {name}: {reason}")

    # Exclude duplicates like 'selfcare (1).pdf'
    processed_files = []
    for f in pdf_files + txt_files:
        if "(1)" in f:
            print(f"Skipping duplicate file: {f}")
            continue
        processed_files.append(f)

    all_chunks = []
    chunk_counter = 0

    for file_name in sorted(processed_files):
        topic = TOPIC_MAP.get(file_name.lower())
        if not topic:
            clean_name = os.path.splitext(file_name)[0].lower()
            topic = re.sub(r"[^\w]+", "_", clean_name).strip("_")

        file_path = os.path.join(active_dir, file_name)
        print(f"Processing: {file_name} -> topic: '{topic}'...")

        if file_name.lower().endswith(".pdf"):
            content = extract_text_from_pdf(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            print(f"  WARNING: Empty content extracted from {file_name}")
            continue

        file_chunks = chunk_text(file_name, topic, content)
        print(f"  Extracted {len(content)} chars -> {len(file_chunks)} chunks.")

        for c in file_chunks:
            c["chunk_id"] = chunk_counter
            all_chunks.append(c)
            chunk_counter += 1

    # Save chunks JSON
    for path in [OUTPUT_FILE, BACKEND_OUTPUT_FILE]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_chunks)} chunks to {path}")

    # Breakdown by topic
    topic_counts = {}
    for c in all_chunks:
        t = c["topic"]
        topic_counts[t] = topic_counts.get(t, 0) + 1

    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"Total Source Documents Processed: {len(processed_files)}")
    print(f"Total Chunks Generated: {len(all_chunks)}")
    print(f"Total Topics: {len(topic_counts)}\n")
    print("Breakdown per Topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"  - {topic}: {count} chunks")

if __name__ == "__main__":
    main()
