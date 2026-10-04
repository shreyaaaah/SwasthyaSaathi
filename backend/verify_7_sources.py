"""
verify_7_sources.py — Targeted 15+ word phrase verification for the 7 updated URLs.
"""
import sys, os, re, requests, warnings
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.agent.orchestrator import extract_pdf_content

raw_docs_dir = os.path.join(BACKEND_DIR, "data", "raw_docs")
sources_md_path = os.path.join(raw_docs_dir, "SOURCES.md")

target_7 = {
    "flood_health_guidelines.txt",
    "air_pollution_advisory.txt",
    "tb_public_faqs.txt",
    "maternal_child_health_guidelines.txt",
    "diabetes_hypertension_guidelines.txt",
    "mental_health_depression_guidelines.txt",
    "standard-treatment-guidelines.pdf"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

print("=" * 110)
print("TARGETED 15+ WORD PHRASE VERIFICATION FOR 7 UPDATED SOURCES")
print("=" * 110 + "\n")

with open(sources_md_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"{'File':<42} | {'HTTP':<5} | {'Result'}")
print("-" * 110)

passes = 0
total = 0

for line in lines:
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if len(parts) < 6 or parts[0].startswith("-") or parts[0].startswith("File") or parts[0].startswith("#"):
        continue

    fname, pub, url, fdate, sha, notes = parts[:6]
    if fname not in target_7:
        continue

    fpath = os.path.join(raw_docs_dir, fname)
    total += 1

    status_str = "N/A"
    page_text_clean = ""
    try:
        r = requests.get(url, headers=headers, timeout=12, verify=False)
        status_str = str(r.status_code)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "").lower()
            if "application/pdf" in ct or url.lower().endswith(".pdf"):
                pdf_text, _ = extract_pdf_content(r.content, fname)
                page_text_clean = re.sub(r'[^\w\s]', ' ', pdf_text.lower())
                page_text_clean = re.sub(r'\s+', ' ', page_text_clean)
            else:
                soup = BeautifulSoup(r.text, "html.parser")
                for elem in soup(["script", "style", "nav", "header", "footer", "form", "aside", "svg"]):
                    elem.decompose()
                raw_extracted = soup.get_text(separator=" ")
                page_text_clean = re.sub(r'[^\w\s]', ' ', raw_extracted.lower())
                page_text_clean = re.sub(r'\s+', ' ', page_text_clean)
    except Exception as ex:
        status_str = "ERR"

    if not os.path.exists(fpath):
        result = "FAIL (File missing from disk)"
        print(f"{fname:<42} | {status_str:<5} | {result}")
        continue

    try:
        if fname.lower().endswith(".pdf"):
            with open(fpath, "rb") as pdf_f:
                local_raw, _ = extract_pdf_content(pdf_f.read(), fname)
        else:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as file_obj:
                local_raw = file_obj.read()
    except Exception as e:
        result = f"FAIL (Read error: {e})"
        print(f"{fname:<42} | {status_str:<5} | {result}")
        continue

    local_clean = re.sub(r'[^\w\s]', ' ', local_raw.lower())
    words = local_clean.split()

    if len(words) < 15:
        result = "FAIL (Local file too short <15 words)"
        print(f"{fname:<42} | {status_str:<5} | {result}")
        continue

    if not page_text_clean:
        result = f"FAIL (Could not fetch/parse URL, HTTP {status_str})"
        print(f"{fname:<42} | {status_str:<5} | {result}")
        continue

    max_offset = min(len(words) - 15, 400)
    phrases = [' '.join(words[i:i+15]) for i in range(0, max_offset, 5)]
    matched = [p for p in phrases if p in page_text_clean]

    if matched:
        result = f"PASS ({len(matched)} phrase matches)"
        passes += 1
    else:
        result = "FAIL (No 15+ word phrase match in fetched page text)"

    print(f"{fname:<42} | {status_str:<5} | {result}")

print(f"\n{'=' * 110}")
print(f"SUMMARY FOR 7 TARGET FILES: {passes}/{total} files PASS")
print(f"{'=' * 110}")
