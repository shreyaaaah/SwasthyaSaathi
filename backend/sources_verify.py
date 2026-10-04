"""
sources_verify.py — Robust & Reliable SOURCES.md 15+ Word Phrase Verification.

Fixes applied:
1. Removed 400-word offset cap — scans entire document word count.
2. Session retry strategy (3 retries with backoff) to handle gov server flakiness.
3. Strict normalization of local and fetched text (lowercased, non-alphanumeric stripped, normalized spaces).
4. Fallback resolution for local files across backend/data/raw_docs, backend/data/advisories, data/advisories.
"""
import sys
import os
import re
import warnings
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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

# Configure robust session with retries
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}

print("=" * 110)
print("SOURCES.MD 15+ WORD PHRASE VERIFICATION (DETERMINISTIC SINGLE RUN)")
print("=" * 110 + "\n")

if not os.path.exists(sources_md_path):
    print(f"Error: SOURCES.md not found at {sources_md_path}")
    sys.exit(1)

with open(sources_md_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"{'File':<38} | {'HTTP':<5} | {'Result'}")
print("-" * 110)

mismatches = []
passes = 0
total = 0

def resolve_local_file(fname):
    candidates = [
        os.path.join(BACKEND_DIR, "data", "raw_docs", fname),
        os.path.join(BACKEND_DIR, "data", "advisories", fname),
        os.path.join(BASE_DIR, "data", "advisories", fname)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def normalize_text(text):
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    return re.sub(r'\s+', ' ', clean).strip()

for line in lines:
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if len(parts) < 6 or parts[0].startswith("-") or parts[0].startswith("File") or parts[0].startswith("#"):
        continue

    fname, pub, url, fdate, sha, notes = parts[:6]
    total += 1
    fpath = resolve_local_file(fname)

    # 1. Fetch URL
    status_str = "N/A"
    page_text_clean = ""
    try:
        r = session.get(url, headers=headers, timeout=15, verify=False)
        status_str = str(r.status_code)
        if r.status_code == 200:
            ct = r.headers.get("Content-Type", "").lower()
            if "application/pdf" in ct or url.lower().endswith(".pdf"):
                pdf_text, _ = extract_pdf_content(r.content, fname)
                page_text_clean = normalize_text(pdf_text)
            else:
                soup = BeautifulSoup(r.text, "html.parser")
                for elem in soup(["script", "style", "nav", "header", "footer", "form", "aside", "svg"]):
                    elem.decompose()
                raw_extracted = soup.get_text(separator=" ")
                page_text_clean = normalize_text(raw_extracted)
    except Exception as ex:
        status_str = "ERR"

    # 2. Read local file
    if not fpath:
        result = "FAIL (File missing from disk)"
        mismatches.append((fname, result))
        print(f"{fname:<38} | {status_str:<5} | {result}")
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
        mismatches.append((fname, result))
        print(f"{fname:<38} | {status_str:<5} | {result}")
        continue

    local_clean = normalize_text(local_raw)
    words = local_clean.split()

    if len(words) < 15:
        result = "FAIL (Local file too short <15 words)"
        mismatches.append((fname, result))
        print(f"{fname:<38} | {status_str:<5} | {result}")
        continue

    if not page_text_clean:
        result = f"FAIL (Could not fetch/parse URL, HTTP {status_str})"
        mismatches.append((fname, result))
        print(f"{fname:<38} | {status_str:<5} | {result}")
        continue

    # 3. 15-word phrase sliding window scan across FULL document
    phrases = [' '.join(words[i:i+15]) for i in range(0, len(words) - 15, 5)]
    matched = [p for p in phrases if p in page_text_clean]

    if matched:
        result = f"PASS ({len(matched)} phrase matches)"
        passes += 1
    else:
        result = "FAIL (No 15+ word phrase match in fetched page text)"
        mismatches.append((fname, result))

    print(f"{fname:<38} | {status_str:<5} | {result}")

print(f"\n{'=' * 110}")
print(f"SUMMARY: {passes}/{total} files PASS | {len(mismatches)}/{total} files FAIL")
print(f"{'=' * 110}")

if mismatches:
    print("\nFailed files:")
    for m_file, m_reason in mismatches:
        print(f"  - {m_file}: {m_reason}")
