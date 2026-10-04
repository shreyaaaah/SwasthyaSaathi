import os
import sys
import requests

sys.stdout.reconfigure(encoding='utf-8')

raw_docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_docs")
sources_path = os.path.join(raw_docs_dir, "SOURCES.md")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

with open(sources_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

urls = []
for line in lines:
    parts = [p.strip() for p in line.split("|") if p.strip()]
    if len(parts) >= 6 and not parts[0].startswith("-") and not parts[0].startswith("File") and not parts[0].startswith("#"):
        urls.append((parts[0], parts[2]))

for fname, url in urls:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        snippet = repr(r.text[:100].replace('\n', ' ').replace('\r', ' '))
        print(f"{fname} | {url} | STATUS: {r.status_code} | SNIPPET: {snippet}")
    except Exception as e:
        print(f"{fname} | {url} | EXCEPTION: {e}")
