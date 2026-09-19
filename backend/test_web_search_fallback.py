"""
test_web_search_fallback.py — Test script to run 3 queries outside the 29-topic KB and verify the live domain-restricted search fallback tool.
"""
import sys
import requests
import json

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"
USER_ID = "web_fallback_user_01"

test_queries = [
    ("1. TYPHOID FEVER SYMPTOMS", "what are the symptoms of typhoid fever"),
    ("2. NEWBORN VACCINES INDIA", "what vaccines does a newborn need in India"),
    ("3. CHIKUNGUNYA VS DENGUE", "how is chikungunya different from dengue")
]

print("======================================================================")
print("LIVE WEB SEARCH FALLBACK INTEGRATION TEST")
print("======================================================================")

for label, query in test_queries:
    print(f"\n{'='*70}")
    print(f"QUERY: [{label}]")
    print(f"User Question: \"{query}\"")
    print('-'*70)

    res = requests.post(f"{BASE_URL}/chat", json={"query": query, "user_id": USER_ID}, timeout=60)
    if res.status_code != 200:
        print(f"HTTP Error {res.status_code}: {res.text[:400]}")
        continue

    data = res.json()
    tools_used = data.get("tools_used", [])
    sources = data.get("sources", [])
    triage_tag = data.get("triage_tag", "")
    answer = data.get("answer", "")
    is_grounded = data.get("is_grounded", True)
    response_time = data.get("response_time_ms", 0)

    print(f"Response Time: {response_time:.2f} ms")
    print(f"Triage Tag:    {triage_tag}")
    print(f"Tools Used:    {tools_used}")
    print(f"Is Grounded:   {is_grounded}")

    kb_sources = [s for s in sources if s.get("source_type") == "knowledge_base"]
    live_sources = [s for s in sources if s.get("source_type") == "live_search"]

    print(f"\n--- FALLBACK CONFIRMATION ---")
    print(f"Knowledge Base Chunks (Score >= 0.45): {len(kb_sources)} (Triggered fallback: {'YES ✅' if len(kb_sources) == 0 else 'NO'})")

    print(f"\n--- RAW LIVE WEB SEARCH RESULTS ({len(live_sources)} items) ---")
    for i, s in enumerate(live_sources, 1):
        print(f"  [{i}] Source Type: '{s.get('source_type')}'")
        print(f"      Title:       {s.get('doc_name') or s.get('title')}")
        print(f"      URL:         {s.get('source_url')}")
        print(f"      Snippet:     \"{s.get('snippet', '')[:200]}...\"")

    print(f"\n--- FINAL GROUNDED ANSWER ---")
    print(answer)
