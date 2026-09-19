"""
ui_verify_grounding.py — Tests both the "no strong match" and emergency cardiac cases
by calling /chat directly and printing the exact JSON returned.
This validates the backend side of the grounding fallback.
"""
import sys, requests, json
sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"
USER_ID  = "ui_verify_user_01"

tests = [
    ("OUT-OF-DOMAIN (capital of France)",   "What is the capital of France?"),
    ("CARDIAC EMERGENCY (chest pain)",       "I have severe chest pain and can't breathe"),
]

for label, query in tests:
    print(f"\n{'='*68}")
    print(f"TEST: {label}")
    print(f"Query: {query}")
    print('='*68)
    res = requests.post(f"{BASE_URL}/chat", json={"query": query, "user_id": USER_ID}, timeout=60)
    if res.status_code != 200:
        print(f"ERROR {res.status_code}: {res.text[:400]}")
        continue
    data = res.json()
    print(f"triage_tag:   {data.get('triage_tag')}")
    print(f"is_grounded:  {data.get('is_grounded')}")
    print(f"tools_used:   {data.get('tools_used')}")
    sources = data.get('sources', [])
    print(f"sources ({len(sources)} total):")
    for s in sources:
        print(f"  - {s.get('doc_name')} | section={s.get('section')} | score={s.get('score')}")
    print(f"\nAnswer (first 500 chars):\n{data.get('answer','')[:500]}")
    print(f"\nFull JSON payload:")
    # Print full response except answer to keep output manageable
    summary = {k: v for k, v in data.items() if k != 'answer'}
    summary['answer_length'] = len(data.get('answer', ''))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
