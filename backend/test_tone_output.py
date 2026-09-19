"""
test_tone_output.py — Test script to collect new conversational responses across 4 key test queries.
"""
import sys
import requests
import json

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"
USER_ID = "tone_test_user_01"

test_queries = [
    ("1. CARDIAC EMERGENCY", "I have severe chest pain and can't breathe — when should I call emergency?"),
    ("2. FEVER SELF-CARE", "I have a high fever of 39 degrees, what should I do at home?"),
    ("3. DIABETES MANAGEMENT", "My blood sugar is high, how do I manage diabetes through diet and medication?"),
    ("4. MYTH CHECK", "Does turmeric cure diabetes? I heard it boosts immunity.")
]

for label, query in test_queries:
    print(f"\n{'='*70}")
    print(f"QUERY: {label}")
    print(f"User Question: \"{query}\"")
    print('-'*70)
    try:
        res = requests.post(f"{BASE_URL}/chat", json={"query": query, "user_id": USER_ID}, timeout=60)
        if res.status_code == 200:
            data = res.json()
            print(f"Triage: {data.get('triage_tag')} | Grounded: {data.get('is_grounded')} | Time: {data.get('response_time_ms')}ms")
            print(f"\n--- NEW CONVERSATIONAL ANSWER ---")
            print(data.get('answer', ''))
        else:
            print(f"HTTP Error {res.status_code}: {res.text[:300]}")
    except Exception as e:
        print(f"Request failed: {e}")
