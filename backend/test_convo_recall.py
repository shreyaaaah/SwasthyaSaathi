import requests
import json
import time
import sys

# Ensure UTF-8 output encoding for windows console
sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"
USER_ID = "convo_recall_user_102"

messages = [
    "I have had a cough for 3 days.",
    "is turmeric good for immunity",
    "What should I do about the cough I mentioned earlier?"
]

print("=== STARTING LIVE CONVERSATIONAL RECALL TEST ===")

for idx, msg in enumerate(messages, 1):
    print(f"\n--- MESSAGE {idx} ---")
    print(f"User: {msg}")
    start = time.time()
    res = requests.post(f"{BASE_URL}/chat", json={"query": msg, "user_id": USER_ID})
    elapsed = round((time.time() - start) * 1000, 2)
    
    if res.status_code == 200:
        data = res.json()
        print(f"Status: 200 OK ({elapsed} ms)")
        print(f"Triage: {data.get('triage_tag')}")
        print(f"Tools Used: {data.get('tools_used')}")
        print(f"Sources: {[s.get('doc_name') for s in data.get('sources', [])]}")
        print(f"Answer:\n{data.get('answer')}")
    else:
        print(f"Error {res.status_code}: {res.text}")

print("\n=== CONVERSATIONAL RECALL TEST COMPLETE ===")
