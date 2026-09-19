"""
test_fallback_direct.py — Runs run_agent directly for the 3 queries to inspect tool calls, fallback execution, and raw sources.
"""
import sys
import os
import json

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.agent.orchestrator import run_agent

test_queries = [
    ("1. TYPHOID FEVER SYMPTOMS", "what are the symptoms of typhoid fever"),
    ("2. NEWBORN VACCINES INDIA", "what vaccines does a newborn need in India"),
    ("3. CHIKUNGUNYA VS DENGUE", "how is chikungunya different from dengue")
]

print("======================================================================")
print("DIRECT ORCHESTRATOR FALLBACK TEST")
print("======================================================================")

for label, query in test_queries:
    print(f"\n{'='*75}")
    print(f"QUERY: [{label}]")
    print(f"User Question: \"{query}\"")
    print('='*75)

    res = run_agent(user_id="direct_test_user_01", query=query)
    
    print(f"\n--- TOOL TRACE ---")
    for t in res.get("tool_call_trace", []):
        print(f"  Tool: '{t['tool']}' ({t['duration_ms']} ms)")
        if t['tool'] == 'search_advisories':
            print(f"    -> search_advisories returned {len(t['result'])} items (>= 0.45 score)")
        elif t['tool'] == 'web_search_health_authority':
            print(f"    -> web_search_health_authority returned {len(t['result'])} live web items")

    print(f"\n--- SOURCES COLLECTED ({len(res.get('sources', []))}) ---")
    for s in res.get("sources", []):
        print(f"  [{s.get('source_type')}] {s.get('doc_name')} | URL: {s.get('source_url', 'N/A')}")

    print(f"\n--- FINAL ANSWER ---")
    print(res.get("answer"))
