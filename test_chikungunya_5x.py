"""
test_chikungunya_5x.py
Run 'how is chikungunya different from dengue' 5 times.
Report: fallback_fired per run.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.stdout.reconfigure(encoding='utf-8')

from app.agent.orchestrator import run_agent

QUERY = "how is chikungunya different from dengue"
N = 5

fallback_count = 0
for i in range(1, N + 1):
    print(f"\n{'='*60}")
    print(f"RUN {i}/5: {QUERY}")
    print('='*60)
    result = run_agent(user_id=f"chik_test_{i}", query=QUERY)
    fired = result.get("fallback_fired", False)
    if fired:
        fallback_count += 1
    print(f"[RUN {i}] fallback_fired = {fired}")

print(f"\n{'='*60}")
print(f"SUMMARY: fallback fired {fallback_count}/{N} times")
print('='*60)
