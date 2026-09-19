"""
inspect_scores_v2.py — Re-tests cosine similarity score distribution after adding
3 new guideline documents (cardiac_emergency, fever_flu_diarrhea, hypertension_diabetes).

Queries specifically designed to test:
1. Chest pain / cardiac emergency  — must now hit cardiac_emergency_guidelines.txt
2. Fever / flu self-care           — must now hit fever_flu_diarrhea_selfcare.txt
3. Diabetes management             — must now hit hypertension_diabetes_management.txt
4. Dengue symptoms                 — existing baseline (should still hit dengue)
5. Turmeric/diabetes myth          — should still fall below 0.45 threshold
6. Out-of-domain                   — should still score near-zero
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.rag.retriever import get_retriever

queries = [
    ("CARDIAC: chest pain + shortness of breath", "I have severe chest pain and can't breathe — when should I call emergency?"),
    ("FEVER/FLU: high fever self-care",            "I have a high fever of 39 degrees, what should I do at home?"),
    ("DIABETES: blood sugar management",           "My blood sugar is high, how do I manage diabetes through diet and medication?"),
    ("DENGUE: dengue symptoms baseline",           "What are the warning signs of severe dengue that need hospital admission?"),
    ("MYTH: turmeric cures diabetes",              "Does turmeric cure diabetes? I heard it boosts immunity."),
    ("OUT-OF-DOMAIN: geography",                   "What is the capital of France?"),
]

retriever = get_retriever()
print("=" * 70)
print("RAG SCORE DISTRIBUTION REPORT v2 — 8-topic knowledge base")
print("=" * 70)

for label, query in queries:
    results = retriever.search(query, top_k=3, min_score=0.0)  # no floor for inspection
    print(f"\n[{label}]")
    print(f"  Query: '{query[:80]}'")
    if not results:
        print("  → No results returned at all")
    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        doc   = r.get("doc_name", "?")
        topic = r.get("topic", "?")
        above_floor = "✅ ABOVE 0.45" if score >= 0.45 else "❌ BELOW 0.45"
        print(f"  [{i}] score={score:.4f} {above_floor} | doc={doc} | topic={topic}")
