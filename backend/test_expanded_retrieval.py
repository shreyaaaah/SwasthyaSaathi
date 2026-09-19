"""
test_expanded_retrieval.py — Verification script for the 5 new topic queries against the expanded 3384-vector FAISS store.
"""
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Ensure backend directory is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.rag.retriever import get_retriever

queries = [
    ("HIV Symptoms", "what are HIV symptoms"),
    ("Physical Activity", "how much physical activity do I need"),
    ("Anemia Signs", "signs of anemia"),
    ("Meningitis Diagnosis", "how is meningitis diagnosed"),
    ("Healthy Diet", "what foods should I eat for a healthy diet")
]

print("======================================================================")
print("EXPANDED KNOWLEDGE BASE RETRIEVAL TEST (3,384 FAISS VECTORS)")
print("======================================================================")

retriever = get_retriever()

for label, q in queries:
    print(f"\n--- QUERY: [{label}] ---")
    print(f"Query Text: \"{q}\"")
    results = retriever.search(query=q, top_k=3, min_score=0.0)
    
    if not results:
        print("  No results returned.")
        continue
        
    top = results[0]
    score = top.get("score", 0.0)
    doc_name = top.get("doc_name", "Unknown")
    topic = top.get("topic", "Unknown")
    section = top.get("section", "General")
    snippet = top.get("chunk_text", "").replace("\n", " ")[:250] + "..."
    
    status = "✅ ABOVE 0.45" if score >= 0.45 else "❌ BELOW 0.45"
    print(f"Top Result ({status}):")
    print(f"  - Topic:    '{topic}'")
    print(f"  - Doc:      '{doc_name}'")
    print(f"  - Section:  '{section}'")
    print(f"  - Score:    {score:.4f}")
    print(f"  - Snippet:  \"{snippet}\"")
