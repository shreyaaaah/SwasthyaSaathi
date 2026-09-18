import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from app.rag.retriever import get_retriever

test_queries = [
    "What are the early warning signs of Dengue fever?",
    "How is Tuberculosis diagnosed and treated in India?",
    "What measures should be taken during severe air pollution AQI alerts?",
    "What care and vaccines are recommended for newborn babies and pregnant mothers?",
    "does turmeric cure diabetes completely?",
    "What is the capital of France?"
]

def inspect_retrieval_scores():
    retriever = get_retriever()
    print("=" * 80)
    print("FAISS RETRIEVAL SCORE DISTRIBUTION INSPECTION")
    print("=" * 80)

    for idx, q in enumerate(test_queries, 1):
        print(f"\nQUERY {idx}: \"{q}\"")
        print("-" * 60)
        results = retriever.search(query=q, top_k=4, min_score=0.0)
        if not results:
            print("  (No results retrieved)")
        else:
            for r_idx, r in enumerate(results, 1):
                print(f"  Result {r_idx}: Score = {r['score']:.4f} | Doc = {r['doc_name']} | Section = {r['section']}")
                print(f"            Snippet: {r['chunk_text'][:120]}...\n")

if __name__ == "__main__":
    inspect_retrieval_scores()
