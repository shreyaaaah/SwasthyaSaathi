from app.rag.retriever import get_retriever

def test_retrieval():
    retriever = get_retriever()
    queries = [
        "What are the early warning signs of Dengue?",
        "What is the standard treatment regimen for Tuberculosis?",
        "What vaccines are scheduled at 6 weeks for infants?",
        "How do I write a C++ program?" # Ungrounded query test
    ]
    
    for q in queries:
        print(f"\n==========================================")
        print(f"QUERY: {q}")
        print(f"==========================================")
        results = retriever.search(q, top_k=2)
        if not results:
            print("--> No matching chunks found (Ungrounded query).")
        for idx, res in enumerate(results, 1):
            print(f"[{idx}] Score: {res.get('score')} | Doc: {res.get('doc_name')} | Section: {res.get('section')}")
            print(f"    Snippet: {res.get('chunk_text')[:150]}...\n")

if __name__ == "__main__":
    test_retrieval()
