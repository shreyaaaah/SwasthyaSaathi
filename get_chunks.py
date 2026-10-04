import sys
sys.path.insert(0, 'backend')
from app.rag.retriever import get_retriever

r = get_retriever()
results = r.search('how is chikungunya different from dengue', top_k=3, min_score=0.0)
for i, res in enumerate(results[:3]):
    score = res['score']
    doc = res['doc_name']
    section = res['section']
    text = res['chunk_text']
    print(f"=== CHUNK [{i+1}] score={score:.4f} doc={doc} section={section} ===")
    print(text)
    print()
