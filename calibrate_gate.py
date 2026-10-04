"""
calibrate_gate.py

Runs 8 in-KB and 8 out-of-KB queries through FAISS retrieval + cross-encoder reranking
to calibrate the specificity gate threshold (Max CE over all chunks above 0.45 FAISS floor).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.stdout.reconfigure(encoding='utf-8')

from app.rag.retriever import get_retriever
from sentence_transformers import CrossEncoder

CE_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
print(f"Loading cross-encoder: {CE_MODEL}")
ce = CrossEncoder(CE_MODEL)

retriever = get_retriever()

# --- 8 IN-KB and 8 OUT-KB Calibration queries ---
IN_KB = [
    ("IN", "what are the symptoms of dengue fever", ["dengue"]),
    ("IN", "how is tuberculosis treated", ["tuberculosis", "tb"]),
    ("IN", "what causes air pollution related illness", ["air pollution", "pollution"]),
    ("IN", "chest pain heart attack emergency", ["cardiac", "chest pain", "heart"]),
    ("IN", "symptoms of malaria", ["malaria"]),
    ("IN", "what vaccines does a newborn need in India", ["vaccine", "immunization", "bcg"]),
    ("IN", "how to manage hypertension and diabetes at home", ["hypertension", "diabetes"]),
    ("IN", "first aid and care during floods", ["flood"]),
]

OUT_KB = [
    ("OUT", "what is the capital of France", []),
    ("OUT", "how do I file income tax return in India", []),
    ("OUT", "recipe for butter chicken", []),
    ("OUT", "what is the treatment for leptospirosis", ["leptospirosis"]),
    ("OUT", "how is chikungunya different from dengue", ["chikungunya"]),
    ("OUT", "what are symptoms of rabies", ["rabies"]),
    ("OUT", "how to apply for a driving license in Delhi", []),
    ("OUT", "best mobile phones under 20000 rupees", []),
]

ALL = IN_KB + OUT_KB

print()
print(f"{'Label':<5} {'FAISS-Top1':>10} {'Max-CE':>10} {'KW-Hit':>8}  Query")
print("-" * 85)

rows = []
for label, query, keywords in ALL:
    raw = retriever.search(query, top_k=5, min_score=0.0)
    valid = [c for c in raw if c.get("score", 0.0) >= 0.45]
    if not valid:
        faiss_top1 = raw[0]['score'] if raw else 0.0
        max_ce = -10.0
        kw_hit = False
    else:
        faiss_top1 = valid[0]['score']
        # Max CE score across all valid chunks above floor
        pairs = [(query, c['chunk_text'][:512]) for c in valid]
        ce_scores = ce.predict(pairs)
        max_ce = float(max(ce_scores))
        
        top3_text = " ".join(r['chunk_text'].lower() for r in raw[:3])
        kw_hit = any(kw.lower() in top3_text for kw in keywords) if keywords else False

    kw_str = "YES" if kw_hit else ("N/A" if not keywords else "NO")
    print(f"{label:<5} {faiss_top1:>10.4f} {max_ce:>10.4f} {kw_str:>8}  {query}")
    rows.append((label, query, faiss_top1, max_ce, kw_hit, keywords))

print()
in_ce = [r[3] for r in rows if r[0] == "IN"]
out_ce = [r[3] for r in rows if r[0] == "OUT"]
in_faiss = [r[2] for r in rows if r[0] == "IN"]
out_faiss = [r[2] for r in rows if r[0] == "OUT"]

print(f"IN-KB  Max-CE min={min(in_ce):.4f}  max={max(in_ce):.4f}  mean={sum(in_ce)/len(in_ce):.4f}")
print(f"OUT-KB Max-CE min={min(out_ce):.4f}  max={max(out_ce):.4f}  mean={sum(out_ce)/len(out_ce):.4f}")

out_max_ce = max(out_ce)
in_min_ce = min(in_ce)
mid_ce = (out_max_ce + in_min_ce) / 2.0
print()
print(f"CALIBRATED MAX-CE THRESHOLD = {mid_ce:.4f} (Midpoint between OUT max={out_max_ce:.4f} and IN min={in_min_ce:.4f})")

