from typing import Dict, Any, List

def pre_retrieval_hook(query: str) -> Dict[str, Any]:
    """
    Extension Point: Pre-retrieval hook.
    Can be expanded to run myth-check validation, query classification, 
    or query expansion prior to FAISS retrieval.
    """
    return {
        "processed_query": query.strip(),
        "is_myth": False,
        "myth_correction": None,
        "bypass_rag": False
    }

def post_generation_hook(answer: str, retrieved_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extension Point: Post-generation hook.
    Can be expanded to attach medical triage urgency tags (e.g., EMERGENCY, 
    OPD_CONSULT, HOME_CARE) or safety verification.
    """
    return {
        "triage_tag": "GENERAL_HEALTH_INFO",
        "final_answer": answer
    }
