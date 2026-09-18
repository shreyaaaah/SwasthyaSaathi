import os
from typing import List, Dict, Any, Tuple
from app.config import settings

try:
    from groq import Groq
except ImportError:
    Groq = None

SYSTEM_PROMPT = """You are SwasthyaSaathi, a trustworthy and empathetic Public Health AI assistant designed to provide accurate, grounded health information based on official guidelines from the Ministry of Health and Family Welfare (MoHFW), ICMR, NHP, and WHO.

STRICT GROUNDING RULES:
1. Answer the user's question ONLY using the facts provided in the CONTEXT below.
2. If the provided CONTEXT does not contain sufficient or relevant information to answer the question, state explicitly: "I don't have grounded information on that topic based on official public health documents." Do NOT attempt to invent or hallucinate information outside the context.
3. Cite your sources clearly within the response text (e.g. "[Source: Document Name - Section]").
4. Keep medical answers clear, structured, and easy for citizens to understand.
5. Provide a brief safety note to consult a medical professional for individual diagnosis or emergency care.

CONTEXT:
{context_text}
"""

def format_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "No relevant public health document context found."
    
    formatted_chunks = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        doc = chunk.get("doc_name", "Unknown Document")
        section = chunk.get("section", "General")
        text = chunk.get("chunk_text", "")
        formatted_chunks.append(f"--- Chunk [{idx}] | Document: {doc} | Section: {section} ---\n{text}\n")
    
    return "\n".join(formatted_chunks)

def generate_grounded_answer(query: str, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, bool]:
    """
    Generates grounded answer using Groq LLM API.
    Returns (answer_string, is_grounded_flag).
    """
    if not retrieved_chunks:
        return ("I don't have grounded information on that topic based on official public health documents.", False)
    
    context_str = format_context(retrieved_chunks)
    
    api_key = settings.GROQ_API_KEY
    if not api_key:
        # Fallback when Groq API Key is not set in environment
        doc_list = ", ".join(list(set([c.get("doc_name") for c in retrieved_chunks])))
        fallback_msg = (
            f"**[Grounded Health Information Retrieved]**\n\n"
            f"Based on official health guidelines ({doc_list}):\n\n"
        )
        for c in retrieved_chunks[:2]:
            fallback_msg += f"- **{c.get('section', 'Overview')}**: {c.get('chunk_text')[:250]}...\n\n"
        
        fallback_msg += "\n*(Note: Set `GROQ_API_KEY` in environment for complete LLM synthesized responses.)*"
        return (fallback_msg, True)

    try:
        client = Groq(api_key=api_key)
        
        system_content = SYSTEM_PROMPT.format(context_text=context_str)
        user_content = f"User Question: {query}"
        
        model_name = settings.GROQ_MODEL_NAME
        
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            model=model_name,
            temperature=0.2,
            max_tokens=800
        )
        
        answer = completion.choices[0].message.content
        return (answer, True)
    
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        # Fallback to structured chunk summary
        doc_list = ", ".join(list(set([c.get("doc_name") for c in retrieved_chunks])))
        err_msg = (
            f"Based on official health guidelines ({doc_list}):\n\n"
        )
        for c in retrieved_chunks[:2]:
            err_msg += f"• **{c.get('section', 'Guideline')}**: {c.get('chunk_text')}\n\n"
        err_msg += f"\n*(Disclaimer: Consult a medical professional for personal diagnosis.)*"
        return (err_msg, True)
