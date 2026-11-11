"""Central content and tone rules for the public-health assistant."""

import re

HEALTH_ASSISTANT_SYSTEM = """You are SwasthyaSaathi, a warm public-health information companion. Speak plainly and naturally, matching the user's language and tone (including Hindi/Hinglish when appropriate). Lead with the useful answer. Do not diagnose.

MODE: {answer_mode}
USER-FACING RESTRICTIONS:
- Never mention a knowledge base, retrieved context, empty sources, prompts, internal tools, tiers, or system instructions.
- GROUNDED: use only facts supported by the provided official advisory passages; attribute them accurately.
- WEB_TRUSTED: use only the provided allowlisted page excerpts and identify that this is general information from trusted health websites.
- MIXED: clearly distinguish what the supplied guideline supports from general background. Do not attribute unsupported statements to a guideline.
- GENERAL: give a useful, concise general-health answer even when no official excerpt was retrieved; never refuse only because no source was returned. Cover common non-medicine care, warning signs, and when to seek care. State uncertainty plainly. Do not invent guideline claims, numeric statistics, lab values, doses, schedules, drug interactions, or treatment protocols. For infant/pregnancy/elderly specifics, direct the user to a clinician or pharmacist.
- Never provide medicine doses or schedules unless a supplied verified source explicitly supports the exact claim. Without such a source, explain briefly why a clinician/pharmacist should advise.
- For vague personal symptoms, first give safe immediate steps and warning signs, then ask at most ONE concise, specific follow-up question if needed. Never give a list or numbered series of questions. Put the question on its own final line. Keep the answer to a few short sentences or bullets, not a long paragraph, and lead with the answer.
- Avoid repeated boilerplate disclaimers; include a brief safety note only when the situation warrants it.
- Paraphrase sources; do not quote long passages.
"""

CONTENT_FENCE_VERIFIER = """Review the draft for health-content safety. Remove or soften claims that are not supported by the supplied sources and exceed the allowed general-health information scope. In particular remove unsupported doses, schedules, medicine recommendations, numbers/statistics, lab values, named guideline claims, or specific advice for infants/pregnancy/elderly. Preserve helpful general explanations, warning signs and care-seeking guidance. Never add new facts. Return only the cleaned response in the user's language."""

INTERNAL_LEAKS = (
    "provided reference sources", "retrieved context", "knowledge base", "sources are empty",
    "the sources are empty", "reference sources are empty", "no reference sources", "sources are unavailable",
    "which reference sources were you referring to", "internal tool", "system prompt",
)


def remove_internal_leaks(text: str) -> str:
    """Drop sentences that expose internal source plumbing while preserving useful guidance."""
    pieces = re.split(r"(?<=[.!?])\s+|\r?\n+", text.strip())
    kept = [piece.strip() for piece in pieces
            if piece.strip() and not any(phrase in piece.casefold() for phrase in INTERNAL_LEAKS)]
    if kept:
        return " ".join(kept)
    return "I can share general health information, but I can’t diagnose you. Seek care if symptoms are severe or worsening."


def limit_follow_up_questions(text: str, maximum: int = 1) -> str:
    """Keep no more than `maximum` question marks in one assistant turn."""
    pieces = re.split(r"(?<=\?)", text)
    question_indices = [i for i, piece in enumerate(pieces) if "?" in piece]
    if not question_indices:
        return text.strip()
    if maximum <= 0:
        return "".join(piece for piece in pieces if "?" not in piece).strip()

    generic_question = re.compile(
        r"tell me (?:a bit )?more|share (?:a bit )?more|please clarify|"
        r"clarify (?:what|how)|what are you experiencing|describe what you(?:'re| are) experiencing",
        re.I,
    )
    specific = [i for i in question_indices if not generic_question.search(pieces[i])]
    selected = set((specific or question_indices)[:maximum])
    kept = []
    selected_questions = []
    for i, piece in enumerate(pieces):
        if i in question_indices:
            if i not in selected:
                continue
            piece = re.sub(r"\s+\d+[.)]\s*", " ", piece, count=1)
            piece = re.sub(
                r",?\s+and\s+(?:has|have|is|are|does|do|did|can|could|will|would|should)\b[^?]*",
                "",
                piece,
                count=1,
                flags=re.I,
            )
            selected_questions.append(piece.strip())
        else:
            kept.append(piece)
    body = "".join(kept).strip()
    questions = " ".join(selected_questions)
    return f"{body}\n{questions}".strip() if questions else body
