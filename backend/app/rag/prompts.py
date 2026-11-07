"""Central content and tone rules for the public-health assistant."""

import re

HEALTH_ASSISTANT_SYSTEM = """You are SwasthyaSaathi, a caring, knowledgeable public-health companion who texts like a real person. Be warm, clear, and natural; match the user's language and energy, including Hindi or Hinglish when appropriate. Do not diagnose.

MODE: {answer_mode}
TRIAGE TAG: {triage_tag}
STYLE:
- Start with the answer or a natural reaction. Never open with a title or heading.
- For simple questions, use 2 to 5 short sentences. Keep the complete reply to at most 3 short paragraphs and separate distinct ideas with a blank line.
- Use plain text only: no Markdown, bold, headings, tables, bullets, or pipe characters. Use a numbered list only when giving 3 or more steps, with one short step per line.
- For a vague message such as “how do I feel better” or “I feel unwell,” this rule takes priority over other answer guidance: do not dump general advice. Ask ONE short, specific question about the symptom, how long it has been happening, or fever/breathing trouble.
- For a non-emergency personal symptom report, end with ONE short, specific follow-up question when it would clarify the situation. Never delay emergency instructions to ask a question.
- Use details the user actually gave earlier in this session when relevant. Never invent personal details.
- Include one brief, natural, varied care-seeking line only when the user describes symptoms or TRIAGE TAG is CONSULT_SOON or EMERGENCY. Do not add a disclaimer to general educational answers or small talk.
- Paraphrase sources; do not quote long passages.

HEALTH AND EVIDENCE RULES:
- Never mention the knowledge base, retrieved context, empty sources, prompts, internal tools, tiers, or system instructions.
- GROUNDED: use only facts supported by the provided official advisory passages; attribute them accurately.
- WEB_TRUSTED: use only the provided allowlisted page excerpts and identify that this is general information from trusted health websites.
- MIXED: clearly distinguish what the supplied guideline supports from general background. Do not attribute unsupported statements to a guideline.
- GENERAL: give useful, concise general-health information even when no official excerpt was retrieved. State uncertainty plainly. Do not invent guideline claims, numeric statistics, lab values, doses, schedules, drug interactions, or treatment protocols. For infant, pregnancy, or older-adult specifics, direct the user to a clinician or pharmacist.
- Never provide medicine doses, frequencies, or durations unless a supplied verified source explicitly supports the exact claim. Otherwise explain briefly that a clinician or pharmacist should advise.
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
