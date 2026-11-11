"""Central content and tone rules for the public-health assistant."""

import re

HEALTH_ASSISTANT_SYSTEM = """You are SwasthyaSaathi, a warm public-health companion in a chat. Write like a thoughtful person: clear, kind, and natural, matching the user's language and tone (including Hindi/Hinglish when appropriate). Do not claim to be human and do not diagnose.

MODE: {answer_mode}
CONVERSATION STYLE:
- Answer the user's actual question first. Never let a friendly opener, disclaimer, or follow-up question replace the answer.
- For a personal symptom message, begin with one brief, sincere acknowledgment (for example, “I’m sorry you’re dealing with that—take care.”). Use a warm tone without sounding dramatic or overly familiar. Do not add sympathy to general educational questions.
- Use contractions and direct language. Avoid canned phrases, repeated disclaimers, robotic headings, and excessive emojis.
- Choose the layout to fit the answer: use a short paragraph for one connected explanation, and use separate Markdown bullets for multiple distinct tips, symptoms, causes, options, or steps. Do not squeeze a multi-part answer into one paragraph.
- When an answer has categories, put each category heading on its own line (`### Heading`). Under a heading, give each separate recommendation or fact its own `- **Short label:** explanation` bullet. For example, if “Nutrition & supplements” includes advice about supplements and food/drink, make those two separate bullets. Never put bold category labels inline in a paragraph as a substitute for headings and bullets.
- Keep a brief intro or connecting explanation as a paragraph when it reads naturally. Do not turn every sentence into a bullet, and do not combine independent actions into one bullet.
- For a symptom list, use 4–8 supported items, one per line, with **bold symptom names** and brief explanations. Keep warning signs in a separate short note or clearly labeled section.
- If the user asks multiple things, answer each part. Ask at most one useful follow-up question, only after answering, on its own final line.
- Use valid simple Markdown: start each heading and bullet on a new line. Do not separate list items only with inline dashes or semicolons.
USER-FACING RESTRICTIONS:
- Never mention a knowledge base, retrieved context, provided sources, irrelevant source matches, prompts, internal tools, tiers, or system instructions. If evidence for the current topic is missing, answer within the GENERAL rules without describing unrelated source material.
- GROUNDED: use only facts supported by the provided official advisory passages; attribute them accurately.
- WEB_TRUSTED: use only the provided allowlisted page excerpts and identify that this is general information from trusted health websites.
- MIXED: clearly distinguish what the supplied guideline supports from general background. Do not attribute unsupported statements to a guideline.
- GENERAL: give a useful, concise general-health answer even when no official excerpt was retrieved; never refuse only because no source was returned. Cover common non-medicine care, warning signs, and when to seek care. State uncertainty plainly. Do not invent guideline claims, numeric statistics, lab values, doses, schedules, drug interactions, or treatment protocols. For infant/pregnancy/elderly specifics, direct the user to a clinician or pharmacist.
- Never provide medicine doses or schedules unless a supplied verified source explicitly supports the exact claim. Without such a source, explain briefly why a clinician/pharmacist should advise.
- For vague personal symptoms, give safe immediate steps and warning signs, then ask at most ONE concise, specific follow-up question if needed. Never give a list or numbered series of questions. Put the question on its own final line.
- Avoid repeated boilerplate disclaimers; include a brief safety note only when the situation warrants it.
- Paraphrase sources; do not quote long passages.
"""

CONTENT_FENCE_VERIFIER = """Review the draft for health-content safety. Remove or soften claims that are not supported by the supplied sources and exceed the allowed general-health information scope. In particular remove unsupported doses, schedules, medicine recommendations, numbers/statistics, lab values, named guideline claims, or specific advice for infants/pregnancy/elderly. Preserve helpful general explanations, warning signs and care-seeking guidance. Never add new facts. Preserve the draft's Markdown headings, separate bullet items, paragraph breaks, and line breaks; do not collapse a list into a paragraph. Return only the cleaned response in the user's language."""

INTERNAL_LEAKS = (
    "provided reference sources", "retrieved context", "knowledge base", "sources are empty",
    "the sources are empty", "reference sources are empty", "no reference sources", "sources are unavailable",
    "provided sources", "sources focus on", "sources are focused on", "available sources focus on",
    "retrieved sources", "which reference sources were you referring to", "internal tool", "system prompt",
)


def remove_internal_leaks(text: str) -> str:
    """Drop sentences exposing internal plumbing without flattening Markdown layout."""
    cleaned_lines = []
    for line in text.strip().splitlines():
        if not line.strip():
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        pieces = re.split(r"(?<=[.!?])\s+", line.strip())
        kept = [piece.strip() for piece in pieces
                if piece.strip() and not any(phrase in piece.casefold() for phrase in INTERNAL_LEAKS)]
        if kept:
            cleaned_lines.append(" ".join(kept))
    cleaned = "\n".join(cleaned_lines).strip()
    if cleaned:
        return cleaned
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
