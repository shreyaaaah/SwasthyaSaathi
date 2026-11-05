"""Fast multilingual rule-first intent routing for conversational chat."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Sequence


class Intent(StrEnum):
    GREETING = "GREETING"
    THANKS = "THANKS"
    GOODBYE = "GOODBYE"
    SMALLTALK = "SMALLTALK"
    HEALTH = "HEALTH"
    OFF_TOPIC = "OFF_TOPIC"
    UNSURE = "UNSURE"


_SOCIAL = {
    Intent.GREETING: re.compile(r"^(?:hi|hii+|hello|hey(?: there)?|namaste|namaskar|sat sri akal|salaam|good morning|good evening|good afternoon)(?:\s+(?:there|doctor|doc|ji))?[!.\s]*$", re.I),
    Intent.THANKS: re.compile(r"^(?:thanks?|thank you|thank u|thx|tnx|ty|shukriya|dhanyavaad|dhanyavad)(?:\s+(?:so much|a lot|bro|ji|dear))?[!.\s]*$", re.I),
    Intent.GOODBYE: re.compile(r"^(?:bye|goodbye|good bye|see you|see ya|take care|ok bye|okay bye|alvida|phir milte hain)(?:\s+.*)?[!.\s]*$", re.I),
    Intent.SMALLTALK: re.compile(r"^(?:how are you|how are u|what's up|whats up|kya haal hai|kaise ho|are you there|who are you|what can you do|ok|okay|cool|hmm)[?.!\s]*$", re.I),
}
_HEALTH_TERMS = re.compile(
    r"\b(?:health|symptom|illness|disease|sick|fever|temperature|bukhar|tapa|pain|dard|rash|itch|khujli|"
    r"cough|khansi|cold|flu|vomit|diarrh|dengue|malaria|chicken ?pox|measles|rabies|chikungunya|"
    r"leptospirosis|medicine|medication|dose|dosage|drug|tablet|pill|pregnan|baby|infant|child|"
    r"doctor|hospital|emergency|infection|blood pressure|diabetes|heart|chest|breath|saans|"
    r"lakshan|ilaaj|ilaj|dawai|dava|sehat|swasthya|bimari|bimaari|doctor|vaccine|vaccin|"
    r"symptom|treatment|prevent|prevention|first aid)\b", re.I
)
_EMERGENCY = re.compile(
    r"(?:i(?:'m| am)?\s+(?:having|getting|experiencing)\s+)?(?:severe\s+)?chest pain.{0,45}(?:sweat|breath|dizzy|faint|nausea)|"
    r"(?:trouble breathing|can't breathe|cannot breathe|severe bleeding|bleeding heavily|unconscious|seizure|"
    r"stroke signs|face (?:is )?drooping|slurred speech|sudden weakness|one-sided weakness|weakness on one side|"
    r"severe chest pain|suicid(?:e|al)|want to kill myself|took too many pills|overdose|"
    r"anaphylaxis|throat closing|swelling of (?:the )?tongue)", re.I
)


def classify_intent(query: str, history: Sequence[dict] | None = None) -> Intent:
    """Return high-confidence intent without an LLM call."""
    text = re.sub(r"\s+", " ", query.strip())
    for intent, pattern in _SOCIAL.items():
        if pattern.fullmatch(text):
            return intent
    if _HEALTH_TERMS.search(text):
        return Intent.HEALTH
    if history and text and re.fullmatch(r"(?:what about .+|and .+|how about .+|same for .+|aur .+|bachon ke liye|for children)\??", text, re.I):
        return Intent.HEALTH
    # Exact trivial off-topic patterns are handled warmly. Ambiguous questions
    # default to the conversational health route rather than a cold refusal.
    if re.search(r"\b(?:capital of|weather in|write code|python function|stock price|football score|recipe for|(?:write|create|make)\s+(?:a\s+)?(?:python|javascript|code|program)|sort function)\b", text, re.I):
        return Intent.OFF_TOPIC
    if len(text.split()) <= 2:
        return Intent.UNSURE
    return Intent.HEALTH


def is_immediate_emergency(query: str) -> bool:
    """Conservative deterministic red-flag check for urgent escalation."""
    return bool(_EMERGENCY.search(query))


def social_reply(intent: Intent, query: str) -> str:
    """Choose a short varied friendly response from a stable query hash."""
    from hashlib import sha256
    options = {
        Intent.GREETING: ["Namaste! What health question can I help with?", "Hi! What would you like to know?", "Hello! I’m here—what’s on your mind?"],
        Intent.THANKS: ["You’re welcome! Take care, and ask anytime if something else comes up.", "Anytime! Glad I could help.", "You’re very welcome. Take care!"],
        Intent.GOODBYE: ["Take care! I’m here if you need help later.", "Bye for now—wishing you well!", "See you. Take good care of yourself!"],
        Intent.SMALLTALK: ["I’m here and ready to help. What would you like to talk about?", "Doing well, thanks for asking. How can I help you today?"],
    }
    choices = options.get(intent, ["I’m here to help."])
    index = int(sha256(query.casefold().encode()).hexdigest()[:8], 16) % len(choices)
    return choices[index]

