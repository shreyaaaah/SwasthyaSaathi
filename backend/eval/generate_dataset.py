"""Build a deterministic >=300-query labeled gate evaluation set."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "vector_store" / "chunk_metadata.json"
UNCOVERED = ROOT / "data" / "uncovered_conditions.json"
OUTPUT = Path(__file__).with_name("queries.jsonl")
SEED = 20261104

IN_TEMPLATES = [
    "What does the guidance say about {topic}?", "How can I prevent {topic}?",
    "What are the warning signs of {topic}?", "{topic} ke baare mein kya guidance hai?",
    "{topic} me kaunsi savdhani rakhen?", "Mujhe {topic} ke lakshan samjhao.",
    "Can you explain {topic} in simple words?", "What should families know about {topic}?",
]
GENERAL = [
    "How do I file income tax in India?", "What is the capital of France?", "Give me a butter chicken recipe.",
    "Which phone should I buy under 20000 rupees?", "Explain how to fix a leaking kitchen tap.",
    "How do I renew my driving licence?", "Write a Python program to sort a list.",
    "What are the best mutual funds this year?", "How can I book a train ticket?", "Tell me a cricket joke.",
    "How do I make sourdough bread?", "What is the weather in Mumbai?", "Help me draft a job application.",
    "How to repair a laptop screen?", "What are GST input credit rules?", "Suggest a weekend trip near Pune.",
    "How do I create a pivot table?", "Explain photosynthesis for class 6.", "How to change a car tyre?",
    "Which movie won the most awards?", "How can I invest in gold?", "How to apply for a passport?",
    "Suggest a vegetarian dinner recipe.", "How do I configure a Wi-Fi router?", "What is the latest football score?",
    "Can you write a birthday poem?", "How do I remove a coffee stain?", "What are the rules of chess?",
    "How can I start an online shop?", "Convert 25 Celsius to Fahrenheit.",
]
SMALLTALK = [
    "hi", "hello", "hey there", "namaste", "sat sri akal", "thanks", "thank u so much",
    "tnx bro", "ok bye", "good morning", "how are you?", "you are helpful", "see you later",
    "shukriya", "dhanyavaad", "theek hai", "okay cool", "good night", "bye bye", "what's up?",
]


def build_dataset(metadata: list[dict], uncovered: list[dict], seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    sections: dict[tuple[str, str, str], dict] = {}
    for item in metadata:
        topic = str(item.get("topic") or Path(str(item.get("doc_name", "health"))).stem).replace("_", " ")
        section = str(item.get("section") or topic)
        key = (str(item.get("doc_name", "")), topic, section)
        sections.setdefault(key, item)
    in_rows: list[dict] = []
    section_rows = list(sections.items())
    rng.shuffle(section_rows)
    for (doc, topic, section), _item in section_rows:
        templates = rng.sample(IN_TEMPLATES, 2)
        for template in templates:
            in_rows.append({"query": template.format(topic=section if section != "General Overview" else topic),
                            "label": "in_kb", "target_doc": doc, "target_topic": topic,
                            "target_terms": topic.split(), "language_style": "hinglish" if any(x in template for x in ("ke ", "me ", "samjhao")) else "english"})
    while len(in_rows) < 150:
        doc, topic, section = rng.choice(list(sections))
        template = rng.choice(IN_TEMPLATES)
        in_rows.append({"query": template.format(topic=section if section != "General Overview" else topic),
                        "label": "in_kb", "target_doc": doc, "target_topic": topic,
                        "target_terms": topic.split(), "language_style": "hinglish" if "ke " in template or "me " in template else "english"})
    rng.shuffle(in_rows)
    in_rows = in_rows[:150]
    variants = ("please", "in simple steps", "for a beginner")
    general_rows = [{"query": f"{q} {variants[idx % len(variants)]}", "label": "general_out_of_domain", "target_doc": None, "target_topic": None,
                     "target_terms": [], "language_style": "english"} for idx, q in enumerate(GENERAL * 3)]
    social_rows = [{"query": q, "label": "smalltalk", "target_doc": None, "target_topic": None,
                    "target_terms": [], "language_style": "mixed"} for q in SMALLTALK * 2]
    near_templates = ["What is the treatment for {condition}?", "What are symptoms of {condition}?",
                      "{condition} ke lakshan kya hain?", "How can I prevent {condition}?"]
    near_rows = []
    for condition in uncovered:
        for alias in condition["aliases"][:1]:
            for template in near_templates:
                near_rows.append({"query": template.format(condition=alias), "label": "near_domain_uncovered",
                                  "target_doc": None, "target_topic": condition["id"],
                                  "target_terms": condition["aliases"],
                                  "language_style": "hinglish" if "ke " in template else "english"})
    while len(near_rows) < 150:
        condition = rng.choice(uncovered)
        alias = condition["aliases"][0]
        template = rng.choice(near_templates)
        near_rows.append({"query": f"{template.format(condition=alias)} please explain", "label": "near_domain_uncovered",
                          "target_doc": None, "target_topic": condition["id"], "target_terms": condition["aliases"],
                          "language_style": "hinglish" if "ke " in template else "english"})
    rng.shuffle(near_rows)
    near_rows = near_rows[:150]
    rows = in_rows + general_rows + near_rows + social_rows
    rng.shuffle(rows)
    # Stratified deterministic split, so no class disappears from either partition.
    for label in sorted({row["label"] for row in rows}):
        group = [row for row in rows if row["label"] == label]
        rng.shuffle(group)
        calibration_count = int(len(group) * 0.70)
        for row in group[:calibration_count]:
            row["split"] = "calibration"
        for row in group[calibration_count:]:
            row["split"] = "test"
    for idx, row in enumerate(rows):
        row["query_id"] = f"Q{idx + 1:04d}"
        row["seed"] = seed
    return rows


def main() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    conditions = json.loads(UNCOVERED.read_text(encoding="utf-8"))["conditions"]
    rows = build_dataset(metadata, conditions)
    OUTPUT.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} labeled queries to {OUTPUT}")
    print("By class:", {label: sum(row["label"] == label for row in rows) for label in sorted({r['label'] for r in rows})})
    print("By split:", {split: sum(row["split"] == split for row in rows) for split in ("calibration", "test")})


if __name__ == "__main__":
    main()
