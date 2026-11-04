"""Print a ranked shortlist of guidance topics to add from refusal aggregates."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from app.rag.coverage import load_uncovered_registry
from app.rag.gaps import ranked_gaps


def main() -> None:
    gaps = ranked_gaps()
    conditions = load_uncovered_registry()
    recommendations = []
    for row in gaps:
        topic = row["topic_text"].casefold()
        condition = next((item for item in conditions if item["id"] in topic or any(alias in topic for alias in item["aliases"])), None)
        recommendations.append({
            "rank": len(recommendations) + 1,
            "topic": row["topic_text"],
            "refusal_type": row["refusal_type"],
            "refusal_count": row["count"],
            "suggested_document_topic": condition["label"] if condition else row["topic_text"],
            "suggested_source": "Find current national MoHFW/NCDC/ICMR guidance, then WHO guidance; verify authority and publication date before ingestion.",
        })
    output = {"recommendations": recommendations}
    path = Path(__file__).with_name("documents_to_ingest_next.json")
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    for item in recommendations:
        print(f"{item['rank']:>3}. {item['topic']} — {item['refusal_count']} refusals; consider: {item['suggested_document_topic']}")
    print(f"Saved {len(recommendations)} recommendations to {path}")


if __name__ == "__main__":
    main()
