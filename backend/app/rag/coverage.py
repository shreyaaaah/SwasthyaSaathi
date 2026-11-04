"""Health-topic coverage lookup for the bounded advisory corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import settings


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def load_uncovered_registry(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load configured uncovered health conditions and their common aliases."""
    registry_path = Path(path or settings.UNCOVERED_CONDITIONS_FILE)
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return data["conditions"]


def build_covered_registry(metadata: list[dict[str, Any]]) -> set[str]:
    """Derive covered labels from indexed document names, topics and sections."""
    covered: set[str] = set()
    for item in metadata:
        for field in ("doc_name", "topic", "section"):
            value = item.get(field)
            if value:
                normalized = _normalise(str(value).rsplit(".", 1)[0])
                if normalized:
                    covered.add(normalized)
    return covered


def detect_uncovered_condition(query: str, conditions: list[dict[str, Any]] | None = None) -> dict[str, str] | None:
    """Return a configured uncovered condition mentioned explicitly in query."""
    text = f" {_normalise(query)} "
    entries = conditions if conditions is not None else load_uncovered_registry()
    for item in sorted(entries, key=lambda value: max(map(len, value.get("aliases", []))), reverse=True):
        for alias in item.get("aliases", []):
            token = _normalise(alias)
            if token and f" {token} " in text:
                return {"id": item["id"], "label": item["label"]}
    return None

