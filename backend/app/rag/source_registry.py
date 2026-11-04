"""Validation helpers for the local health-document provenance registry."""

import hashlib
from pathlib import Path


class SourceRegistryError(ValueError):
    """Raised when local documents do not match their provenance registry."""


def read_source_registry(raw_docs_dir: str | Path) -> dict[str, dict[str, str]]:
    raw_docs_dir = Path(raw_docs_dir)
    registry_path = raw_docs_dir / "SOURCES.md"
    if not registry_path.is_file():
        raise SourceRegistryError(f"Source registry is missing: {registry_path}")

    entries = {}
    for line_number, line in enumerate(registry_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.startswith("|") or line.startswith("|---") or "| File |" in line:
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) < 6:
            raise SourceRegistryError(f"Malformed registry row at {registry_path}:{line_number}")
        name, publisher, url, review_date, sha256_prefix, notes = fields[:6]
        if name in entries:
            raise SourceRegistryError(f"Duplicate registry entry for {name}")
        if len(sha256_prefix) < 16:
            raise SourceRegistryError(f"SHA-256 prefix for {name} must be at least 16 characters")
        entries[name] = {
            "publisher": publisher,
            "url": url,
            "review_date": review_date,
            "sha256": sha256_prefix.lower(),
            "notes": notes,
        }
    return entries


def validate_source_registry(raw_docs_dir: str | Path) -> dict[str, dict[str, str]]:
    raw_docs_dir = Path(raw_docs_dir)
    entries = read_source_registry(raw_docs_dir)
    local_files = {
        path.name for path in raw_docs_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}
        and path.name != "SOURCES.md"
    }
    unregistered = sorted(local_files - entries.keys())
    missing = sorted(entries.keys() - local_files)
    if unregistered or missing:
        details = []
        if unregistered:
            details.append(f"unregistered files: {', '.join(unregistered)}")
        if missing:
            details.append(f"missing files: {', '.join(missing)}")
        raise SourceRegistryError("Source registry mismatch: " + "; ".join(details))

    mismatches = []
    for name, entry in entries.items():
        digest = hashlib.sha256((raw_docs_dir / name).read_bytes()).hexdigest()
        if not digest.startswith(entry["sha256"]):
            mismatches.append(f"{name} (expected {entry['sha256']}, got {digest[:16]})")
    if mismatches:
        raise SourceRegistryError("Source hash mismatch: " + "; ".join(mismatches))
    return entries
