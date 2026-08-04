"""Bible data loading, validation, and reference formatting."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError


class Verse(BaseModel):
    """Represents a Bible verse."""

    book: str
    chapter: int
    verse: int
    text: str
    # optional fields can be added with defaults


class BibleDataError(Exception):
    """Raised when Bible data is invalid or malformed."""

    pass


def load_bible_data(file_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate Bible verses from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise BibleDataError(f"Bible data file not found: {file_path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise BibleDataError(f"Invalid JSON in {file_path}: {e}") from e

    if not isinstance(data, list):
        raise BibleDataError("Bible data must be a JSON array")

    required_fields = {
        "id",
        "book",
        "chapter",
        "verse",
        "text",
        "testament",
        "category",
    }
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            raise BibleDataError(f"Record {i} is not a dictionary")
        missing = required_fields - record.keys()
        if missing:
            raise BibleDataError(f"Record {i} missing required fields: {missing}")
        # basic type checks
        if not isinstance(record["id"], int) or record["id"] < 0:
            raise BibleDataError(f"Record {i} has invalid 'id'")
        if not isinstance(record["book"], str) or not record["book"].strip():
            raise BibleDataError(f"Record {i} has empty 'book'")
        if not isinstance(record["chapter"], int) or record["chapter"] < 1:
            raise BibleDataError(f"Record {i} has invalid 'chapter'")
        if not isinstance(record["verse"], int) or record["verse"] < 1:
            raise BibleDataError(f"Record {i} has invalid 'verse'")
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise BibleDataError(f"Record {i} has empty 'text'")
        if record["testament"] not in ("OT", "NT"):
            raise BibleDataError(f"Record {i} 'testament' must be 'OT' or 'NT'")
        if not isinstance(record["category"], str) or not record["category"].strip():
            raise BibleDataError(f"Record {i} has empty 'category'")

    return data


def get_verse_reference(verse: dict[str, Any]) -> str:
    """Format a verse record into a human‑readable reference (e.g., 'John 3:16')."""
    return f"{verse['book']} {verse['chapter']}:{verse['verse']}"


def validate_verse(verse: dict[str, Any]) -> bool:
    """Return True if the verse dict contains all required fields with valid types."""
    try:
        Verse(**verse)
        return True
    except ValidationError:
        return False
