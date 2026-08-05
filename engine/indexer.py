"""Bible data loading, validation, and reference formatting."""

import json
import sqlite3
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


def _load_bible_data_from_sqlite(path: Path) -> list[dict[str, Any]]:
    """Load Bible verses from a SQLite database."""
    if not path.exists():
        raise BibleDataError(f"SQLite database not found: {path}")

    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row  # enables dict-like access
        cursor = conn.cursor()

        # Get all verses
        cursor.execute("""
            SELECT id, book, chapter, verse, text, testament, category
            FROM verses
            ORDER BY id
        """)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            raise BibleDataError(f"No verses found in database: {path}")

        # Convert sqlite3.Row objects to regular dicts
        return [dict(row) for row in rows]

    except sqlite3.OperationalError as e:
        raise BibleDataError(f"SQLite error in {path}: {e}") from e


def _load_bible_data_from_json(path: Path) -> list[dict[str, Any]]:
    """Load and validate Bible verses from a JSON or SQLite file."""
    path = Path(path)
    if not path.exists():
        raise BibleDataError(f"Bible data file not found: {path}")

    # Check if the file is a SQLite database
    if path.suffix == ".db":
        return _load_bible_data_from_sqlite(path)

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise BibleDataError(f"Invalid JSON in {path}: {e}") from e

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


def load_bible_data(file_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate Bible verses from JSON or SQLite."""
    path = Path(file_path)

    if not path.exists():
        raise BibleDataError(f"Bible data file not found: {file_path}")

    # Delegate to the appropriate loader based on file extension
    if path.suffix.lower() == ".json":
        return _load_bible_data_from_json(path)
    elif path.suffix.lower() == ".db":
        return _load_bible_data_from_sqlite(path)
    else:
        raise BibleDataError(f"Unsupported file type: {path.suffix}")


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
