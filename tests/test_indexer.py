"""Tests for Bible data loading and formatting."""

import json
import tempfile
from pathlib import Path

import pytest

from engine.indexer import BibleDataError, get_verse_reference, load_bible_data


def test_load_bible_data_valid() -> None:
    """Test loading valid Bible data."""
    data = [
        {
            "id": 1,
            "book": "John",
            "chapter": 3,
            "verse": 16,
            "text": "For God so loved...",
            "testament": "NT",
            "category": "Salvation",
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        loaded = load_bible_data(f.name)
        assert loaded == data
    Path(f.name).unlink()


def test_load_bible_data_missing_field() -> None:
    """Test that missing required fields raise error."""
    data = [{"id": 1, "book": "John"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        with pytest.raises(BibleDataError) as exc:
            load_bible_data(f.name)
        assert "missing required fields" in str(exc.value)
    Path(f.name).unlink()


def test_get_verse_reference() -> None:
    """Test reference formatting."""
    verse = {"book": "Romans", "chapter": 8, "verse": 28}
    assert get_verse_reference(verse) == "Romans 8:28"
