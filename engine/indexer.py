"""Bible data loading and validation module."""

import json
from pathlib import Path
from typing import Any, Dict, List, Union


class BibleDataError(Exception):
    """Exception raised for errors in Bible data."""
    pass


def validate_verse(verse: Dict[str, Any]) -> bool:
    """
    Validate that a verse dictionary contains all required fields.
    
    Args:
        verse: Dictionary representing a Bible verse
        
    Returns:
        True if valid
        
    Raises:
        BibleDataError: If validation fails
    """
    required_fields = ["id", "book", "chapter", "verse", "text", "testament", "category"]
    
    for field in required_fields:
        if field not in verse:
            raise BibleDataError(f"Missing required field: {field}")
    
    if not isinstance(verse["id"], int):
        raise BibleDataError(f"id must be an integer, got {type(verse['id'])}")
    
    if not isinstance(verse["book"], str) or not verse["book"].strip():
        raise BibleDataError("book must be a non-empty string")
    
    if not isinstance(verse["chapter"], int) or verse["chapter"] <= 0:
        raise BibleDataError("chapter must be a positive integer")
    
    if not isinstance(verse["verse"], int) or verse["verse"] <= 0:
        raise BibleDataError("verse must be a positive integer")
    
    if not isinstance(verse["text"], str) or not verse["text"].strip():
        raise BibleDataError("text must be a non-empty string")
    
    if verse["testament"] not in ["OT", "NT"]:
        raise BibleDataError(f"testament must be 'OT' or 'NT', got {verse['testament']}")
    
    if not isinstance(verse["category"], str) or not verse["category"].strip():
        raise BibleDataError("category must be a non-empty string")
    
    return True


def get_verse_reference(verse: Dict[str, Any]) -> str:
    """
    Format a verse reference as Book Chapter:Verse.
    
    Args:
        verse: Dictionary containing book, chapter, and verse keys
        
    Returns:
        Formatted reference string
    """
    return f"{verse['book']} {verse['chapter']}:{verse['verse']}"


def load_bible_data(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load and validate Bible data from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing Bible verses
        
    Returns:
        List of validated verse dictionaries
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        BibleDataError: If the data validation fails
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Bible data file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in {file_path}: {e.msg}", e.doc, e.pos)
    
    if not isinstance(data, list):
        raise BibleDataError(f"Expected JSON array, got {type(data).__name__}")
    
    if not data:
        raise BibleDataError("Bible data array is empty")
    
    validated_verses: List[Dict[str, Any]] = []
    ids_seen = set()
    
    for i, verse in enumerate(data):
        try:
            validate_verse(verse)
            
            # Check for duplicate IDs
            verse_id = verse["id"]
            if verse_id in ids_seen:
                raise BibleDataError(f"Duplicate verse ID: {verse_id}")
            ids_seen.add(verse_id)
            
            validated_verses.append(verse)
            
        except BibleDataError as e:
            raise BibleDataError(f"Error validating verse at index {i}: {e}")
    
    return validated_verses