"""Unit tests for the engine modules."""

import pytest
import json
from pathlib import Path
from typing import List, Dict, Any

from engine.indexer import load_bible_data, get_verse_reference, validate_verse, BibleDataError
from engine.hybrid_search import HybridSearchEngine, SearchResult
from config import SearchConfig


class TestIndexer:
    """Test the indexer module."""
    
    def test_load_bible_data_valid(self, tmp_path):
        """Test loading valid Bible data."""
        test_data = [
            {
                "id": 1,
                "book": "Genesis",
                "chapter": 1,
                "verse": 1,
                "text": "In the beginning God created the heavens and the earth.",
                "testament": "OT",
                "category": "Creation"
            }
        ]
        
        test_file = tmp_path / "test_bible.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        verses = load_bible_data(test_file)
        assert len(verses) == 1
        assert verses[0]["book"] == "Genesis"
    
    def test_load_bible_data_missing_field(self, tmp_path):
        """Test loading invalid Bible data."""
        test_data = [
            {
                "id": 1,
                "book": "Genesis",
                "chapter": 1,
                "text": "In the beginning...",
                "testament": "OT",
                "category": "Creation"
            }
        ]
        
        test_file = tmp_path / "test_bible.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        with pytest.raises(BibleDataError):
            load_bible_data(test_file)
    
    def test_get_verse_reference(self):
        """Test verse reference formatting."""
        verse = {
            "book": "John",
            "chapter": 3,
            "verse": 16
        }
        reference = get_verse_reference(verse)
        assert reference == "John 3:16"


class TestHybridSearch:
    """Test the hybrid search engine."""
    
    @pytest.fixture
    def sample_verses(self):
        """Provide sample verses for testing."""
        return [
            {
                "id": 1,
                "book": "Genesis",
                "chapter": 1,
                "verse": 1,
                "text": "In the beginning God created the heavens and the earth.",
                "testament": "OT",
                "category": "Creation"
            },
            {
                "id": 2,
                "book": "Psalms",
                "chapter": 23,
                "verse": 1,
                "text": "The Lord is my shepherd; I shall not want.",
                "testament": "OT",
                "category": "Worship"
            },
            {
                "id": 3,
                "book": "John",
                "chapter": 3,
                "verse": 16,
                "text": "For God so loved the world that he gave his one and only Son.",
                "testament": "NT",
                "category": "Salvation"
            }
        ]
    
    def test_search_engine_initialization(self, sample_verses):
        """Test search engine initialization."""
        config = SearchConfig(top_k=2, vector_size=384)
        engine = HybridSearchEngine(config)
        engine.initialize(sample_verses)
    
    def test_search(self, sample_verses):
        """Test search functionality."""
        config = SearchConfig(top_k=2, vector_size=384)
        engine = HybridSearchEngine(config)
        engine.initialize(sample_verses)
        
        results = engine.search("God created", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)
    
    def test_search_with_filters(self, sample_verses):
        """Test search with filters."""
        config = SearchConfig(top_k=2, vector_size=384)
        engine = HybridSearchEngine(config)
        engine.initialize(sample_verses)
        
        results = engine.search("God", top_k=2, testament="NT")
        assert all(r.testament == "NT" for r in results)
    
    def test_search_fast_mode(self, sample_verses):
        """Test fast search mode."""
        config = SearchConfig(top_k=2, vector_size=384)
        engine = HybridSearchEngine(config)
        engine.initialize(sample_verses)
        
        results = engine.search_fast("shepherd", top_k=1)
        assert len(results) <= 1
        assert results[0].sparse_score == 0.0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
