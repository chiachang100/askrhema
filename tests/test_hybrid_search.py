"""Tests for hybrid search engine."""

import pytest

from config import EmbeddingConfig, SearchConfig
from engine.hybrid_search import HybridSearchEngine, SearchResult
from engine.indexer import load_bible_data


@pytest.fixture
def bible_data() -> list[dict]:
    """Load sample data for testing."""
    return load_bible_data("data/sample_bible.json")


@pytest.fixture
def search_engine(bible_data: list[dict]) -> HybridSearchEngine:
    """Create a HybridSearchEngine with sample data."""
    search_config = SearchConfig(top_k=3)
    embedding_config = EmbeddingConfig()
    return HybridSearchEngine(bible_data, search_config, embedding_config)


def test_search_results_have_scores(search_engine: HybridSearchEngine) -> None:
    """Check that search returns results with RRF scores."""
    results = search_engine.search("love")
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)
    # Scores should be positive
    assert all(r.rrf_score > 0 for r in results)


def test_search_respects_top_k(search_engine: HybridSearchEngine) -> None:
    """Check that top_k limits results."""
    results = search_engine.search("God", top_k=2)
    assert len(results) <= 2


def test_search_with_filters(search_engine: HybridSearchEngine) -> None:
    """Test book and testament filters."""
    results = search_engine.search("God", book_filter="John")
    assert all(r.verse["book"] == "John" for r in results)

    results = search_engine.search("God", testament_filter="OT")
    assert all(r.verse["testament"] == "OT" for r in results)


def test_rrf_score_ordering(search_engine: HybridSearchEngine) -> None:
    """Check that results are sorted by RRF score descending."""
    results = search_engine.search("God")
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)