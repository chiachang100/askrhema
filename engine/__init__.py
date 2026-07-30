"""Engine module for TBS - contains search and LLM functionality."""

from engine.indexer import load_bible_data, get_verse_reference, validate_verse
from engine.hybrid_search import HybridSearchEngine, SearchResult
from engine.llm_provider import stream_llm_response, get_available_models

# Re-export exceptions
from engine.indexer import BibleDataError
from engine.llm_provider import LLMProviderError

__all__ = [
    "load_bible_data",
    "get_verse_reference",
    "validate_verse",
    "BibleDataError",
    "HybridSearchEngine",
    "SearchResult",
    "stream_llm_response",
    "get_available_models",
    "LLMProviderError",
]
