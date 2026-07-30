'll create the complete TBS (Tuixiu Bible Search) codebase with all the specified files and functionality. Let me start by generating each file in the required structure.

python
# pyproject.toml
[project]
name = "tuixiu-bible-search"
version = "1.0.0"
description = "TBS (Tuixiu Bible Search) - A hybrid, local-first Bible search and exegesis tool"
readme = "README.md"
requires-python = ">=3.14"
authors = [
    {name = "TBS Team", email = "tbs@example.com"}
]
license = {text = "MIT"}
classifiers = [
    "Programming Language :: Python :: 3.14",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

dependencies = [
    "streamlit>=1.40.0",
    "qdrant-client>=1.12.0",
    "sentence-transformers>=3.0.0",
    "rank-bm25>=0.2.2",
    "google-genai>=0.1.0",
    "openai>=1.50.0",
    "httpx>=0.27.0",
    "pydantic>=2.9.0",
    "numpy>=1.26.0",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "tokenizers>=0.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100
target-version = ['py314']

[tool.ruff]
line-length = 100
target-version = "py314"
select = ["E", "F", "W", "I", "N", "D", "UP"]
ignore = ["D100", "D104", "D107", "D203", "D212"]

[tool.mypy]
python_version = "3.14"
strict = true
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = false
disallow_untyped_defs = true
disallow_any_unimported = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
python
# data/sample_bible.json
[
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
    "book": "Genesis",
    "chapter": 1,
    "verse": 27,
    "text": "So God created mankind in his own image, in the image of God he created them; male and female he created them.",
    "testament": "OT",
    "category": "Creation"
  },
  {
    "id": 3,
    "book": "Psalms",
    "chapter": 23,
    "verse": 1,
    "text": "The Lord is my shepherd; I shall not want.",
    "testament": "OT",
    "category": "Worship"
  },
  {
    "id": 4,
    "book": "Psalms",
    "chapter": 23,
    "verse": 4,
    "text": "Yea, though I walk through the valley of the shadow of death, I will fear no evil: for thou art with me; thy rod and thy staff they comfort me.",
    "testament": "OT",
    "category": "Comfort"
  },
  {
    "id": 5,
    "book": "Isaiah",
    "chapter": 40,
    "verse": 31,
    "text": "But they that wait upon the Lord shall renew their strength; they shall mount up with wings as eagles; they shall run, and not be weary; and they shall walk, and not faint.",
    "testament": "OT",
    "category": "Prophecy"
  },
  {
    "id": 6,
    "book": "Isaiah",
    "chapter": 53,
    "verse": 5,
    "text": "But he was wounded for our transgressions, he was bruised for our iniquities: the chastisement of our peace was upon him; and with his stripes we are healed.",
    "testament": "OT",
    "category": "Prophecy"
  },
  {
    "id": 7,
    "book": "Matthew",
    "chapter": 5,
    "verse": 3,
    "text": "Blessed are the poor in spirit: for theirs is the kingdom of heaven.",
    "testament": "NT",
    "category": "Sermon on the Mount"
  },
  {
    "id": 8,
    "book": "Matthew",
    "chapter": 5,
    "verse": 8,
    "text": "Blessed are the pure in heart: for they shall see God.",
    "testament": "NT",
    "category": "Sermon on the Mount"
  },
  {
    "id": 9,
    "book": "John",
    "chapter": 3,
    "verse": 16,
    "text": "For God so loved the world that he gave his one and only Son, that whoever believes in him shall not perish but have eternal life.",
    "testament": "NT",
    "category": "Salvation"
  },
  {
    "id": 10,
    "book": "John",
    "chapter": 14,
    "verse": 6,
    "text": "Jesus answered, I am the way and the truth and the life. No one comes to the Father except through me.",
    "testament": "NT",
    "category": "Salvation"
  },
  {
    "id": 11,
    "book": "Romans",
    "chapter": 8,
    "verse": 28,
    "text": "And we know that in all things God works for the good of those who love him, who have been called according to his purpose.",
    "testament": "NT",
    "category": "Faith"
  },
  {
    "id": 12,
    "book": "Romans",
    "chapter": 12,
    "verse": 2,
    "text": "Do not conform to the pattern of this world, but be transformed by the renewing of your mind. Then you will be able to test and approve what God's will is—his good, pleasing and perfect will.",
    "testament": "NT",
    "category": "Faith"
  }
]
python
# config.py
"""Configuration module for TBS (Tuixiu Bible Search)."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchConfig:
    """Configuration for search parameters."""
    top_k: int = 5
    rrf_k_constant: int = 60
    vector_size: int = 384
    collection_name: str = "bible_verses"
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.rrf_k_constant < 1:
            raise ValueError("rrf_k_constant must be at least 1")
        if self.vector_size <= 0:
            raise ValueError("vector_size must be positive")


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    system_prompt: str = (
        "You are TBS (Tuixiu Bible Search), an expert biblical assistant. "
        "Ground your answer strictly in the provided Bible passages. "
        "Always cite the Book, Chapter, and Verse for every passage reference."
    )
    temperature: float = 0.7
    max_tokens: int = 1000
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_default_model: str = "llama2"
    google_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError("device must be one of: cpu, cuda, mps")


# Default configuration instances
DEFAULT_SEARCH_CONFIG = SearchConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()


def get_system_prompt() -> str:
    """Get the default system prompt for AI exegesis."""
    return DEFAULT_LLM_CONFIG.system_prompt


__all__ = [
    "SearchConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "DEFAULT_SEARCH_CONFIG",
    "DEFAULT_LLM_CONFIG",
    "DEFAULT_EMBEDDING_CONFIG",
    "get_system_prompt",
]
python
# engine/__init__.py
"""Engine module for TBS - contains search and LLM functionality."""

from engine.indexer import load_bible_data, get_verse_reference, validate_verse
from engine.hybrid_search import HybridSearchEngine, SearchResult
from engine.llm_provider import stream_llm_response, get_available_models

__all__ = [
    "load_bible_data",
    "get_verse_reference",
    "validate_verse",
    "HybridSearchEngine",
    "SearchResult",
    "stream_llm_response",
    "get_available_models",
]
python
# engine/indexer.py
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
python
# engine/hybrid_search.py
"""Hybrid search engine combining dense vector search and sparse BM25 search."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import math

from config import DEFAULT_SEARCH_CONFIG, DEFAULT_EMBEDDING_CONFIG, SearchConfig


@dataclass
class SearchResult:
    """Represents a search result with metadata and scores."""
    id: int
    book: str
    chapter: int
    verse: int
    text: str
    testament: str
    category: str
    dense_score: float
    sparse_score: float
    fused_score: float
    dense_rank: int
    sparse_rank: int
    
    def get_reference(self) -> str:
        """Get the formatted reference for this verse."""
        return f"{self.book} {self.chapter}:{self.verse}"


class HybridSearchEngine:
    """
    Hybrid search engine combining dense vector search with BM25 sparse search.
    Uses RRF (Reciprocal Rank Fusion) for combining results.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None) -> None:
        """
        Initialize the hybrid search engine.
        
        Args:
            config: Search configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_SEARCH_CONFIG
        self._embedding_model: Optional[SentenceTransformer] = None
        self._qdrant_client: Optional[QdrantClient] = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._verses: Optional[List[Dict[str, Any]]] = None
        self._is_initialized: bool = False
        
        # Tokenized corpora for BM25
        self._tokenized_corpus: List[List[str]] = []
        self._verse_id_to_index: Dict[int, int] = {}
        self._index_to_verse_id: Dict[int, int] = {}
    
    @st.cache_resource
    def _get_embedding_model(_self) -> SentenceTransformer:
        """Get or load the sentence transformer model with caching."""
        return SentenceTransformer(DEFAULT_EMBEDDING_CONFIG.model_name)
    
    @st.cache_resource
    def _get_qdrant_client(_self) -> QdrantClient:
        """Get or create the Qdrant in-memory client with caching."""
        return QdrantClient(":memory:")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        # Simple tokenization - split on whitespace and punctuation
        return text.lower().split()
    
    def initialize(self, verses: List[Dict[str, Any]]) -> None:
        """
        Initialize the search engine with Bible data.
        
        Args:
            verses: List of verse dictionaries
        """
        if self._is_initialized:
            return
            
        self._verses = verses
        
        # Get cached resources
        self._embedding_model = self._get_embedding_model()
        self._qdrant_client = self._get_qdrant_client()
        
        # Prepare data for indexing
        texts = [verse["text"] for verse in verses]
        self._tokenized_corpus = [self._tokenize_text(text) for text in texts]
        
        # Build index mappings
        for idx, verse in enumerate(verses):
            verse_id = verse["id"]
            self._verse_id_to_index[verse_id] = idx
            self._index_to_verse_id[idx] = verse_id
        
        # Create BM25 index
        self._bm25_index = BM25Okapi(self._tokenized_corpus)
        
        # Create Qdrant collection
        self._create_qdrant_collection(verses)
        
        self._is_initialized = True
    
    def _create_qdrant_collection(self, verses: List[Dict[str, Any]]) -> None:
        """Create Qdrant collection and index verses."""
        collection_name = self.config.collection_name
        vector_size = self.config.vector_size
        
        # Recreate collection if it exists
        collections = self._qdrant_client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            self._qdrant_client.delete_collection(collection_name)
        
        # Create collection
        self._qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
        # Index each verse
        points = []
        for i, verse in enumerate(verses):
            # Generate embedding
            embedding = self._embedding_model.encode(verse["text"])
            
            # Create point
            point = PointStruct(
                id=verse["id"],
                vector=embedding.tolist(),
                payload={
                    "book": verse["book"],
                    "chapter": verse["chapter"],
                    "verse": verse["verse"],
                    "text": verse["text"],
                    "testament": verse["testament"],
                    "category": verse["category"]
                }
            )
            points.append(point)
        
        # Upload points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._qdrant_client.upsert(
                collection_name=collection_name,
                points=batch
            )
    
    def _build_filter(self, book: Optional[str] = None, testament: Optional[str] = None) -> Optional[Filter]:
        """Build a Qdrant filter based on book and testament filters."""
        conditions = []
        
        if book:
            conditions.append(
                FieldCondition(
                    key="book",
                    match=MatchValue(value=book)
                )
            )
        
        if testament:
            conditions.append(
                FieldCondition(
                    key="testament",
                    match=MatchValue(value=testament)
                )
            )
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    def _perform_dense_search(self, query: str, top_k: int, 
                             book: Optional[str] = None, 
                             testament: Optional[str] = None) -> List[Tuple[int, float, int]]:
        """Perform dense vector search using Qdrant."""
        if not self._is_initialized:
            raise RuntimeError("Search engine not initialized")
        
        # Generate query embedding
        query_embedding = self._embedding_model.encode(query).tolist()
        
        # Build filter
        qdrant_filter = self._build_filter(book, testament)
        
        # Perform search
        search_results = self._qdrant_client.search(
            collection_name=self.config.collection_name,
            query_vector=query_embedding,
            limit=top_k * 2,  # Get more results for better fusion
            query_filter=qdrant_filter
        )
        
        # Extract results
        results = []
        for rank, hit in enumerate(search_results):
            verse_id = hit.id
            score = hit.score
            results.append((verse_id, score, rank + 1))
        
        return results
    
    def _perform_sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float, int]]:
        """Perform sparse BM25 search."""
        if not self._is_initialized or self._bm25_index is None:
            raise RuntimeError("Search engine not initialized")
        
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        
        # Get BM25 scores
        scores = self._bm25_index.get_scores(query_tokens)
        
        # Get top_k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices):
            score = scores[idx]
            if score > 0:  # Only include positive scores
                verse_id = self._index_to_verse_id[idx]
                results.append((verse_id, float(score), rank + 1))
        
        return results
    
    def search(self, query: str, top_k: Optional[int] = None,
               book: Optional[str] = None,
               testament: Optional[str] = None) -> List[SearchResult]:
        """
        Perform hybrid search using RRF fusion.
        
        Args:
            query: Search query string
            top_k: Number of results to return (uses config if None)
            book: Filter by book name
            testament: Filter by testament (OT or NT)
            
        Returns:
            List of SearchResult objects sorted by fused score
        """
        if not self._is_initialized:
            raise RuntimeError("Search engine not initialized")
        
        top_k = top_k or self.config.top_k
        k_constant = self.config.rrf_k_constant
        
        # Perform dense search
        dense_results = self._perform_dense_search(query, top_k * 2, book, testament)
        
        # Perform sparse search
        sparse_results = self._perform_sparse_search(query, top_k * 2)
        
        # Combine results using RRF
        all_verse_ids: Set[int] = set()
        for verse_id, _, _ in dense_results:
            all_verse_ids.add(verse_id)
        for verse_id, _, _ in sparse_results:
            all_verse_ids.add(verse_id)
        
        # Calculate RRF scores
        rrf_scores = {}
        dense_ranks = {}
        sparse_ranks = {}
        
        # Convert results to dictionaries for quick lookup
        dense_dict = {verse_id: (score, rank) for verse_id, score, rank in dense_results}
        sparse_dict = {verse_id: (score, rank) for verse_id, score, rank in sparse_results}
        
        for verse_id in all_verse_ids:
            rrf_score = 0.0
            
            # Add dense score if present
            if verse_id in dense_dict:
                _, rank = dense_dict[verse_id]
                rrf_score += 1.0 / (k_constant + rank)
                dense_ranks[verse_id] = rank
            else:
                dense_ranks[verse_id] = top_k + 1
            
            # Add sparse score if present
            if verse_id in sparse_dict:
                _, rank = sparse_dict[verse_id]
                rrf_score += 1.0 / (k_constant + rank)
                sparse_ranks[verse_id] = rank
            else:
                sparse_ranks[verse_id] = top_k + 1
            
            rrf_scores[verse_id] = rrf_score
        
        # Sort by RRF score
        sorted_verse_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Create SearchResult objects
        results = []
        for verse_id, fused_score in sorted_verse_ids:
            # Find the verse data
            verse = next(v for v in self._verses if v["id"] == verse_id)
            
            # Get individual scores
            dense_score = dense_dict.get(verse_id, (0.0, 0))[0]
            sparse_score = sparse_dict.get(verse_id, (0.0, 0))[0]
            
            result = SearchResult(
                id=verse_id,
                book=verse["book"],
                chapter=verse["chapter"],
                verse=verse["verse"],
                text=verse["text"],
                testament=verse["testament"],
                category=verse["category"],
                dense_score=dense_score,
                sparse_score=sparse_score,
                fused_score=fused_score,
                dense_rank=dense_ranks[verse_id],
                sparse_rank=sparse_ranks[verse_id]
            )
            results.append(result)
        
        return results
    
    def search_fast(self, query: str, top_k: Optional[int] = None,
                    book: Optional[str] = None,
                    testament: Optional[str] = None) -> List[SearchResult]:
        """
        Fast search using only dense vector search (no BM25).
        
        Args:
            query: Search query string
            top_k: Number of results to return
            book: Filter by book name
            testament: Filter by testament (OT or NT)
            
        Returns:
            List of SearchResult objects sorted by dense score
        """
        if not self._is_initialized:
            raise RuntimeError("Search engine not initialized")
        
        top_k = top_k or self.config.top_k
        
        # Perform dense search
        dense_results = self._perform_dense_search(query, top_k, book, testament)
        
        # Create SearchResult objects
        results = []
        for verse_id, dense_score, dense_rank in dense_results:
            verse = next(v for v in self._verses if v["id"] == verse_id)
            
            result = SearchResult(
                id=verse_id,
                book=verse["book"],
                chapter=verse["chapter"],
                verse=verse["verse"],
                text=verse["text"],
                testament=verse["testament"],
                category=verse["category"],
                dense_score=dense_score,
                sparse_score=0.0,
                fused_score=dense_score,  # Use dense score as fused score
                dense_rank=dense_rank,
                sparse_rank=0
            )
            results.append(result)
        
        return results
python
# engine/llm_provider.py
"""LLM provider module for streaming responses from multiple AI services."""

import json
from typing import Generator, List, Dict, Any, Optional
import httpx
import streamlit as st

from config import DEFAULT_LLM_CONFIG, LLMConfig


class LLMProviderError(Exception):
    """Exception raised for LLM provider errors."""
    pass


def get_available_models(provider: str) -> List[str]:
    """
    Get available models for a given provider.
    
    Args:
        provider: The provider name ('ollama', 'google', 'openai')
        
    Returns:
        List of model names
    """
    if provider == "ollama":
        return ["llama2", "llama3", "mixtral", "gemma", "phi"]
    elif provider == "google":
        return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    elif provider == "openai":
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    else:
        return []


def _format_context_verses(context_verses: List[Dict[str, Any]]) -> str:
    """Format context verses for the LLM prompt."""
    if not context_verses:
        return "No specific Bible passages provided."
    
    formatted = []
    for verse in context_verses:
        reference = f"{verse['book']} {verse['chapter']}:{verse['verse']}"
        formatted.append(f"{reference} - {verse['text']}")
    
    return "\n".join(formatted)


def stream_llm_response(
    provider: str,
    model_name: str,
    prompt: str,
    system_prompt: str,
    api_key: Optional[str] = None,
    context_verses: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Generator[str, None, None]:
    """
    Stream a response from the specified LLM provider.
    
    Args:
        provider: The provider name ('ollama', 'google', 'openai')
        model_name: The model name to use
        prompt: The user prompt
        system_prompt: The system prompt
        api_key: API key for cloud providers
        context_verses: List of context verses
        temperature: Temperature for generation (uses default if None)
        max_tokens: Maximum tokens to generate (uses default if None)
        
    Yields:
        Chunks of the response as strings
        
    Raises:
        LLMProviderError: If there's an error with the provider
        ValueError: If provider is unsupported or missing required config
    """
    config = DEFAULT_LLM_CONFIG
    temperature = temperature or config.temperature
    max_tokens = max_tokens or config.max_tokens
    
    # Format context
    context_text = _format_context_verses(context_verses) if context_verses else ""
    
    # Build the full prompt
    full_prompt = f"{system_prompt}\n\nContext passages:\n{context_text}\n\nUser query: {prompt}"
    
    if provider == "ollama":
        yield from _stream_ollama(model_name, full_prompt, system_prompt, temperature, max_tokens)
    elif provider == "google":
        if not api_key:
            raise ValueError("API key required for Google Gemini")
        yield from _stream_google(model_name, full_prompt, system_prompt, api_key, temperature, max_tokens)
    elif provider == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI")
        yield from _stream_openai(model_name, full_prompt, system_prompt, api_key, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _stream_ollama(
    model_name: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int
) -> Generator[str, None, None]:
    """Stream response from Ollama."""
    config = DEFAULT_LLM_CONFIG
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream(
                "POST",
                config.ollama_url,
                json=payload
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
    except httpx.TimeoutException:
        raise LLMProviderError("Ollama request timed out. Ensure Ollama is running and responsive.")
    except httpx.ConnectError:
        raise LLMProviderError("Failed to connect to Ollama. Ensure Ollama is running at " + config.ollama_url)
    except httpx.HTTPStatusError as e:
        raise LLMProviderError(f"Ollama returned error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise LLMProviderError(f"Ollama error: {str(e)}")


def _stream_google(
    model_name: str,
    prompt: str,
    system_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int
) -> Generator[str, None, None]:
    """Stream response from Google Gemini."""
    try:
        from google import genai
        
        client = genai.Client(api_key=api_key)
        
        # Combine system prompt with user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = client.models.generate_content_stream(
            model=model_name,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except ImportError:
        raise LLMProviderError("google-genai package not installed. Run: uv add google-genai")
    except Exception as e:
        raise LLMProviderError(f"Google Gemini error: {str(e)}")


def _stream_openai(
    model_name: str,
    prompt: str,
    system_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int
) -> Generator[str, None, None]:
    """Stream response from OpenAI."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except ImportError:
        raise LLMProviderError("openai package not installed. Run: uv add openai")
    except Exception as e:
        raise LLMProviderError(f"OpenAI error: {str(e)}")
python
# app.py
"""Main Streamlit UI application for TBS (Tuixiu Bible Search)."""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import streamlit as st
import pandas as pd

from config import DEFAULT_SEARCH_CONFIG, DEFAULT_LLM_CONFIG, get_system_prompt
from engine import (
    load_bible_data,
    get_verse_reference,
    HybridSearchEngine,
    SearchResult,
    stream_llm_response,
    get_available_models
)


# Page configuration
st.set_page_config(
    page_title="TBS - Tuixiu Bible Search",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .verse-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1e3a5f;
    }
    .verse-reference {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
    }
    .verse-text {
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    .verse-meta {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .ai-response {
        background-color: #e8f4f8;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 4px solid #007bff;
    }
    .score-badge {
        display: inline-block;
        background-color: #e9ecef;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        color: #495057;
        margin-right: 0.5rem;
    }
    .category-tag {
        display: inline-block;
        background-color: #d4edda;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        color: #155724;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "search_engine" not in st.session_state:
        st.session_state.search_engine = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "ai_response" not in st.session_state:
        st.session_state.ai_response = ""
    if "ollama_api_key" not in st.session_state:
        st.session_state.ollama_api_key = ""
    if "google_api_key" not in st.session_state:
        st.session_state.google_api_key = ""
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = "ollama"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "llama2"
    if "ai_mode" not in st.session_state:
        st.session_state.ai_mode = False
    if "fast_mode" not in st.session_state:
        st.session_state.fast_mode = False
    if "initialized" not in st.session_state:
        st.session_state.initialized = False


@st.cache_data(ttl=3600)
def load_bible_data_cached(file_path: str) -> List[Dict[str, Any]]:
    """Load Bible data with caching."""
    return load_bible_data(file_path)


@st.cache_resource(ttl=3600)
def get_search_engine() -> HybridSearchEngine:
    """Get or create the search engine with caching."""
    return HybridSearchEngine()


def display_sidebar() -> tuple[str, str, str, bool, int, Optional[str], Optional[str]]:
    """Display the sidebar and return configuration values."""
    st.sidebar.title("⚙️ Configuration")
    
    # Provider selection
    provider = st.sidebar.selectbox(
        "🤖 AI Provider",
        ["ollama", "google", "openai"],
        format_func=lambda x: {
            "ollama": "Local Ollama",
            "google": "Google Gemini",
            "openai": "OpenAI"
        }.get(x, x),
        help="Select the LLM provider for AI exegesis"
    )
    st.session_state.selected_provider = provider
    
    # Model selection
    models = get_available_models(provider)
    default_model = models[0] if models else ""
    model = st.sidebar.selectbox(
        "🧠 Model",
        models,
        help="Select the model to use for AI responses"
    )
    st.session_state.selected_model = model
    
    # API key inputs
    api_key = None
    if provider == "ollama":
        # Ollama doesn't require an API key
        st.sidebar.info("🔗 Connect to local Ollama service at http://localhost:11434")
    elif provider == "google":
        api_key = st.sidebar.text_input(
            "🔑 Google API Key",
            type="password",
            value=st.session_state.google_api_key,
            help="Enter your Google Gemini API key"
        )
        st.session_state.google_api_key = api_key
        if not api_key:
            st.sidebar.warning("Please enter your Google API key")
    elif provider == "openai":
        api_key = st.sidebar.text_input(
            "🔑 OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Enter your OpenAI API key"
        )
        st.session_state.openai_api_key = api_key
        if not api_key:
            st.sidebar.warning("Please enter your OpenAI API key")
    
    st.sidebar.divider()
    
    # Search settings
    st.sidebar.subheader("🔍 Search Settings")
    
    top_k = st.sidebar.slider(
        "📊 Results Depth",
        min_value=1,
        max_value=10,
        value=DEFAULT_SEARCH_CONFIG.top_k,
        help="Number of search results to return"
    )
    
    fast_mode = st.sidebar.toggle(
        "⚡ Fast Mode",
        value=st.session_state.fast_mode,
        help="Use only dense vector search (faster but less comprehensive)"
    )
    st.session_state.fast_mode = fast_mode
    
    # Filters
    st.sidebar.subheader("🎯 Filters")
    
    # Get available books from data if loaded
    books = ["All"]
    if st.session_state.search_engine:
        # We don't have direct access to verses, but we can use what's loaded
        pass
    
    book_filter = st.sidebar.selectbox(
        "📖 Book",
        ["All", "Genesis", "Psalms", "Isaiah", "Matthew", "John", "Romans"],
        help="Filter by book of the Bible"
    )
    
    testament_filter = st.sidebar.selectbox(
        "📜 Testament",
        ["All", "OT", "NT"],
        help="Filter by Old or New Testament"
    )
    
    st.sidebar.divider()
    
    # AI Mode toggle
    ai_mode = st.sidebar.toggle(
        "🧠 AI Exegesis Mode",
        value=st.session_state.ai_mode,
        help="Enable AI-powered analysis of the search results"
    )
    st.session_state.ai_mode = ai_mode
    
    return provider, model, api_key, ai_mode, top_k, book_filter, testament_filter


def display_verse_card(result: SearchResult) -> None:
    """Display a single verse card."""
    with st.container():
        st.markdown(f"""
        <div class="verse-card">
            <div class="verse-reference">{result.get_reference()}</div>
            <div class="verse-text">{result.text}</div>
            <div class="verse-meta">
                <span class="category-tag">{result.category}</span>
                <span class="score-badge">RRF: {result.fused_score:.3f}</span>
                <span class="score-badge">Dense: {result.dense_score:.3f}</span>
                <span class="score-badge">Sparse: {result.sparse_score:.3f}</span>
                <span>📖 {result.testament}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def generate_ai_response(
    provider: str,
    model: str,
    query: str,
    api_key: Optional[str],
    results: List[SearchResult],
    system_prompt: str
) -> None:
    """Generate and stream AI response based on search results."""
    if not results:
        st.warning("No search results to analyze")
        return
    
    # Prepare context from search results
    context_verses = [
        {
            "book": r.book,
            "chapter": r.chapter,
            "verse": r.verse,
            "text": r.text
        }
        for r in results
    ]
    
    # Generate prompt
    prompt = f"""Analyze the following Bible passages and provide exegetical insights:

Search query: "{query}"

Please provide:
1. A summary of the passages found
2. Key theological themes
3. Connections between the passages
4. Practical applications

Format your response with clear sections and verse citations."""

    # Stream the response
    response_container = st.empty()
    full_response = ""
    
    try:
        with st.spinner("Generating AI response..."):
            for chunk in stream_llm_response(
                provider=provider,
                model_name=model,
                prompt=prompt,
                system_prompt=system_prompt,
                api_key=api_key,
                context_verses=context_verses
            ):
                full_response += chunk
                response_container.markdown(f"""
                <div class="ai-response">
                    <h4>🤖 AI Exegesis</h4>
                    {full_response}
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating AI response: {str(e)}")


def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    
    # Load Bible data
    data_path = Path(__file__).parent / "data" / "sample_bible.json"
    try:
        verses = load_bible_data_cached(str(data_path))
    except FileNotFoundError:
        st.error(f"Bible data file not found: {data_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading Bible data: {str(e)}")
        st.stop()
    
    # Initialize search engine
    search_engine = get_search_engine()
    if not st.session_state.initialized:
        with st.spinner("Initializing search engine..."):
            search_engine.initialize(verses)
            st.session_state.search_engine = search_engine
            st.session_state.initialized = True
    
    # Sidebar configuration
    provider, model, api_key, ai_mode, top_k, book_filter, testament_filter = display_sidebar()
    
    # Main content
    st.markdown('<div class="main-header">📖 TBS — Tuixiu Bible Search</div>', unsafe_allow_html=True)
    st.caption("Hybrid Bible Search with AI-Powered Exegesis")
    st.divider()
    
    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "🔍 Search the Bible",
            placeholder="Enter your search query...",
            value=st.session_state.query,
            label_visibility="collapsed"
        )
        st.session_state.query = query
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Search execution
    if search_button and query:
        with st.spinner("Searching..."):
            # Prepare filters
            book = book_filter if book_filter != "All" else None
            testament = testament_filter if testament_filter != "All" else None
            
            # Perform search
            try:
                if st.session_state.fast_mode:
                    results = search_engine.search_fast(
                        query=query,
                        top_k=top_k,
                        book=book,
                        testament=testament
                    )
                else:
                    results = search_engine.search(
                        query=query,
                        top_k=top_k,
                        book=book,
                        testament=testament
                    )
                
                st.session_state.search_results = results
                
            except Exception as e:
                st.error(f"Search error: {str(e)}")
                results = []
    
    # Display results
    results = st.session_state.search_results
    
    if results:
        # Show count
        st.info(f"Found {len(results)} results")
        
        # Display results in columns
        cols = st.columns(1)
        for result in results:
            with cols[0]:
                display_verse_card(result)
        
        # AI Response
        if ai_mode and results:
            st.divider()
            generate_ai_response(
                provider=provider,
                model=model,
                query=query,
                api_key=api_key if provider in ["google", "openai"] else None,
                results=results[:3],  # Use top 3 for context
                system_prompt=get_system_prompt()
            )
    
    elif search_button and query:
        st.info("No results found. Try adjusting your search query or filters.")
    
    elif not query:
        st.info("Enter a search query to begin exploring the Bible.")
    
    # Footer
    st.divider()
    st.caption("TBS — Tuixiu Bible Search v1.0.0 | Powered by Hybrid Search & AI")


if __name__ == "__main__":
    main()
markdown
# README.md

# TBS — Tuixiu Bible Search

A hybrid, local-first Bible search and exegesis tool built with Python 3.14+, Streamlit, Qdrant, BM25, and AI integration.

## 🚀 Features

- **Hybrid Search**: Combines dense vector search (semantic) with sparse BM25 search (keyword) using Reciprocal Rank Fusion (RRF)
- **AI-Powered Exegesis**: Get AI-generated insights from multiple providers (Ollama, Google Gemini, OpenAI)
- **Local-First**: Uses in-memory Qdrant for vector search and local Ollama support
- **Fast Mode**: Option for quick semantic-only search
- **Filters**: Filter by Bible book and testament
- **Beautiful UI**: Modern Streamlit interface with verse cards and real-time streaming

## 📋 Prerequisites

- **Python 3.14+** (required)
- **uv** package manager
- **Optional**: **Ollama** for local LLM support

## 🛠️ Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
2. Clone/Initialize Project
bash
# Create project directory
uv init tuixiu-bible-search --app
cd tuixiu-bible-search

# Copy all source files into the directory
# Ensure the following structure:
# tuixiu-bible-search/
# ├── app.py
# ├── config.py
# ├── engine/
# │   ├── __init__.py
# │   ├── hybrid_search.py
# │   ├── indexer.py
# │   └── llm_provider.py
# └── data/
#     └── sample_bible.json
3. Pin Python Version
bash
uv python pin 3.14
4. Install Dependencies
bash
# Install all dependencies (latest versions)
uv add "streamlit@latest" \
       "qdrant-client@latest" \
       "sentence-transformers@latest" \
       "rank-bm25@latest" \
       "google-genai@latest" \
       "openai@latest" \
       "httpx@latest" \
       "pydantic@latest" \
       "numpy@latest" \
       "torch@latest" \
       "transformers@latest" \
       "tokenizers@latest"

# Or use pinned versions for stability
uv add "streamlit>=1.40.0" \
       "qdrant-client>=1.12.0" \
       "sentence-transformers>=3.0.0" \
       "rank-bm25>=0.2.2" \
       "google-genai>=0.1.0" \
       "openai>=1.50.0" \
       "httpx>=0.27.0" \
       "pydantic>=2.9.0" \
       "numpy>=1.26.0" \
       "torch>=2.0.0" \
       "transformers>=4.30.0" \
       "tokenizers>=0.13.0"

# Generate lockfile
uv lock
🚀 Running the Application
Start the Application
bash
uv run streamlit run app.py
The application will open in your browser at http://localhost:8501.

Options
Port: Change port with --server.port 8502

Headless: Run with --server.headless true

⚙️ Configuration
Ollama Setup
Install Ollama from ollama.ai

Pull a model:

bash
ollama pull llama2
# or
ollama pull llama3
Ensure Ollama is running:

bash
ollama serve
Google Gemini Setup
Get API key from Google AI Studio

Enter the API key in the app sidebar

OpenAI Setup
Get API key from OpenAI Platform

Enter the API key in the app sidebar

🔍 Using the Application
Search: Enter a query in the search box and click "Search"

Filter: Use sidebar filters to narrow by book or testament

Fast Mode: Toggle for quick semantic-only search

AI Mode: Enable AI exegesis for analyzed results

Provider: Choose between Ollama, Google Gemini, or OpenAI

🛠️ Development
Code Quality
bash
# Run formatter
uv run black .

# Run linter
uv run ruff check .

# Run type checker
uv run mypy .
Testing
bash
# Run tests (when implemented)
uv run pytest
Updating Dependencies
bash
# Update to latest versions
uv sync --upgrade
📊 Architecture
Frontend: Streamlit with custom CSS

Vector Search: Qdrant in-memory with sentence-transformers

Sparse Search: BM25Okapi from rank-bm25

Fusion: Reciprocal Rank Fusion (RRF) with k=60

AI: Streaming via Ollama, Google, or OpenAI

🐛 Troubleshooting
Common Issues
Qdrant in-memory issues:

No configuration needed - runs entirely in memory

Ollama connection errors:

Ensure Ollama is running: ollama serve

Check port: http://localhost:11434

API key errors:

Verify API keys are entered correctly in the sidebar

Ensure your Google/OpenAI account has access to the selected models

Memory issues:

The embedding model uses memory; consider using smaller models

Reduce batch sizes in config.py

Logs
Check Streamlit logs for detailed error messages:

bash
uv run streamlit run app.py --logger.level=debug
📝 License
MIT License

🙏 Acknowledgments
Built with Streamlit, Qdrant, and sentence-transformers

BM25 implementation from rank-bm25

AI integration with Ollama, Google Gemini, and OpenAI

Happy Bible Searching! 📖✨

text

Now I'll create the complete codebase with all the necessary files. Let me compile the final output.

---
## Complete Codebase

'll continue by providing the complete implementation of the remaining files and ensure all code is fully functional.

python
# engine/__init__.py (complete)
"""Engine module for TBS - contains search and LLM functionality."""

from engine.indexer import load_bible_data, get_verse_reference, validate_verse, BibleDataError
from engine.hybrid_search import HybridSearchEngine, SearchResult
from engine.llm_provider import stream_llm_response, get_available_models, LLMProviderError

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
Now let me also create a comprehensive pyproject.toml with all the necessary configurations:

toml
# pyproject.toml (complete with all configurations)
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tuixiu-bible-search"
version = "1.0.0"
description = "TBS (Tuixiu Bible Search) - A hybrid, local-first Bible search and exegesis tool"
readme = "README.md"
requires-python = ">=3.14"
authors = [
    {name = "TBS Team", email = "tbs@example.com"}
]
license = {text = "MIT"}
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Religion",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.14",
    "Operating System :: OS Independent",
    "Topic :: Religion",
    "Topic :: Text Processing :: Indexing",
]

dependencies = [
    "streamlit>=1.40.0",
    "qdrant-client>=1.12.0",
    "sentence-transformers>=3.0.0",
    "rank-bm25>=0.2.2",
    "google-genai>=0.1.0",
    "openai>=1.50.0",
    "httpx>=0.27.0",
    "pydantic>=2.9.0",
    "numpy>=1.26.0",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "tokenizers>=0.13.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
    "mypy>=1.8.0",
    "pre-commit>=3.5.0",
]

[project.urls]
Homepage = "https://github.com/tuixiu/tuixiu-bible-search"
Repository = "https://github.com/tuixiu/tuixiu-bible-search"
Issues = "https://github.com/tuixiu/tuixiu-bible-search/issues"

[tool.black]
line-length = 100
target-version = ['py314']
include = '\.pyi?$'
extend-exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.ruff]
line-length = 100
target-version = "py314"
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "W",   # pycodestyle warnings
    "I",   # isort
    "N",   # pep8-naming
    "D",   # pydocstyle
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "SIM", # flake8-simplify
    "TCH", # flake8-type-checking
]
ignore = [
    "D100", # Missing docstring in public module
    "D104", # Missing docstring in public package
    "D107", # Missing docstring in __init__
    "D203", # 1 blank line required before class docstring
    "D212", # Multi-line docstring summary should start at the first line
    "E501", # Line too long (handled by black)
]
exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
]

[tool.ruff.isort]
known-first-party = ["engine"]

[tool.mypy]
python_version = "3.14"
strict = true
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = false
disallow_untyped_defs = true
disallow_any_unimported = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
strict_concatenate = true
explicit_package_bases = true
check_untyped_defs = true
disallow_subclassing_any = true
disallow_any_generics = true
disallow_any_decorated = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q --strict-markers"
testpaths = [
    "tests",
]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

[tool.coverage.run]
source = ["engine", "app.py", "config.py"]
omit = ["tests/*", "*/__pycache__/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if __name__ == .__main__.:",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if False:",
]
Now let me also create a .gitignore file for the project:

gitignore
# .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
pythonenv*
*.egg-info/
dist/
build/
*.egg
.eggs/
*.whl
*.tar.gz

# Virtual environments
.venv/
venv/
virtualenv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
*.log

# Streamlit
.streamlit/secrets.toml
.streamlit/cache/

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre
.pyre/

# PyCharm
.idea/
*.iml

# VS Code
.vscode/
*.code-workspace

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
*.cover
.hypothesis/

# UV
uv.lock

# Database files
*.db
*.sqlite
*.sqlite3

# Logs
logs/
*.log

# Temporary files
tmp/
temp/
*.tmp

# OS files
Thumbs.db
.DS_Store
desktop.ini

# Secret keys and API keys
.env
*.key
*.pem
*.crt
*.csr

# Large files
*.pkl
*.h5
*.pb
*.onnx
*.ckpt
*.pt
*.pth

# Data files (except sample)
data/*.json
!data/sample_bible.json
And a .pre-commit-config.yaml for code quality automation:

yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
      - id: check-toml
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict]

  - repo: https://github.com/psf/black-pre-commit-mirror
    rev: 24.2.0
    hooks:
      - id: black
        language_version: python3.14
Now let me provide a complete app.py with all the enhancements and proper session state management:

python
# app.py (complete with all functionality)
"""Main Streamlit UI application for TBS (Tuixiu Bible Search)."""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
import streamlit as st
import pandas as pd
from datetime import datetime

from config import DEFAULT_SEARCH_CONFIG, DEFAULT_LLM_CONFIG, get_system_prompt
from engine import (
    load_bible_data,
    get_verse_reference,
    HybridSearchEngine,
    SearchResult,
    stream_llm_response,
    get_available_models,
    BibleDataError,
    LLMProviderError
)


# Page configuration
st.set_page_config(
    page_title="TBS - Tuixiu Bible Search",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .verse-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1e3a5f;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .verse-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .verse-reference {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .verse-text {
        font-size: 1.05rem;
        line-height: 1.7;
        margin-top: 0.5rem;
        color: #2c3e50;
    }
    .verse-meta {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }
    .score-badge {
        display: inline-block;
        background-color: #e9ecef;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #495057;
        font-weight: 500;
    }
    .score-badge.high {
        background-color: #d4edda;
        color: #155724;
    }
    .score-badge.medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .category-tag {
        display: inline-block;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #155724;
        font-weight: 500;
    }
    .testament-tag {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .testament-tag.ot {
        background-color: #cce5ff;
        color: #004085;
    }
    .testament-tag.nt {
        background-color: #ffe5cc;
        color: #854d00;
    }
    .ai-response {
        background: linear-gradient(135deg, #e8f4f8 0%, #d4e8f0 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .ai-response h4 {
        color: #1e3a5f;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .ai-response h4 .spinner {
        display: inline-block;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0;
        font-size: 3rem;
    }
    .sidebar-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 1rem;
    }
    .search-stats {
        background: #f8f9fa;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .stat-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #495057;
        font-size: 0.9rem;
    }
    .stat-item .stat-value {
        font-weight: 600;
        color: #1e3a5f;
    }
    .footer {
        margin-top: 3rem;
        padding: 1rem;
        text-align: center;
        color: #6c757d;
        font-size: 0.85rem;
        border-top: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "search_engine": None,
        "search_results": [],
        "query": "",
        "ai_response": "",
        "ollama_api_key": "",
        "google_api_key": "",
        "openai_api_key": "",
        "selected_provider": "ollama",
        "selected_model": "llama2",
        "ai_mode": False,
        "fast_mode": False,
        "initialized": False,
        "last_search_time": None,
        "search_count": 0,
        "error_message": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


@st.cache_data(ttl=3600)
def load_bible_data_cached(file_path: str) -> List[Dict[str, Any]]:
    """Load Bible data with caching."""
    return load_bible_data(file_path)


@st.cache_resource(ttl=3600)
def get_search_engine() -> HybridSearchEngine:
    """Get or create the search engine with caching."""
    return HybridSearchEngine()


def display_sidebar() -> tuple[str, str, str, bool, int, Optional[str], Optional[str]]:
    """Display the sidebar and return configuration values."""
    st.sidebar.markdown('<div class="sidebar-logo">📖</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-title">TBS Config</div>', unsafe_allow_html=True)
    st.sidebar.divider()
    
    # Provider selection
    st.sidebar.subheader("🤖 AI Provider")
    provider = st.sidebar.selectbox(
        "Select Provider",
        ["ollama", "google", "openai"],
        format_func=lambda x: {
            "ollama": "🏠 Local Ollama",
            "google": "☁️ Google Gemini",
            "openai": "☁️ OpenAI"
        }.get(x, x),
        help="Select the LLM provider for AI exegesis",
        key="provider_select"
    )
    st.session_state.selected_provider = provider
    
    # Model selection
    st.sidebar.subheader("🧠 Model")
    models = get_available_models(provider)
    default_model = models[0] if models else ""
    model = st.sidebar.selectbox(
        "Select Model",
        models,
        help="Select the model to use for AI responses",
        key="model_select"
    )
    st.session_state.selected_model = model
    
    # API key inputs
    st.sidebar.subheader("🔑 API Keys")
    api_key = None
    
    if provider == "ollama":
        st.sidebar.info("🔗 Using local Ollama at http://localhost:11434")
        if st.sidebar.button("🔄 Check Ollama Status"):
            try:
                import httpx
                response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
                if response.status_code == 200:
                    st.sidebar.success("✅ Ollama is running!")
                    models_list = response.json().get("models", [])
                    if models_list:
                        st.sidebar.info(f"Available models: {', '.join([m.get('name', 'unknown') for m in models_list])}")
                else:
                    st.sidebar.error("❌ Ollama returned an error")
            except Exception as e:
                st.sidebar.error(f"❌ Cannot connect to Ollama: {str(e)}")
    elif provider == "google":
        api_key = st.sidebar.text_input(
            "Google API Key",
            type="password",
            value=st.session_state.google_api_key,
            help="Enter your Google Gemini API key from Google AI Studio",
            placeholder="AIza..."
        )
        st.session_state.google_api_key = api_key
        if not api_key:
            st.sidebar.warning("⚠️ Please enter your Google API key")
    elif provider == "openai":
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key,
            help="Enter your OpenAI API key from platform.openai.com",
            placeholder="sk-..."
        )
        st.session_state.openai_api_key = api_key
        if not api_key:
            st.sidebar.warning("⚠️ Please enter your OpenAI API key")
    
    st.sidebar.divider()
    
    # Search settings
    st.sidebar.subheader("🔍 Search Settings")
    
    top_k = st.sidebar.slider(
        "📊 Results Depth",
        min_value=1,
        max_value=10,
        value=DEFAULT_SEARCH_CONFIG.top_k,
        help="Number of search results to return",
        key="top_k_slider"
    )
    
    fast_mode = st.sidebar.toggle(
        "⚡ Fast Mode",
        value=st.session_state.fast_mode,
        help="Use only dense vector search (faster but less comprehensive)",
        key="fast_mode_toggle"
    )
    st.session_state.fast_mode = fast_mode
    
    # Filters
    st.sidebar.subheader("🎯 Filters")
    
    # Get unique books from data if available
    books = ["All"]
    if st.session_state.search_engine:
        # We'll use the predefined list from sample data
        books = ["All", "Genesis", "Psalms", "Isaiah", "Matthew", "John", "Romans"]
    
    book_filter = st.sidebar.selectbox(
        "📖 Book",
        books,
        help="Filter by book of the Bible",
        key="book_filter_select"
    )
    
    testament_filter = st.sidebar.selectbox(
        "📜 Testament",
        ["All", "OT", "NT"],
        help="Filter by Old or New Testament",
        key="testament_filter_select"
    )
    
    st.sidebar.divider()
    
    # AI Mode toggle
    st.sidebar.subheader("🧠 AI Features")
    ai_mode = st.sidebar.toggle(
        "🤖 AI Exegesis Mode",
        value=st.session_state.ai_mode,
        help="Enable AI-powered analysis of the search results",
        key="ai_mode_toggle"
    )
    st.session_state.ai_mode = ai_mode
    
    # Display stats
    if st.session_state.search_count > 0:
        st.sidebar.divider()
        st.sidebar.subheader("📊 Statistics")
        st.sidebar.metric("Searches", st.session_state.search_count)
        if st.session_state.last_search_time:
            st.sidebar.caption(f"Last search: {st.session_state.last_search_time}")
    
    return provider, model, api_key, ai_mode, top_k, book_filter, testament_filter


def display_verse_card(result: SearchResult, index: int) -> None:
    """Display a single verse card with enhanced styling."""
    # Determine score badge class
    score_class = "medium"
    if result.fused_score > 0.5:
        score_class = "high"
    
    # Determine testament class
    testament_class = "ot" if result.testament == "OT" else "nt"
    
    with st.container():
        st.markdown(f"""
        <div class="verse-card" style="animation: fadeIn 0.5s ease;">
            <div class="verse-reference">
                <span>{result.get_reference()}</span>
                <span class="category-tag">{result.category}</span>
                <span class="testament-tag {testament_class}">{result.testament}</span>
            </div>
            <div class="verse-text">"{result.text}"</div>
            <div class="verse-meta">
                <span class="score-badge {score_class}">🎯 RRF: {result.fused_score:.3f}</span>
                <span class="score-badge">📊 Dense: {result.dense_score:.3f}</span>
                <span class="score-badge">📝 Sparse: {result.sparse_score:.3f}</span>
                <span class="score-badge">#️⃣ Rank: {index + 1}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def generate_ai_response(
    provider: str,
    model: str,
    query: str,
    api_key: Optional[str],
    results: List[SearchResult],
    system_prompt: str
) -> None:
    """Generate and stream AI response based on search results."""
    if not results:
        st.warning("No search results to analyze")
        return
    
    # Prepare context from search results
    context_verses = [
        {
            "book": r.book,
            "chapter": r.chapter,
            "verse": r.verse,
            "text": r.text
        }
        for r in results[:5]  # Use top 5 for context
    ]
    
    # Generate a comprehensive prompt
    prompt = f"""Please provide a thorough exegetical analysis of the following Bible passages found for the query: "{query}"

Context Passages:
{chr(10).join([f"- {v['book']} {v['chapter']}:{v['verse']}: {v['text']}" for v in context_verses])}

Please structure your response as follows:

1. **Summary**: Brief overview of the passages and their connection to the query
2. **Theological Themes**: Key theological concepts and doctrines present in these passages
3. **Historical Context**: Important historical or cultural background
4. **Practical Application**: How these passages apply to modern Christian living
5. **Connections**: Interconnections between these passages and other related scriptures

Format your response with clear sections, proper citations, and maintain a scholarly yet accessible tone.
Be thorough but concise. Ground all insights in the provided scripture passages."""

    # Stream the response
    response_container = st.empty()
    full_response = ""
    
    try:
        with st.spinner("Generating AI exegesis..."):
            response_container.markdown("""
            <div class="ai-response">
                <h4>🤖 AI Exegesis <span class="spinner">⏳</span></h4>
                <p><em>Generating insights...</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            for chunk in stream_llm_response(
                provider=provider,
                model_name=model,
                prompt=prompt,
                system_prompt=system_prompt,
                api_key=api_key,
                context_verses=context_verses,
                temperature=0.7,
                max_tokens=1500
            ):
                full_response += chunk
                response_container.markdown(f"""
                <div class="ai-response">
                    <h4>🤖 AI Exegesis</h4>
                    <div style="margin-top: 1rem; line-height: 1.8;">
                        {full_response}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    except LLMProviderError as e:
        st.error(f"AI Provider Error: {str(e)}")
        response_container.empty()
    except Exception as e:
        st.error(f"Error generating AI response: {str(e)}")
        response_container.empty()


def handle_search(
    search_engine: HybridSearchEngine,
    query: str,
    top_k: int,
    book_filter: str,
    testament_filter: str,
    fast_mode: bool
) -> List[SearchResult]:
    """Execute the search with error handling."""
    try:
        # Prepare filters
        book = book_filter if book_filter != "All" else None
        testament = testament_filter if testament_filter != "All" else None
        
        # Perform search based on mode
        if fast_mode:
            results = search_engine.search_fast(
                query=query,
                top_k=top_k,
                book=book,
                testament=testament
            )
        else:
            results = search_engine.search(
                query=query,
                top_k=top_k,
                book=book,
                testament=testament
            )
        
        # Update session stats
        st.session_state.search_count += 1
        st.session_state.last_search_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.error_message = None
        
        return results
        
    except Exception as e:
        st.session_state.error_message = str(e)
        return []


def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    
    # Load Bible data
    data_path = Path(__file__).parent / "data" / "sample_bible.json"
    try:
        verses = load_bible_data_cached(str(data_path))
    except FileNotFoundError:
        st.error(f"❌ Bible data file not found: {data_path}")
        st.info("Please ensure the data directory contains sample_bible.json")
        st.stop()
    except BibleDataError as e:
        st.error(f"❌ Bible data validation error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error loading Bible data: {str(e)}")
        st.stop()
    
    # Initialize search engine
    search_engine = get_search_engine()
    if not st.session_state.initialized:
        with st.spinner("🔮 Initializing search engine..."):
            try:
                search_engine.initialize(verses)
                st.session_state.search_engine = search_engine
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"❌ Failed to initialize search engine: {str(e)}")
                st.stop()
    
    # Sidebar configuration
    provider, model, api_key, ai_mode, top_k, book_filter, testament_filter = display_sidebar()
    
    # Main content
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header">📖 TBS — Tuixiu Bible Search</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Hybrid Bible Search with AI-Powered Exegesis</div>', unsafe_allow_html=True)
    
    with col2:
        # Quick stats
        if st.session_state.search_count > 0:
            st.metric("Searches", st.session_state.search_count)
    
    st.divider()
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "🔍 Search the Bible",
            placeholder="Enter your search query (e.g., 'God created', 'faith in action', 'for God so loved')",
            value=st.session_state.query,
            label_visibility="collapsed",
            key="search_input"
        )
        st.session_state.query = query
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Handle Enter key press
    if query and (search_button or st.session_state.get("search_triggered", False)):
        st.session_state.search_triggered = False
        with st.spinner("🔍 Searching..."):
            results = handle_search(
                search_engine,
                query,
                top_k,
                book_filter,
                testament_filter,
                st.session_state.fast_mode
            )
            st.session_state.search_results = results
    
    # Display search stats
    if st.session_state.search_results:
        results = st.session_state.search_results
        st.markdown(f"""
        <div class="search-stats">
            <div class="stat-item">
                <span>📊 Found</span>
                <span class="stat-value">{len(results)} results</span>
            </div>
            <div class="stat-item">
                <span>🔍 Query</span>
                <span class="stat-value">"{query}"</span>
            </div>
            <div class="stat-item">
                <span>⚡ Mode</span>
                <span class="stat-value">{'Fast' if st.session_state.fast_mode else 'Hybrid'}</span>
            </div>
            {f'<div class="stat-item"><span>📖 Filter</span><span class="stat-value">{book_filter if book_filter != "All" else "None"}</span></div>' if book_filter != "All" else ''}
            {f'<div class="stat-item"><span>📜 Testament</span><span class="stat-value">{testament_filter if testament_filter != "All" else "None"}</span></div>' if testament_filter != "All" else ''}
        </div>
        """, unsafe_allow_html=True)
    
    # Display results
    results = st.session_state.search_results
    
    if results:
        # Display results with progressive reveal
        for idx, result in enumerate(results):
            display_verse_card(result, idx)
            
        # AI Response
        if ai_mode and results:
            st.divider()
            with st.expander("🤖 AI Exegesis", expanded=True):
                generate_ai_response(
                    provider=provider,
                    model=model,
                    query=query,
                    api_key=api_key if provider in ["google", "openai"] else None,
                    results=results,
                    system_prompt=get_system_prompt()
                )
    
    elif st.session_state.get("error_message"):
        st.error(f"❌ {st.session_state.error_message}")
        st.session_state.error_message = None
    
    elif search_button and query:
        st.info("🔍 No results found. Try adjusting your search query or filters.")
    
    elif not query:
        # Show helpful examples
        st.markdown("""
        ### 💡 Search Examples
        Try these queries:
        - "God created the heavens"
        - "faith salvation"
        - "love thy neighbor"
        - "for God so loved the world"
        - "blessed are the poor in spirit"
        
        ### 🎯 Tips
        - Use quotes for exact phrases
        - Enable AI Exegesis for deep analysis
        - Use Fast Mode for quicker results
        - Filter by book or testament for focused searches
        """)
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div class="footer">
        <strong>📖 TBS — Tuixiu Bible Search</strong> v1.0.0
        &nbsp;•&nbsp; Powered by Hybrid Search &amp; AI
        &nbsp;•&nbsp; {datetime.now().strftime('%Y')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
Now let me also create a comprehensive config.py with all the configuration options:

python
# config.py (complete with all configuration classes)
"""Configuration module for TBS (Tuixiu Bible Search)."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os


@dataclass
class SearchConfig:
    """Configuration for search parameters."""
    top_k: int = 5
    rrf_k_constant: int = 60
    vector_size: int = 384
    collection_name: str = "bible_verses"
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.rrf_k_constant < 1:
            raise ValueError("rrf_k_constant must be at least 1")
        if self.vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if not self.collection_name:
            raise ValueError("collection_name cannot be empty")


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    system_prompt: str = (
        "You are TBS (Tuixiu Bible Search), an expert biblical assistant. "
        "Ground your answer strictly in the provided Bible passages. "
        "Always cite the Book, Chapter, and Verse for every passage reference. "
        "Provide thorough, scholarly exegesis while remaining accessible to lay readers. "
        "When discussing theological concepts, be precise and reference scripture."
    )
    temperature: float = 0.7
    max_tokens: int = 1000
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_default_model: str = "llama2"
    google_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"
    timeout_seconds: int = 60
    retry_attempts: int = 3
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        if not self.ollama_url:
            raise ValueError("ollama_url cannot be empty")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress_bar: bool = False
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if not self.model_name:
            raise ValueError("model_name cannot be empty")


@dataclass
class UIConfig:
    """Configuration for UI elements."""
    page_title: str = "TBS - Tuixiu Bible Search"
    page_icon: str = "📖"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    max_results_display: int = 10
    enable_animations: bool = True
    theme: str = "light"
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_results_display < 1:
            raise ValueError("max_results_display must be at least 1")
        if self.layout not in ["wide", "centered"]:
            raise ValueError("layout must be 'wide' or 'centered'")


# Default configuration instances
DEFAULT_SEARCH_CONFIG = SearchConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
DEFAULT_UI_CONFIG = UIConfig()


def get_system_prompt() -> str:
    """Get the default system prompt for AI exegesis."""
    return DEFAULT_LLM_CONFIG.system_prompt


def get_llm_config(provider: str) -> Dict[str, Any]:
    """Get LLM configuration for a specific provider."""
    config = {
        "ollama": {
            "url": DEFAULT_LLM_CONFIG.ollama_url,
            "default_model": DEFAULT_LLM_CONFIG.ollama_default_model,
        },
        "google": {
            "model": DEFAULT_LLM_CONFIG.google_model,
        },
        "openai": {
            "model": DEFAULT_LLM_CONFIG.openai_model,
        }
    }
    return config.get(provider, {})


__all__ = [
    "SearchConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "UIConfig",
    "DEFAULT_SEARCH_CONFIG",
    "DEFAULT_LLM_CONFIG",
    "DEFAULT_EMBEDDING_CONFIG",
    "DEFAULT_UI_CONFIG",
    "get_system_prompt",
    "get_llm_config",
]
And finally, let me create a test file to demonstrate testing structure:

python
# tests/test_engine.py
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
        # Create test data
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
        
        # Write to temporary file
        test_file = tmp_path / "test_bible.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Load and validate
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
                # Missing verse field
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
        # Should not raise any exceptions
    
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
This completes the entire TBS (Tuixiu Bible Search) codebase. The project includes:

Complete project structure with all necessary files

Fully implemented search engine with hybrid search combining dense vectors and BM25

AI integration with Ollama, Google Gemini, and OpenAI

Beautiful Streamlit UI with all features including search, filters, and AI exegesis

Proper configuration with validation

Comprehensive error handling and session state management

Testing structure for future testing

Development tools configured (black, ruff, mypy, pre-commit)

The application is ready to run with uv run streamlit run app.py after following the installation instructions in the README.

---
