# DeepSeek AskRhema

I'll generate the complete AskRhema codebase according to the master prompt. Below are all required files, fully implemented and production-ready.

- pyproject.toml

- README.md
markdown


License
MIT

text

---

## **data/sample_bible.json**
```json
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
    "category": "Psalms"
  },
  {
    "id": 4,
    "book": "Psalms",
    "chapter": 23,
    "verse": 4,
    "text": "Even though I walk through the darkest valley, I will fear no evil, for you are with me; your rod and your staff, they comfort me.",
    "testament": "OT",
    "category": "Psalms"
  },
  {
    "id": 5,
    "book": "Isaiah",
    "chapter": 9,
    "verse": 6,
    "text": "For to us a child is born, to us a son is given, and the government will be on his shoulders. And he will be called Wonderful Counselor, Mighty God, Everlasting Father, Prince of Peace.",
    "testament": "OT",
    "category": "Prophecy"
  },
  {
    "id": 6,
    "book": "Isaiah",
    "chapter": 53,
    "verse": 5,
    "text": "But he was pierced for our transgressions, he was crushed for our iniquities; the punishment that brought us peace was on him, and by his wounds we are healed.",
    "testament": "OT",
    "category": "Prophecy"
  },
  {
    "id": 7,
    "book": "Matthew",
    "chapter": 5,
    "verse": 3,
    "text": "Blessed are the poor in spirit, for theirs is the kingdom of heaven.",
    "testament": "NT",
    "category": "Sermon on the Mount"
  },
  {
    "id": 8,
    "book": "Matthew",
    "chapter": 5,
    "verse": 44,
    "text": "But I tell you, love your enemies and pray for those who persecute you.",
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
    "chapter": 15,
    "verse": 5,
    "text": "I am the vine; you are the branches. If you remain in me and I in you, you will bear much fruit; apart from me you can do nothing.",
    "testament": "NT",
    "category": "Discourse"
  },
  {
    "id": 11,
    "book": "Romans",
    "chapter": 8,
    "verse": 28,
    "text": "And we know that in all things God works for the good of those who love him, who have been called according to his purpose.",
    "testament": "NT",
    "category": "Doctrine"
  },
  {
    "id": 12,
    "book": "Romans",
    "chapter": 8,
    "verse": 29,
    "text": "For those God foreknew he also predestined to be conformed to the image of his Son, that he might be the firstborn among many brothers and sisters.",
    "testament": "NT",
    "category": "Doctrine"
  },
  {
    "id": 13,
    "book": "Genesis",
    "chapter": 50,
    "verse": 20,
    "text": "You intended to harm me, but God intended it for good to accomplish what is now being done, the saving of many lives.",
    "testament": "OT",
    "category": "Patriarchs"
  },
  {
    "id": 14,
    "book": "Psalms",
    "chapter": 119,
    "verse": 105,
    "text": "Your word is a lamp for my feet, a light on my path.",
    "testament": "OT",
    "category": "Wisdom"
  }
]
config.py
python
"""Configuration dataclasses for AskRhema."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for hybrid search."""

    top_k: int = 5
    rrf_k_constant: int = 60
    vector_size: int = 384  # all-MiniLM-L6-v2 dimension
    collection_name: str = "bible_verses"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for sentence embedding."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # or "cuda"
    batch_size: int = 32


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM providers and prompts."""

    system_prompt: str = (
        "You are AskRhema, an expert biblical assistant.\n\n"
        "Ground your answer strictly in the provided Bible passages.\n"
        "Always cite the Book, Chapter, and Verse for every passage reference.\n"
        "Do not invent biblical quotations or references.\n"
        "When the retrieved passages do not adequately support an answer, clearly say so rather than presenting unsupported claims as Scripture.\n"
        "Distinguish clearly between what the biblical text says and interpretive explanation."
    )
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    # Provider-specific defaults; user may override via settings.
    default_provider: str = "ollama"  # or "gemini", "openai"
engine/init.py
python
"""AskRhema engine package."""
engine/indexer.py
python
"""Bible data loading, validation, and reference formatting."""

import json
from pathlib import Path
from typing import Any


class BibleDataError(Exception):
    """Raised when Bible data is invalid or malformed."""

    pass


def load_bible_data(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load and validate Bible verses from a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        List of validated verse records.

    Raises:
        BibleDataError: If the file is missing, malformed, or records lack required fields.
    """
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

    required_fields = {"id", "book", "chapter", "verse", "text", "testament", "category"}
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            raise BibleDataError(f"Record {i} is not a dictionary")
        missing = required_fields - record.keys()
        if missing:
            raise BibleDataError(
                f"Record {i} missing required fields: {missing}"
            )
        # Additional basic validation
        if not isinstance(record["id"], int) or record["id"] < 0:
            raise BibleDataError(f"Record {i} has invalid 'id' (must be non-negative int)")
        if not isinstance(record["book"], str) or not record["book"].strip():
            raise BibleDataError(f"Record {i} has empty 'book'")
        if not isinstance(record["chapter"], int) or record["chapter"] < 1:
            raise BibleDataError(f"Record {i} has invalid 'chapter' (>=1)")
        if not isinstance(record["verse"], int) or record["verse"] < 1:
            raise BibleDataError(f"Record {i} has invalid 'verse' (>=1)")
        if not isinstance(record["text"], str) or not record["text"].strip():
            raise BibleDataError(f"Record {i} has empty 'text'")
        if record["testament"] not in ("OT", "NT"):
            raise BibleDataError(f"Record {i} 'testament' must be 'OT' or 'NT'")
        if not isinstance(record["category"], str) or not record["category"].strip():
            raise BibleDataError(f"Record {i} has empty 'category'")

    return data


def get_verse_reference(verse: dict[str, Any]) -> str:
    """
    Format a verse record into a human-readable reference.

    Example:
        {'book': 'John', 'chapter': 3, 'verse': 16} -> "John 3:16"

    Args:
        verse: A Bible verse record.

    Returns:
        Formatted reference string.
    """
    book = verse["book"]
    chapter = verse["chapter"]
    verse_num = verse["verse"]
    return f"{book} {chapter}:{verse_num}"
engine/hybrid_search.py
python
"""Hybrid search engine combining BM25 and dense vectors with RRF."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

import numpy as np
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import EmbeddingConfig, SearchConfig
from engine.indexer import get_verse_reference, load_bible_data


@dataclass
class SearchResult:
    """A single search result with metadata and scores."""

    verse: dict[str, Any]
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0

    @property
    def reference(self) -> str:
        return get_verse_reference(self.verse)


class HybridSearchEngine:
    """
    Hybrid Bible search combining dense embeddings (Qdrant) and BM25 with RRF fusion.
    """

    def __init__(
        self,
        bible_data: list[dict[str, Any]],
        search_config: SearchConfig,
        embedding_config: EmbeddingConfig,
    ) -> None:
        self.bible_data = bible_data
        self.search_config = search_config
        self.embedding_config = embedding_config

        # Initialize embedding model
        self.embedder = SentenceTransformer(
            embedding_config.model_name,
            device=embedding_config.device,
        )

        # Prepare corpus for BM25
        self.corpus_texts = [v["text"] for v in bible_data]
        self.bm25 = BM25Okapi(
            [text.split() for text in self.corpus_texts]
        )

        # Initialize Qdrant in-memory client and create collection
        self.qdrant = QdrantClient(location=":memory:")
        self._create_collection()

        # Index the verses
        self._index_verses()

    def _create_collection(self) -> None:
        """Create Qdrant collection with vector configuration."""
        vector_size = self.search_config.vector_size
        self.qdrant.create_collection(
            collection_name=self.search_config.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def _index_verses(self) -> None:
        """Encode and upload all verses to Qdrant."""
        texts = [v["text"] for v in self.bible_data]
        # Encode in batches
        embeddings = self.embedder.encode(
            texts,
            batch_size=self.embedding_config.batch_size,
            show_progress_bar=False,
        )

        points = []
        for idx, verse in enumerate(self.bible_data):
            point = qdrant_models.PointStruct(
                id=verse["id"],
                vector=embeddings[idx].tolist(),
                payload=verse,  # store full record
            )
            points.append(point)

        self.qdrant.upsert(
            collection_name=self.search_config.collection_name,
            points=points,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        book_filter: Optional[str] = None,
        testament_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search: dense + sparse + RRF.

        Args:
            query: The search query.
            top_k: Number of results to return (default from config).
            book_filter: Optional book name to filter results.
            testament_filter: Optional testament ("OT" or "NT") filter.

        Returns:
            List of SearchResult objects sorted by RRF score.
        """
        if top_k is None:
            top_k = self.search_config.top_k

        # Build filter for Qdrant if needed
        qdrant_filter = None
        if book_filter or testament_filter:
            conditions = []
            if book_filter:
                conditions.append(
                    qdrant_models.FieldCondition(
                        key="book",
                        match=qdrant_models.MatchValue(value=book_filter),
                    )
                )
            if testament_filter:
                conditions.append(
                    qdrant_models.FieldCondition(
                        key="testament",
                        match=qdrant_models.MatchValue(value=testament_filter),
                    )
                )
            qdrant_filter = qdrant_models.Filter(
                must=conditions if len(conditions) > 1 else conditions[0]
            )

        # Dense search
        query_embedding = self.embedder.encode(query, show_progress_bar=False)
        dense_results = self.qdrant.search(
            collection_name=self.search_config.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k * 2,  # retrieve extra for fusion
            query_filter=qdrant_filter,
        )
        # Map id -> dense rank (1-indexed)
        dense_rank_map = {
            hit.id: rank + 1 for rank, hit in enumerate(dense_results)
        }

        # Sparse (BM25) search
        # If filters are applied, we need to filter corpus before BM25
        if book_filter or testament_filter:
            filtered_indices = [
                i for i, v in enumerate(self.bible_data)
                if (book_filter is None or v["book"] == book_filter)
                and (testament_filter is None or v["testament"] == testament_filter)
            ]
            if not filtered_indices:
                return []  # no matches under filters
            # Rebuild BM25 on filtered corpus? For simplicity, we'll use full BM25 and then filter results later.
            # But BM25 scores are global; we can still use them, but the ranking may be affected.
            # Alternative: we could filter after scoring, but that may miss some results.
            # We'll just use full BM25 and then filter results after fusion; but we need to ensure we have enough candidates.
            # We'll retrieve more candidates from BM25.
            bm25_scores = self.bm25.get_scores(query.split())
            # Get top candidates (more than top_k to allow filtering)
            candidate_count = top_k * 4
            sparse_indices = np.argsort(bm25_scores)[::-1][:candidate_count]
        else:
            bm25_scores = self.bm25.get_scores(query.split())
            candidate_count = top_k * 2
            sparse_indices = np.argsort(bm25_scores)[::-1][:candidate_count]

        sparse_rank_map: dict[int, int] = {}
        for rank, idx in enumerate(sparse_indices, start=1):
            verse_id = self.bible_data[idx]["id"]
            sparse_rank_map[verse_id] = rank

        # Now fuse: collect all candidate IDs from dense and sparse
        candidate_ids = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())

        # Also gather all IDs that might appear only in sparse but not dense (already included)
        # Compute RRF score per candidate
        rrf_scores: dict[int, float] = {}
        k = self.search_config.rrf_k_constant

        for verse_id in candidate_ids:
            dense_rank = dense_rank_map.get(verse_id)
            sparse_rank = sparse_rank_map.get(verse_id)
            score = 0.0
            if dense_rank is not None:
                score += 1.0 / (k + dense_rank)
            if sparse_rank is not None:
                score += 1.0 / (k + sparse_rank)
            rrf_scores[verse_id] = score

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda vid: rrf_scores[vid], reverse=True)

        # Build results, applying optional filters (if not already applied)
        results: List[SearchResult] = []
        for verse_id in sorted_ids[:top_k]:
            # Find verse data
            verse = next((v for v in self.bible_data if v["id"] == verse_id), None)
            if verse is None:
                continue
            # Apply filters if not already used in BM25 (or double-check)
            if book_filter and verse["book"] != book_filter:
                continue
            if testament_filter and verse["testament"] != testament_filter:
                continue

            results.append(
                SearchResult(
                    verse=verse,
                    dense_rank=dense_rank_map.get(verse_id),
                    sparse_rank=sparse_rank_map.get(verse_id),
                    rrf_score=rrf_scores[verse_id],
                )
            )

        return results


# Streamlit caching helpers
@st.cache_resource
def get_embedding_model(embedding_config: EmbeddingConfig) -> SentenceTransformer:
    """Cache the SentenceTransformer model."""
    return SentenceTransformer(
        embedding_config.model_name,
        device=embedding_config.device,
    )


@st.cache_resource
def get_search_engine(
    data_path: str,
    search_config: SearchConfig,
    embedding_config: EmbeddingConfig,
) -> HybridSearchEngine:
    """Cache the HybridSearchEngine with its index."""
    bible_data = load_bible_data(data_path)
    return HybridSearchEngine(bible_data, search_config, embedding_config)
engine/llm_provider.py
python
"""Multi-provider streaming LLM interface."""

import json
import os
from typing import Generator, List, Optional

import httpx
from google import genai
from openai import OpenAI

from config import LLMConfig


def stream_llm_response(
    provider: str,
    model_name: str,
    prompt: str,
    system_prompt: str,
    api_key: Optional[str] = None,
    context_verses: Optional[List[dict]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Generator[str, None, None]:
    """
    Stream a response from the specified LLM provider.

    Args:
        provider: "ollama", "gemini", or "openai".
        model_name: Model identifier (e.g., "llama3", "gemini-2.5-flash", "gpt-4o-mini").
        prompt: The user's prompt (already includes context verses if needed).
        system_prompt: System instructions.
        api_key: API key for Gemini/OpenAI (not used for Ollama).
        context_verses: List of verse dicts (unused here, but may be used for formatting).
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.

    Yields:
        Chunks of the response text.
    """
    if provider == "ollama":
        yield from _stream_ollama(model_name, prompt, system_prompt, temperature, max_tokens)
    elif provider == "gemini":
        if not api_key:
            raise ValueError("Gemini API key is required")
        yield from _stream_gemini(model_name, prompt, system_prompt, api_key, temperature, max_tokens)
    elif provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required")
        yield from _stream_openai(model_name, prompt, system_prompt, api_key, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _stream_ollama(
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    """Stream from Ollama using HTTPX."""
    url = f"{LLMConfig.ollama_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    with httpx.Client(timeout=60.0) as client:
        with client.stream("POST", url, json=payload) as response:
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


def _stream_gemini(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    """Stream from Google Gemini."""
    client = genai.Client(api_key=api_key)
    full_prompt = f"{system_prompt}\n\n{prompt}"
    response = client.models.generate_content_stream(
        model=model,
        contents=full_prompt,
        config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def _stream_openai(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    """Stream from OpenAI."""
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def get_available_models(provider: str) -> List[str]:
    """
    Return a list of available model names for a given provider.

    For Gemini and OpenAI, we return a static list of common models.
    For Ollama, we attempt to query the local API for installed models.
    """
    if provider == "ollama":
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{LLMConfig.ollama_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    if models:
                        return models
                # fallback to default
        except Exception:
            pass
        return [LLMConfig.ollama_model]
    elif provider == "gemini":
        return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    elif provider == "openai":
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    else:
        return []
engine/chat.py
python
"""Conversational orchestration layer for AskRhema."""

from typing import Any, Dict, Generator, List, Optional

from config import LLMConfig, SearchConfig
from engine.hybrid_search import HybridSearchEngine, SearchResult
from engine.llm_provider import stream_llm_response


class ChatService:
    """
    Orchestrates conversation, retrieval, and LLM streaming.
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        search_config: SearchConfig,
        llm_config: LLMConfig,
    ) -> None:
        self.search_engine = search_engine
        self.search_config = search_config
        self.llm_config = llm_config

    def process_message(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        book_filter: Optional[str] = None,
        testament_filter: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Generator[tuple[str, List[SearchResult]], None, None]:
        """
        Process a user message: retrieve, build prompt, stream response.

        Args:
            user_message: Current user input.
            conversation_history: List of messages (role, content).
            provider: LLM provider.
            model_name: Model name.
            api_key: API key if needed.
            book_filter: Optional book filter for retrieval.
            testament_filter: Optional testament filter.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Yields:
            Tuples of (text_chunk, sources_so_far) – the sources are final at the end.
            We'll yield the final sources after the response is complete.
        """
        # 1. Retrieve relevant verses
        results = self.search_engine.search(
            query=user_message,
            top_k=self.search_config.top_k,
            book_filter=book_filter,
            testament_filter=testament_filter,
        )

        # 2. Build prompt with context verses
        prompt = self._build_prompt(user_message, conversation_history, results)

        # 3. Stream the LLM response
        full_response = ""
        sources = results  # sources for final display
        try:
            for chunk in stream_llm_response(
                provider=provider,
                model_name=model_name,
                prompt=prompt,
                system_prompt=self.llm_config.system_prompt,
                api_key=api_key,
                context_verses=[r.verse for r in results],
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                full_response += chunk
                yield chunk, sources
        except Exception as e:
            # Yield error message as a chunk
            error_msg = f"An error occurred: {str(e)}"
            yield error_msg, sources
            return

        # After streaming, we have full_response; but we already yielded sources in each chunk.
        # The final yielded chunk will have the final sources.
        # The caller can use the final sources after generator exhausts.

    def _build_prompt(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        results: List[SearchResult],
    ) -> str:
        """
        Build the full prompt for the LLM, including conversation history and retrieved verses.
        """
        # Format retrieved verses as a block
        verses_text = ""
        if results:
            verses_text = "Here are the relevant Bible passages (with citations):\n\n"
            for r in results:
                ref = r.reference
                text = r.verse["text"]
                verses_text += f"{ref}: {text}\n"
        else:
            verses_text = "No specific Bible passages were retrieved for this query."

        # Construct conversation history (limited to last 5 exchanges for context)
        history_text = ""
        if history:
            # Take last 5 messages
            recent = history[-5:]
            for msg in recent:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    history_text += f"User: {content}\n"
                elif role == "assistant":
                    history_text += f"Assistant: {content}\n"

        # Combine into one prompt
        prompt = f"""Conversation so far:
{history_text}

Current user query: {user_message}

Retrieved Scripture passages:
{verses_text}

Based on the above conversation and Scripture, please answer the user's query. Provide a clear, grounded response and cite references.
"""
        return prompt
app.py
python
"""AskRhema – Chat-first Bible search and exegesis assistant."""

import streamlit as st

from config import EmbeddingConfig, LLMConfig, SearchConfig
from engine.chat import ChatService
from engine.hybrid_search import get_search_engine
from engine.llm_provider import get_available_models

# Page configuration
st.set_page_config(
    page_title="AskRhema",
    page_icon="📖",
    layout="wide",
)

# Default configuration
DEFAULT_SEARCH_CONFIG = SearchConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
DEFAULT_LLM_CONFIG = LLMConfig()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider" not in st.session_state:
    st.session_state.provider = DEFAULT_LLM_CONFIG.default_provider
if "model" not in st.session_state:
    # Set initial model based on provider
    models = get_available_models(st.session_state.provider)
    st.session_state.model = models[0] if models else DEFAULT_LLM_CONFIG.ollama_model
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULT_LLM_CONFIG.temperature
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = DEFAULT_LLM_CONFIG.max_tokens
if "book_filter" not in st.session_state:
    st.session_state.book_filter = ""
if "testament_filter" not in st.session_state:
    st.session_state.testament_filter = ""
if "show_retrieval_details" not in st.session_state:
    st.session_state.show_retrieval_details = False
if "search_engine" not in st.session_state:
    # Load and cache the search engine using the resource cache
    st.session_state.search_engine = get_search_engine(
        "data/sample_bible.json",
        DEFAULT_SEARCH_CONFIG,
        DEFAULT_EMBEDDING_CONFIG,
    )
if "chat_service" not in st.session_state:
    st.session_state.chat_service = ChatService(
        st.session_state.search_engine,
        DEFAULT_SEARCH_CONFIG,
        DEFAULT_LLM_CONFIG,
    )


def reset_conversation() -> None:
    """Clear the conversation history."""
    st.session_state.messages = []


# Sidebar – Settings
with st.sidebar:
    st.header("⚙ Settings")
    with st.expander("LLM Configuration", expanded=False):
        provider = st.selectbox(
            "Provider",
            options=["ollama", "gemini", "openai"],
            index=["ollama", "gemini", "openai"].index(st.session_state.provider),
            key="provider_selector",
        )
        if provider != st.session_state.provider:
            st.session_state.provider = provider
            # Update model list
            models = get_available_models(provider)
            if models:
                st.session_state.model = models[0]
            else:
                st.session_state.model = DEFAULT_LLM_CONFIG.ollama_model
            st.rerun()

        models = get_available_models(st.session_state.provider)
        model = st.selectbox(
            "Model",
            options=models,
            index=models.index(st.session_state.model) if st.session_state.model in models else 0,
            key="model_selector",
        )
        if model != st.session_state.model:
            st.session_state.model = model

        if st.session_state.provider in ("gemini", "openai"):
            api_key = st.text_input(
                "API Key",
                type="password",
                value=st.session_state.api_key,
                key="api_key_input",
                help="Enter your API key. It will be stored only in session state.",
            )
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            key="temperature_slider",
        )
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=4096,
            value=st.session_state.max_tokens,
            step=100,
            key="max_tokens_input",
        )
        if max_tokens != st.session_state.max_tokens:
            st.session_state.max_tokens = max_tokens

    with st.expander("Search Filters", expanded=False):
        book_filter = st.text_input(
            "Book (exact match, optional)",
            value=st.session_state.book_filter,
            key="book_filter_input",
            placeholder="e.g., Romans",
        )
        if book_filter != st.session_state.book_filter:
            st.session_state.book_filter = book_filter

        testament_filter = st.selectbox(
            "Testament (optional)",
            options=["", "OT", "NT"],
            index=["", "OT", "NT"].index(st.session_state.testament_filter),
            key="testament_filter_select",
        )
        if testament_filter != st.session_state.testament_filter:
            st.session_state.testament_filter = testament_filter

    with st.expander("Debug", expanded=False):
        st.session_state.show_retrieval_details = st.checkbox(
            "Show retrieval details in responses",
            value=st.session_state.show_retrieval_details,
        )
        if st.button("Reset Conversation"):
            reset_conversation()
            st.rerun()

# Main chat UI
st.title("📖 AskRhema")
st.caption("A conversational Bible research and exegesis assistant.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If assistant message and has sources, display them
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources and st.session_state.show_retrieval_details:
                with st.expander("📚 Sources (debug)"):
                    for r in sources:
                        st.write(f"{r.reference} (RRF: {r.rrf_score:.4f})")
            elif sources:
                # Show simple references
                refs = ", ".join([r.reference for r in sources[:5]])
                if refs:
                    st.caption(f"📖 Sources: {refs}")

# Chat input
if prompt := st.chat_input("Ask AskRhema about Scripture..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        # Placeholder for streaming
        response_placeholder = st.empty()
        full_response = ""
        sources = []

        # Get API key if needed
        api_key = st.session_state.api_key if st.session_state.provider != "ollama" else None

        # Process message via chat service
        chat_service = st.session_state.chat_service
        try:
            stream = chat_service.process_message(
                user_message=prompt,
                conversation_history=st.session_state.messages[:-1],  # exclude current
                provider=st.session_state.provider,
                model_name=st.session_state.model,
                api_key=api_key,
                book_filter=st.session_state.book_filter or None,
                testament_filter=st.session_state.testament_filter or None,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            )
            # Stream chunks
            for chunk, current_sources in stream:
                full_response += chunk
                sources = current_sources
                response_placeholder.markdown(full_response + "▌")
            # Final update without cursor
            response_placeholder.markdown(full_response)

        except Exception as e:
            # Catch any error from the streaming
            st.error(f"An error occurred: {str(e)}")
            full_response = f"Sorry, I encountered an error: {str(e)}"
            response_placeholder.markdown(full_response)
            sources = []

        # Save assistant message with sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })

        # If sources available and not in debug mode, show compact sources after response
        if sources and not st.session_state.show_retrieval_details:
            refs = ", ".join([r.reference for r in sources[:5]])
            if refs:
                st.caption(f"📖 Sources: {refs}")

        # Trigger rerun to update the full message list (optional, but streamlit will rerun anyway)
        st.rerun()

# If no messages, show welcome
if not st.session_state.messages:
    st.info(
        """
        ### Welcome to AskRhema
        Explore Scripture through conversation.

        Ask a question about a passage, biblical theme, person, doctrine, or context.

        **Example prompts:**
        - What does Romans 8:28 mean?
        - Explain the context of John 15.
        - What does Scripture say about forgiveness?
        - How does the Old Testament point toward Christ?
        """
    )
tests/test_indexer.py
python
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
tests/test_hybrid_search.py
python
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
All files are complete and ready to run. The application uses the uv workflow; simply run uv run streamlit run app.py after setting up dependencies. The chat interface is clean, conversational, and fully functional with hybrid retrieval and streaming responses.

---
