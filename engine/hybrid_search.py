"""Hybrid search engine combining BM25 and dense vectors with RRF."""

from dataclasses import dataclass
from typing import Any

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
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0

    @property
    def reference(self) -> str:
        """Return the formatted Bible verse reference."""
        return get_verse_reference(self.verse)


class HybridSearchEngine:
    """Hybrid search combining dense embeddings (Qdrant) and BM25 with RRF fusion."""

    def __init__(
        self,
        bible_data: list[dict[str, Any]],
        search_config: SearchConfig,
        embedding_config: EmbeddingConfig,
    ) -> None:
        self.bible_data = bible_data
        self.search_config = search_config
        self.embedding_config = embedding_config

        self.embedder = SentenceTransformer(
            embedding_config.model_name,
            device=embedding_config.device,
        )
        self.corpus_texts = [v["text"] for v in bible_data]
        self.bm25 = BM25Okapi([text.split() for text in self.corpus_texts])

        self.qdrant = QdrantClient(location=":memory:")
        self._create_collection()
        self._index_verses()

    def _create_collection(self) -> None:
        vector_size = self.search_config.vector_size
        self.qdrant.create_collection(
            collection_name=self.search_config.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def _index_verses(self) -> None:
        texts = [v["text"] for v in self.bible_data]
        embeddings = self.embedder.encode(
            texts,
            batch_size=self.embedding_config.batch_size,
            show_progress_bar=False,
        )
        points = [
            qdrant_models.PointStruct(
                id=verse["id"],
                vector=embeddings[idx].tolist(),
                payload=verse,
            )
            for idx, verse in enumerate(self.bible_data)
        ]
        self.qdrant.upsert(
            collection_name=self.search_config.collection_name,
            points=points,
        )

    def _dense_search(
        self,
        query_vector: list[float],
        limit: int,
        qdrant_filter: qdrant_models.Filter | None,
    ) -> list[Any]:
        """Perform dense vector search using Qdrant's current query API."""
        response = self.qdrant.query_points(
            collection_name=self.search_config.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
        )
        return response.points

    def search(
        self,
        query: str,
        top_k: int | None = None,
        book_filter: str | None = None,
        testament_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search for relevant Bible passages matching the query."""
        if top_k is None:
            top_k = self.search_config.top_k

        # Build Qdrant filter
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
        dense_results = self._dense_search(
            query_vector=query_embedding.tolist(),
            limit=top_k * 2,
            qdrant_filter=qdrant_filter,
        )
        dense_rank_map = {hit.id: rank + 1 for rank, hit in enumerate(dense_results)}

        # Sparse (BM25) search
        bm25_scores = self.bm25.get_scores(query.split())
        candidate_count = top_k * 4 if (book_filter or testament_filter) else top_k * 2
        sparse_indices = np.argsort(bm25_scores)[::-1][:candidate_count]
        sparse_rank_map = {}
        for rank, idx in enumerate(sparse_indices, start=1):
            verse_id = self.bible_data[idx]["id"]
            sparse_rank_map[verse_id] = rank

        # Fusion (RRF)
        candidate_ids = set(dense_rank_map.keys()) | set(sparse_rank_map.keys())
        rrf_scores = {}
        k = self.search_config.rrf_k_constant
        for verse_id in candidate_ids:
            score = 0.0
            if verse_id in dense_rank_map:
                score += 1.0 / (k + dense_rank_map[verse_id])
            if verse_id in sparse_rank_map:
                score += 1.0 / (k + sparse_rank_map[verse_id])
            rrf_scores[verse_id] = score

        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda vid: rrf_scores[vid], reverse=True
        )

        results = []
        for verse_id in sorted_ids[:top_k]:
            verse = next((v for v in self.bible_data if v["id"] == verse_id), None)
            if verse is None:
                continue
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
    """Create and cache the sentence-transformer embedding model."""
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
    """Create and cache the hybrid search engine."""
    bible_data = load_bible_data(data_path)
    return HybridSearchEngine(bible_data, search_config, embedding_config)
