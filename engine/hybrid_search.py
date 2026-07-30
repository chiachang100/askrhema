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
import logging

from config import DEFAULT_SEARCH_CONFIG, DEFAULT_EMBEDDING_CONFIG, SearchConfig

# Setup logger
logger = logging.getLogger(__name__)


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
        self._verses: List[Dict[str, Any]] = []
        self._is_initialized: bool = False
        
        # Tokenized corpora for BM25
        self._tokenized_corpus: List[List[str]] = []
        self._verse_id_to_index: Dict[int, int] = {}
        self._index_to_verse_id: Dict[int, int] = {}
    
    @st.cache_resource
    def _get_embedding_model(_self) -> SentenceTransformer:
        """Get or load the sentence transformer model with caching."""
        try:
            logger.info(f"Loading embedding model: {DEFAULT_EMBEDDING_CONFIG.model_name}")
            model = SentenceTransformer(DEFAULT_EMBEDDING_CONFIG.model_name)
            logger.info("Embedding model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    @st.cache_resource
    def _get_qdrant_client(_self) -> QdrantClient:
        """Get or create the Qdrant in-memory client with caching."""
        try:
            logger.info("Creating Qdrant in-memory client")
            client = QdrantClient(":memory:")
            logger.info("Qdrant client created successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to create Qdrant client: {str(e)}")
            raise RuntimeError(f"Failed to create Qdrant client: {str(e)}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        return text.lower().split()
    
    def initialize(self, verses: List[Dict[str, Any]]) -> None:
        """
        Initialize the search engine with Bible data.
        
        Args:
            verses: List of verse dictionaries
        """
        if self._is_initialized:
            logger.info("Search engine already initialized")
            return
            
        if not verses:
            raise ValueError("No verses provided for initialization")
        
        self._verses = verses
        logger.info(f"Initializing search engine with {len(verses)} verses")
        
        # Get cached resources
        try:
            logger.info("Loading embedding model...")
            self._embedding_model = self._get_embedding_model()
            if self._embedding_model is None:
                raise RuntimeError("Embedding model is None after loading")
            
            logger.info("Creating Qdrant client...")
            self._qdrant_client = self._get_qdrant_client()
            if self._qdrant_client is None:
                raise RuntimeError("Qdrant client is None after creation")
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
        
        # Prepare data for indexing
        texts = [verse["text"] for verse in verses]
        self._tokenized_corpus = [self._tokenize_text(text) for text in texts]
        
        # Build index mappings
        for idx, verse in enumerate(verses):
            verse_id = verse["id"]
            self._verse_id_to_index[verse_id] = idx
            self._index_to_verse_id[idx] = verse_id
        
        # Create BM25 index
        try:
            logger.info("Creating BM25 index...")
            self._bm25_index = BM25Okapi(self._tokenized_corpus)
            if self._bm25_index is None:
                raise RuntimeError("BM25 index is None after creation")
            logger.info("BM25 index created successfully")
        except Exception as e:
            logger.error(f"Failed to create BM25 index: {str(e)}")
            raise RuntimeError(f"Failed to create BM25 index: {str(e)}")
        
        # Create Qdrant collection
        try:
            logger.info("Creating Qdrant collection...")
            self._create_qdrant_collection(verses)
            logger.info("Qdrant collection created successfully")
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {str(e)}")
            raise RuntimeError(f"Failed to create Qdrant collection: {str(e)}")
        
        self._is_initialized = True
        logger.info("Search engine initialization complete!")
    
    def _ensure_initialized(self) -> None:
        """
        Ensure the search engine is initialized with all components.
        Raises RuntimeError if any component is missing.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Search engine not initialized. Call initialize() first."
            )
        
        if self._qdrant_client is None:
            raise RuntimeError(
                "Qdrant client is not initialized. The engine may not have been properly initialized."
            )
        
        if self._bm25_index is None:
            raise RuntimeError(
                "BM25 index is not initialized. The engine may not have been properly initialized."
            )
        
        if self._embedding_model is None:
            raise RuntimeError(
                "Embedding model is not initialized. The engine may not have been properly initialized."
            )
        
        if not self._verses:
            raise RuntimeError(
                "No verses loaded. The engine may not have been properly initialized."
            )
        
        if not self._tokenized_corpus:
            raise RuntimeError(
                "Tokenized corpus is empty. The engine may not have been properly initialized."
            )
        
        if not self._index_to_verse_id:
            raise RuntimeError(
                "Index mapping is empty. The engine may not have been properly initialized."
            )
    
    def _create_qdrant_collection(self, verses: List[Dict[str, Any]]) -> None:
        """Create Qdrant collection and index verses."""
        # Ensure client exists
        if self._qdrant_client is None:
            raise RuntimeError("Qdrant client is None. Cannot create collection.")
        
        if self._embedding_model is None:
            raise RuntimeError("Embedding model is None. Cannot encode verses.")
        
        collection_name = self.config.collection_name
        vector_size = self.config.vector_size
        
        # Check if collection exists and delete it
        try:
            collections = self._qdrant_client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                logger.info(f"Deleting existing collection: {collection_name}")
                self._qdrant_client.delete_collection(collection_name)
        except Exception as e:
            logger.warning(f"Could not check/delete existing collection: {str(e)}")
            # Continue anyway - the collection might not exist
        
        # Create collection
        try:
            logger.info(f"Creating collection: {collection_name} with size {vector_size}")
            self._qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create Qdrant collection: {str(e)}")
        
        # Index each verse
        points = []
        total_verses = len(verses)
        logger.info(f"Encoding {total_verses} verses...")
        
        for i, verse in enumerate(verses):
            try:
                # Generate embedding with null check
                if self._embedding_model is None:
                    raise RuntimeError("Embedding model became None during processing")
                
                # Show progress
                if (i + 1) % 10 == 0 or i == total_verses - 1:
                    logger.info(f"Encoded {i + 1}/{total_verses} verses")
                
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
            except Exception as e:
                raise RuntimeError(f"Failed to encode verse {verse['id']} ({verse['book']} {verse['chapter']}:{verse['verse']}): {str(e)}")
        
        # Upload points in batches
        if not points:
            raise RuntimeError("No points were created from verses")
        
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        logger.info(f"Uploading {len(points)} points in {total_batches} batches...")
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            batch_num = i // batch_size + 1
            try:
                self._qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch {batch_num}/{total_batches}")
            except Exception as e:
                raise RuntimeError(f"Failed to upload batch {batch_num}: {str(e)}")
        
        logger.info("All verses indexed successfully!")
    
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
        self._ensure_initialized()
        
        if self._embedding_model is None:
            raise RuntimeError("Embedding model is None")
        
        if self._qdrant_client is None:
            raise RuntimeError("Qdrant client is None")
        
        # Generate query embedding
        query_embedding = self._embedding_model.encode(query).tolist()
        
        # Build filter
        qdrant_filter = self._build_filter(book, testament)
        
        search_results = []
        
        # Try different API versions
        try:
            # Try qdrant-client >= 1.7.0 with query_points
            response = self._qdrant_client.query_points(
                collection_name=self.config.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter
            )
            search_results = response.points if hasattr(response, 'points') else []
        except (AttributeError, TypeError) as e:
            try:
                # Try older version with search
                search_results = self._qdrant_client.search(
                    collection_name=self.config.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    query_filter=qdrant_filter
                )
            except Exception as e2:
                # Try with different parameter names for even older versions
                try:
                    search_results = self._qdrant_client.search(
                        collection_name=self.config.collection_name,
                        vector=query_embedding,
                        limit=top_k,
                        filter=qdrant_filter
                    )
                except Exception as e3:
                    raise RuntimeError(f"Qdrant search failed: {str(e3)}")
        
        # Extract results
        results = []
        for rank, hit in enumerate(search_results):
            if hasattr(hit, 'id'):
                verse_id = hit.id
                score = hit.score if hasattr(hit, 'score') else 0.0
            elif isinstance(hit, dict):
                verse_id = hit.get('id')
                score = hit.get('score', 0.0)
            else:
                continue
                
            if verse_id is not None:
                results.append((verse_id, float(score), rank + 1))
        
        logger.debug(f"Dense search returned {len(results)} results")
        return results
    
    def _perform_sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float, int]]:
        """Perform sparse BM25 search."""
        self._ensure_initialized()
        
        # Double-check that _bm25_index is not None (type guard)
        if self._bm25_index is None:
            raise RuntimeError("BM25 index is None. This should not happen after _ensure_initialized().")
        
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        
        # Get BM25 scores
        try:
            scores = self._bm25_index.get_scores(query_tokens)
        except Exception as e:
            raise RuntimeError(f"BM25 search failed: {str(e)}")
        
        # Get top_k results
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices):
            score = scores[idx]
            if score > 0:
                verse_id = self._index_to_verse_id[idx]
                results.append((verse_id, float(score), rank + 1))
        
        logger.debug(f"Sparse search returned {len(results)} results")
        return results
    
    def _apply_filters_to_results(self, results: List[SearchResult], 
                                   book: Optional[str] = None, 
                                   testament: Optional[str] = None) -> List[SearchResult]:
        """
        Apply filters to search results (post-filtering for sparse search).
        
        Args:
            results: List of SearchResult objects
            book: Filter by book name
            testament: Filter by testament (OT or NT)
            
        Returns:
            Filtered list of SearchResult objects
        """
        if not book and not testament:
            return results
        
        filtered_results = []
        for result in results:
            if book and result.book != book:
                continue
            if testament and result.testament != testament:
                continue
            filtered_results.append(result)
        
        return filtered_results
    
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
        self._ensure_initialized()
        
        if not self._verses:
            logger.warning("No verses loaded")
            return []
        
        top_k = top_k or self.config.top_k
        k_constant = self.config.rrf_k_constant
        
        logger.info(f"Searching for: '{query}' with top_k={top_k}, book={book}, testament={testament}")
        
        # Perform dense search (with filters applied at Qdrant level)
        dense_results = self._perform_dense_search(query, top_k * 3, book, testament)
        
        # Perform sparse search (without filters, then apply post-filtering)
        sparse_results = self._perform_sparse_search(query, top_k * 3)
        
        # Combine results using RRF
        all_verse_ids: Set[int] = set()
        all_verse_ids.update(verse_id for verse_id, _, _ in dense_results)
        all_verse_ids.update(verse_id for verse_id, _, _ in sparse_results)
        
        if not all_verse_ids:
            logger.info("No results found")
            return []
        
        # Calculate RRF scores
        dense_dict = {verse_id: (score, rank) for verse_id, score, rank in dense_results}
        sparse_dict = {verse_id: (score, rank) for verse_id, score, rank in sparse_results}
        
        rrf_scores = {}
        dense_ranks = {}
        sparse_ranks = {}
        
        for verse_id in all_verse_ids:
            rrf_score = 0.0
            
            if verse_id in dense_dict:
                _, rank = dense_dict[verse_id]
                rrf_score += 1.0 / (k_constant + rank)
                dense_ranks[verse_id] = rank
            else:
                dense_ranks[verse_id] = top_k + 1
            
            if verse_id in sparse_dict:
                _, rank = sparse_dict[verse_id]
                rrf_score += 1.0 / (k_constant + rank)
                sparse_ranks[verse_id] = rank
            else:
                sparse_ranks[verse_id] = top_k + 1
            
            rrf_scores[verse_id] = rrf_score
        
        # Sort by RRF score
        sorted_verse_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
        
        # Create SearchResult objects
        verse_lookup = {v["id"]: v for v in self._verses}
        results = []
        
        for verse_id, fused_score in sorted_verse_ids:
            verse = verse_lookup.get(verse_id)
            if verse is None:
                continue
            
            dense_score = dense_dict.get(verse_id, (0.0, 0))[0]
            sparse_score = sparse_dict.get(verse_id, (0.0, 0))[0]
            
            results.append(SearchResult(
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
            ))
        
        # Apply post-filtering for sparse search results
        if book or testament:
            results = self._apply_filters_to_results(results, book, testament)
        
        # Limit to top_k
        results = results[:top_k]
        
        logger.info(f"Search returned {len(results)} results")
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
        self._ensure_initialized()
        
        if not self._verses:
            return []
        
        top_k = top_k or self.config.top_k
        
        dense_results = self._perform_dense_search(query, top_k, book, testament)
        
        verse_lookup = {v["id"]: v for v in self._verses}
        results = []
        
        for verse_id, dense_score, dense_rank in dense_results:
            verse = verse_lookup.get(verse_id)
            if verse is None:
                continue
            
            results.append(SearchResult(
                id=verse_id,
                book=verse["book"],
                chapter=verse["chapter"],
                verse=verse["verse"],
                text=verse["text"],
                testament=verse["testament"],
                category=verse["category"],
                dense_score=dense_score,
                sparse_score=0.0,
                fused_score=dense_score,
                dense_rank=dense_rank,
                sparse_rank=0
            ))
        
        return results
    
    def is_initialized(self) -> bool:
        """Check if the search engine is initialized."""
        return self._is_initialized
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the search engine for debugging.
        
        Returns:
            Dictionary with status information
        """
        return {
            "is_initialized": self._is_initialized,
            "has_embedding_model": self._embedding_model is not None,
            "has_qdrant_client": self._qdrant_client is not None,
            "has_bm25_index": self._bm25_index is not None,
            "num_verses": len(self._verses),
            "has_tokenized_corpus": len(self._tokenized_corpus) > 0,
            "has_index_mapping": len(self._index_to_verse_id) > 0,
            "collection_name": self.config.collection_name if self._is_initialized else None,
        }