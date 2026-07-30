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


# Default configuration instances
DEFAULT_SEARCH_CONFIG = SearchConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()


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
    "DEFAULT_SEARCH_CONFIG",
    "DEFAULT_LLM_CONFIG",
    "DEFAULT_EMBEDDING_CONFIG",
    "get_system_prompt",
    "get_llm_config",
]