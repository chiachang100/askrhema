"""Configuration module for AskRhema."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

# Initialize module-level logger for config
logger = logging.getLogger(__name__)
logger.info("Config module initialized successfully.")


@dataclass(frozen=True)
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

    # ollama_url: str = "http://localhost:11434/api/generate"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_default_model: str = "llama3"
    # google_model: str = "gemini-2.5-flash"
    gemini_model: str = "gemini-2.5-flash"
    openai_model: str = "gpt-4o-mini"

    # Provider-specific defaults; user may override via settings.
    default_provider: str = os.getenv("ASKRHEMA_DEFAULT_PROVIDER", "ollama")

    temperature: float = 0.7
    max_tokens: int = 1024
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


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for embedding model."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # or "cuda"
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
class LanguageConfig:
    """Configuration for language support."""

    default_language: str = "en"
    available_languages: dict[str, str] = field(
        default_factory=lambda: {
            "en": "English",
            "zh-Hans": "简体中文",
            "zh-Hant": "繁體中文",
        }
    )
    language_names: dict[str, str] = field(
        default_factory=lambda: {
            "en": "English",
            "zh-Hans": "简体中文 (Simplified Chinese)",
            "zh-Hant": "繁體中文 (Traditional Chinese)",
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.default_language not in self.available_languages:
            raise ValueError(
                f"Default language {self.default_language} not in available languages"
            )


# Default configuration instances
DEFAULT_SEARCH_CONFIG = SearchConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
DEFAULT_LANGUAGE_CONFIG = LanguageConfig()


SYSTEM_PROMPTS = {
    "en": "You are AskRhema, an expert, reverent, and scholarly biblical assistant. Ground your answer strictly in the provided Bible passages.",
    "zh-Hant": "你是 AskRhema，一位專業、敬虔且具備學者素養的聖經解經助手。請嚴格根據以下提供的聖經經文回答問題。",
    "zh-Hans": "你是 AskRhema，一位专业、敬虔且具备学者素养的圣经解经助手。请严格根据以下提供的圣经经文回答问题。",
}


def get_system_prompt(language: str) -> str:
    """Return the system prompt for the specified language."""
    logger.info(f"[config] Fetching system prompt for language: {language}")

    ai_prompt = SYSTEM_PROMPTS["en"]

    if language in ["zh-Hant", "zh-TW"]:
        ai_prompt = SYSTEM_PROMPTS["zh-Hant"]
    elif language in ["zh-Hans", "zh-CN", "zh"]:
        ai_prompt = SYSTEM_PROMPTS["zh-Hans"]

    logger.info(f"[config] Fetched system prompt: [{ai_prompt}].")

    return ai_prompt


def get_llm_config(provider: str) -> dict[str, Any]:
    """Get LLM configuration for a specific provider."""
    config = {
        "ollama": {
            "url": DEFAULT_LLM_CONFIG.ollama_url,
            "default_model": DEFAULT_LLM_CONFIG.ollama_default_model,
        },
        "gemini": {
            "model": DEFAULT_LLM_CONFIG.gemini_model,
        },
        "openai": {
            "model": DEFAULT_LLM_CONFIG.openai_model,
        },
    }
    return config.get(provider, {})


__all__ = [
    "SearchConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "LanguageConfig",
    "DEFAULT_SEARCH_CONFIG",
    "DEFAULT_LLM_CONFIG",
    "DEFAULT_EMBEDDING_CONFIG",
    "DEFAULT_LANGUAGE_CONFIG",
    "get_system_prompt",
    "get_llm_config",
]
