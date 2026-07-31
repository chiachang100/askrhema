"""Configuration module for SeekRhema."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import logging
from pathlib import Path

# Initialize module-level logger for config
logger = logging.getLogger(__name__)
logger.info("Config module initialized successfully.")

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
        "You are SeekRhema, an expert biblical assistant. "
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
class LanguageConfig:
    """Configuration for language support."""
    default_language: str = "en"
    available_languages: Dict[str, str] = field(default_factory=lambda: {
        "en": "English",
        "zh-Hans": "简体中文",
        "zh-Hant": "繁體中文",
    })
    language_names: Dict[str, str] = field(default_factory=lambda: {
        "en": "English",
        "zh-Hans": "简体中文 (Simplified Chinese)",
        "zh-Hant": "繁體中文 (Traditional Chinese)",
    })
    
    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.default_language not in self.available_languages:
            raise ValueError(f"Default language {self.default_language} not in available languages")


# Default configuration instances
DEFAULT_SEARCH_CONFIG = SearchConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()
DEFAULT_LANGUAGE_CONFIG = LanguageConfig()

def get_system_prompt(language: str) -> str:
    logger.info(f"[config] Fetching system prompt for language: {language}")
    
    SYSTEM_PROMPTS = {
        "en": "You are SeekRhema, an expert, reverent, and scholarly biblical assistant. Ground your answer strictly in the provided Bible passages.",
        "zh-TW": "你是 SeekRhema，一位專業、敬虔且具備學者素養的聖經解經助手。請嚴格根據以下提供的聖經經文回答問題。",
        "zh-CN": "你是 SeekRhema，一位专业、敬虔且具备学者素养的圣经解经助手。请严格根据以下提供的圣经经文回答问题。"
    }

    ai_prompt = SYSTEM_PROMPTS["en"]

    if language in ["zh-TW", "zh-Hant"]:
        ai_prompt = SYSTEM_PROMPTS["zh-TW"]
    elif language in ["zh-CN", "zh-Hans", "zh"]:
        ai_prompt=  SYSTEM_PROMPTS["zh-CN"]

    logger.info(f"[config] Fetched system prompt: [{ai_prompt}].")

    return ai_prompt

def v1_get_system_prompt(language: str = "en") -> str:
    "Get the system prompt for AI exegesis in the specified language."
    if language == "zh-Hans":
        return (
            "您是SeekRhema，一位专业的圣经助手。"
            "请严格根据提供的圣经经文来回答。"
            "对于每一处经文引用，请务必注明书卷、章节和经文编号。"
            "提供全面、学术性的解经，同时保持平易近人。"
            "讨论神学概念时，要精确并引用经文。"
        )
    elif language == "zh-Hant":
        return (
            "您是SeekRhema，一位專業的聖經助手。"
            "請嚴格根據提供的聖經經文來回答。"
            "對於每一處經文引用，請務必註明書卷、章節和經文編號。"
            "提供全面、學術性的解經，同時保持平易近人。"
            "討論神學概念時，要精確並引用經文。"
        )
    else:
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
    "LanguageConfig",
    "DEFAULT_SEARCH_CONFIG",
    "DEFAULT_LLM_CONFIG",
    "DEFAULT_EMBEDDING_CONFIG",
    "DEFAULT_LANGUAGE_CONFIG",
    "get_system_prompt",
    "get_llm_config",
]