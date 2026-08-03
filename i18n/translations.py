"""Translation management for AskRhema."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st


class Translations:
    """Manages translations for the AskRhema application."""

    _instance: Optional['Translations'] = None
    _translations: Dict[str, Dict[str, str]] = {}
    _current_language: str = "en"
    _available_languages: Dict[str, str] = {
        "en": "English",
        "zh-Hans": "简体中文 (Simplified Chinese)",
        "zh-Hant": "繁體中文 (Traditional Chinese)",
    }

    def __new__(cls) -> 'Translations':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_translations()
        return cls._instance

    def _load_translations(self) -> None:
        """Load all translation files from the locales subdirectory."""
        locales_dir = Path(__file__).parent / "locales"

        for lang_code in self._available_languages.keys():
            file_path = locales_dir / f"{lang_code}.json"
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self._translations[lang_code] = json.load(f)
                except Exception as e:
                    print(f"Failed to load translations for {lang_code}: {e}")
                    self._translations[lang_code] = {}
            else:
                self._translations[lang_code] = {}

    def get(self, key: str, language: Optional[str] = None) -> str:
        lang = language or self._current_language
        if lang not in self._translations:
            return key

        parts = key.split('.')
        value = self._translations[lang]
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return key
        return str(value) if value else key

    def set_language(self, language: str) -> None:
        if language in self._available_languages:
            self._current_language = language
            st.session_state.language = language

    def get_current_language(self) -> str:
        return st.session_state.get("language", "en")

    def get_available_languages(self) -> Dict[str, str]:
        return self._available_languages

    def get_language_name(self, code: str) -> str:
        return self._available_languages.get(code, code)


def get_translations() -> Translations:
    return Translations()


def get_language_name(lang_code: str) -> str:
    return Translations().get_language_name(lang_code)


def get_available_languages() -> Dict[str, str]:
    return Translations().get_available_languages()