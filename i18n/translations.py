"""Translation management for AskRhema."""

import json
from pathlib import Path

import streamlit as st


class Translations:
    """Manages translations for the AskRhema application."""

    _instance: Translations | None = None
    _translations: dict[str, dict[str, str]] = {}
    _current_language: str = "en"
    _available_languages: dict[str, str] = {
        "en": "English",
        "zh-Hans": "简体中文 (Simplified Chinese)",
        "zh-Hant": "繁體中文 (Traditional Chinese)",
    }

    def __new__(cls) -> Translations:
        """Return the singleton Translations instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_translations()
        return cls._instance

    def _load_translations(self) -> None:
        """Load all translation files from the locales subdirectory."""
        locales_dir = Path(__file__).parent / "locales"

        for lang_code in self._available_languages:
            file_path = locales_dir / f"{lang_code}.json"
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        self._translations[lang_code] = json.load(f)
                except Exception as e:
                    print(f"Failed to load translations for {lang_code}: {e}")
                    self._translations[lang_code] = {}
            else:
                self._translations[lang_code] = {}

    def get(self, key: str, language: str | None = None) -> str:
        """Return a translated string for the specified dot-notation key."""
        lang = language or self._current_language

        if lang not in self._translations:
            lang = "en"

        value: object = self._translations[lang]

        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return key
            value = value[part]

        return value if isinstance(value, str) else key

    def set_language(self, language: str) -> None:
        """Set the current application language."""
        if language in self._available_languages:
            self._current_language = language
            st.session_state.language = language

    def get_current_language(self) -> str:
        """Return the current application language."""
        return st.session_state.get("language", "en")

    def get_available_languages(self) -> dict[str, str]:
        """Return the available language codes and names."""
        return self._available_languages

    def get_language_name(self, code: str) -> str:
        """Return the display name for a language code."""
        return self._available_languages.get(code, code)


def get_translations() -> Translations:
    """Return the translations singleton."""
    return Translations()


def get_language_name(lang_code: str) -> str:
    """Return the display name for a language code."""
    return Translations().get_language_name(lang_code)


def get_available_languages() -> dict[str, str]:
    """Return the available language mappings."""
    return Translations().get_available_languages()
