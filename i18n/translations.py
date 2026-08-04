"""Translation management for AskRhema."""

import json
from pathlib import Path
from typing import Any

import streamlit as st


class Translations:
    """Manages translations for the AskRhema application."""

    _instance: Translations | None = None

    _available_languages: dict[str, str] = {
        "en": "English",
        "zh-Hans": "简体中文 (Simplified Chinese)",
        "zh-Hant": "繁體中文 (Traditional Chinese)",
    }

    def __new__(cls) -> Translations:
        """Return the singleton Translations instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the translations manager once."""
        if getattr(self, "_initialized", False):
            return

        self._translations: dict[str, dict[str, Any]] = {}
        self._current_language = "en"
        self._initialized = True

        self._load_translations()

    def _load_translations(self) -> None:
        """Load all translation files from the locales subdirectory."""
        locales_dir = Path(__file__).parent / "locales"

        for lang_code in self._available_languages:
            file_path = locales_dir / f"{lang_code}.json"

            if not file_path.exists():
                self._translations[lang_code] = {}
                continue

            try:
                with file_path.open(encoding="utf-8") as file:
                    data = json.load(file)

                if isinstance(data, dict):
                    self._translations[lang_code] = data
                else:
                    print(
                        f"Invalid translation format for {lang_code}: "
                        "expected a JSON object."
                    )
                    self._translations[lang_code] = {}

            except (OSError, json.JSONDecodeError) as exc:
                print(f"Failed to load translations for {lang_code}: {exc}")
                self._translations[lang_code] = {}

    def _lookup(self, key: str, language: str) -> str | None:
        """Look up a dot-notation translation key for a language."""
        value: Any = self._translations.get(language, {})

        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None

            value = value[part]

        return value if isinstance(value, str) else None

    def get(self, key: str, language: str | None = None) -> str:
        """Return a translated string using English as a fallback."""
        lang = language or self._current_language

        if lang not in self._available_languages:
            lang = "en"

        # Try the requested language first.
        translation = self._lookup(key, lang)
        if translation is not None:
            return translation

        # Fall back to English for missing translations.
        if lang != "en":
            translation = self._lookup(key, "en")
            if translation is not None:
                return translation

        # Return the key itself when no translation exists.
        return key

    def set_language(self, language: str) -> None:
        """Set the current application language."""
        if language not in self._available_languages:
            return

        self._current_language = language
        st.session_state.language = language

    def get_current_language(self) -> str:
        """Return the current application language."""
        return self._current_language

    def get_available_languages(self) -> dict[str, str]:
        """Return the available language codes and names."""
        return self._available_languages.copy()

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
