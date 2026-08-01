# I18N Support

To support multiple languages (English, Traditional Chinese, and Simplified Chinese) in AskRhema, you'll need to implement internationalization (i18n). Here's a comprehensive solution:

## 1. Create Language Configuration Files
Create i18n/ Directory Structure
```text
askrhema/
├── i18n/
│   ├── __init__.py
│   ├── translations.py
│   ├── locales/
│   │   ├── en.json
│   │   ├── zh-Hans.json
│   │   └── zh-Hant.json
```

- i18n/__init__.py
- i18n/translations.py
- i18n/translations.py
- i18n/locales/en.json
- i18n/locales/zh-Hans.json
- i18n/locales/zh-Hant.json


1. Update `config.py` to Support Languages
python
# config.py (updated with language support)

3. Update app.py with Language Support
Here's the updated app.py with language support:

- app.py (updated with language support)

4. Update `pyproject.toml` to Include i18n Module

5. Summary
This implementation provides:

Three languages: English, Simplified Chinese, and Traditional Chinese

Language selector in the sidebar

Comprehensive translations for all UI elements

Language-specific AI prompts for better exegesis in each language

Translation system that's easy to extend with more languages

Session state persistence for language preference

To add more languages, simply:

1. Add a new JSON file in i18n/locales/
2. Add the language code to LanguageConfig.available_languages in config.py
3. Add translations for all keys in the new JSON file

---
