# i18n.py

AVAILABLE_LANGUAGES = {"en": "English", "zh-TW": "繁體中文", "zh-CN": "简体中文"}

TRANSLATIONS = {
    "en": {
        "ui.ai.prompt_template": "Please provide a thorough exegetical analysis of the following Bible passages for the query: '{query}'",
        "ui.ai.structure_summary": "1. **Summary**",
        "ui.ai.structure_themes": "2. **Theological Themes**",
        "ui.ai.structure_context": "3. **Historical Context**",
        "ui.ai.structure_application": "4. **Practical Application**",
    },
    "zh-TW": {
        "ui.ai.prompt_template": "請根據搜尋查詢「{query}」對以下聖經經文提供深入的釋經分析：",
        "ui.ai.structure_summary": "1. **摘要**",
        "ui.ai.structure_themes": "2. **神學主題**",
        "ui.ai.structure_context": "3. **歷史背景**",
        "ui.ai.structure_application": "4. **生活應用**",
    },
    "zh-CN": {
        "ui.ai.prompt_template": "请根据搜索查询“{query}”对以下圣经经文提供深入的释经分析：",
        "ui.ai.structure_summary": "1. **摘要**",
        "ui.ai.structure_themes": "2. **神学主题**",
        "ui.ai.structure_context": "3. **历史背景**",
        "ui.ai.structure_application": "4. **实际应用**",
    },
}
