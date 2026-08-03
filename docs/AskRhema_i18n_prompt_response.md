# AskRhema i18n Prompt and Response

To support multi-language prompts and responses for English, Traditional Chinese (zh-TW), and Simplified Chinese (zh-CN) in AskRhema, you need to align three layers:

- Translation Infrastructure (`i18n.py`) — Storing UI strings and prompts.
- System Prompts (`config.py`) — Guiding the LLM to respond in the selected language.
- Dynamic Prompt Assembly (`app.py` / `engine/llm_provider.py`) — Injecting context in the user's language.

## Step 1: Update System Prompts in config.py
Add distinct system prompts tuned for each target locale so the LLM automatically matches the chosen language:

Python
# config.py

SYSTEM_PROMPTS = {
    "en": """You are AskRhema, an expert, reverent, and scholarly biblical assistant.
Ground your answer strictly in the provided Bible passages. Always cite the exact Book, Chapter, and Verse numbers.
Respond in clear, accessible English.""",

    "zh-TW": """你是 AskRhema，一位專業、敬虔且具備學者素養的聖經解經助手。
請嚴格根據以下提供的聖經經文回答問題。每次引用經文時，必須明確標註書卷、章節與節數。
請使用繁體中文（台灣習慣用語）進行回答。""",

    "zh-CN": """你是 AskRhema，一位专业、敬虔且具备学者素养的圣经解经助手。
请严格根据以下提供的圣经经文回答问题。每次引用经文时，必须明确标注书卷、章节与节数。
请使用简体中文进行回答。"""
}

def get_system_prompt(lang: str) -> str:
    return SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
Step 2: Language Dictionary in i18n.py
Ensure your language configuration supports all three language keys: en, zh-TW, and zh-CN.

Python
# i18n.py

AVAILABLE_LANGUAGES = {
    "en": "English",
    "zh-TW": "繁體中文",
    "zh-CN": "简体中文"
}

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
    }
}
Step 3: Dynamic User Prompt Generation in app.py
Replace the hardcoded if language.startswith("zh") block in generate_ai_response() with a dynamic localized template based on st.session_state.language:

Python
def build_exegetical_prompt(query: str, context_verses: list[dict], lang: str) -> str:
    """Build a language-specific user prompt for the LLM."""
    verses_text = "\n".join([f"- {v['book']} {v['chapter']}:{v['verse']} - {v['text']}" for v in context_verses])

    if lang == "zh-TW":
        return f"""請根據查詢「{query}」檢索出的以下聖經經文，提供嚴謹且詳細的釋經分析：

【上下文經文】：
{verses_text}

請按以下結構回答：
1. **摘要**：簡要說明經文與查詢的關聯。
2. **神學主題**：經文中展現的核心神學教義。
3. **歷史與文化背景**：相關的歷史背景與上下文脈絡。
4. **實際應用**：如何在現代基督徒生活中回應與應用。

請使用繁體中文回答，確保所有見解均以提供的經文為依據，並標註相應的經文出處。"""

    elif lang == "zh-CN":
        return f"""请根据查询“{query}”检索出的以下圣经经文，提供严谨且详细的释经分析：

【上下文经文】：
{verses_text}

请按以下结构回答：
1. **摘要**：简要说明经文与查询的关联。
2. **神学主题**：经文中展现的核心神学教义。
3. **历史与文化背景**：相关的历史背景与上下文脉络。
4. **实际应用**：如何在现代基督徒生活中回应与应用。

请使用简体中文回答，确保所有见解均以提供的经文为依据，并标注相应的经文出处。"""

    else:  # Default to English
        return f"""Please provide a thorough exegetical analysis of the following Bible passages retrieved for the query: "{query}"

[Context Passages]:
{verses_text}

Please structure your response as follows:
1. **Summary**: Brief overview of passages and their relevance.
2. **Theological Themes**: Key theological concepts present.
3. **Historical Context**: Relevant historical and cultural background.
4. **Practical Application**: How to apply these truths today.

Provide proper scripture citations and ground all insights strictly in the provided text."""
Step 4: Connecting it inside generate_ai_response()
Now simply retrieve the localized prompt and system prompt inside app.py:

Python
# Inside generate_ai_response() in app.py

current_lang = st.session_state.get("language", "en")

# 1. Fetch language-specific system prompt
system_prompt = get_system_prompt(current_lang)

# 2. Build language-specific user prompt
user_prompt = build_exegetical_prompt(query, context_verses, current_lang)

# 3. Stream from LLM
for chunk in stream_llm_response(
    provider=provider,
    model_name=model,
    prompt=user_prompt,
    system_prompt=system_prompt,
    api_key=api_key,
    temperature=0.3  # Lower temperature keeps exegesis factual and grounded
):
    full_response += chunk
    # Update Streamlit UI container...
Key Benefits of This Approach
Zero Hallucination Language Drift: Specifying the requested target language explicitly in both the System Prompt and the User Prompt prevents the LLM from unexpectedly switching back to English midway through streaming.

Proper Terminology Usage: Traditional Chinese uses local terms like 繁體中文 and 釋經 / 生活應用, whereas Simplified Chinese uses 简体中文 and 实际应用.

Easy Scalability: Adding a new language (e.g., Spanish or Korean) requires updating only the dictionary entries in config.py and i18n.py.

---
