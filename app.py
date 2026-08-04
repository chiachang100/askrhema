"""AskRhema – Chat-first Bible search and exegesis assistant with i18n."""

import streamlit as st

from config import (
    DEFAULT_EMBEDDING_CONFIG,
    DEFAULT_LLM_CONFIG,
    DEFAULT_SEARCH_CONFIG,
    get_system_prompt,
)
from engine.chat import ChatService
from engine.hybrid_search import get_search_engine
from engine.llm_provider import get_available_models
from i18n.translations import get_available_languages, get_translations

# Page config
st.set_page_config(page_title="AskRhema", page_icon="📖", layout="wide")

# Initialize i18n
t = get_translations()

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "provider" not in st.session_state:
    st.session_state.provider = DEFAULT_LLM_CONFIG.default_provider
if "model" not in st.session_state:
    models = get_available_models(st.session_state.provider)
    st.session_state.model = models[0] if models else DEFAULT_LLM_CONFIG.ollama_model
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "temperature" not in st.session_state:
    st.session_state.temperature = DEFAULT_LLM_CONFIG.temperature
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = DEFAULT_LLM_CONFIG.max_tokens
if "book_filter" not in st.session_state:
    st.session_state.book_filter = ""
if "testament_filter" not in st.session_state:
    st.session_state.testament_filter = ""
if "show_retrieval_details" not in st.session_state:
    st.session_state.show_retrieval_details = False
if "language" not in st.session_state:
    st.session_state.language = "en"

# Load search engine and chat service
if "search_engine" not in st.session_state:
    st.session_state.search_engine = get_search_engine(
        "data/sample_bible.json",
        DEFAULT_SEARCH_CONFIG,
        DEFAULT_EMBEDDING_CONFIG,
    )
if "chat_service" not in st.session_state:
    st.session_state.chat_service = ChatService(
        st.session_state.search_engine,
        DEFAULT_SEARCH_CONFIG,
        DEFAULT_LLM_CONFIG,
    )

# Ensure language is synced with the Translations singleton
t.set_language(st.session_state.language)


def reset_conversation() -> None:
    """Reset the current conversation history."""
    st.session_state.messages = []


# --- Sidebar ---
with st.sidebar:
    st.header("⚙ " + t.get("ui.sidebar.title"))

    # Language selector
    with st.expander("Language", expanded=False):
        lang_options = get_available_languages()
        lang_codes = list(lang_options.keys())
        lang_display = [f"{code} - {name}" for code, name in lang_options.items()]
        current_index = (
            lang_codes.index(st.session_state.language)
            if st.session_state.language in lang_codes
            else 0
        )
        selected_display = st.selectbox(
            "Select language",
            options=lang_display,
            index=current_index,
            key="lang_selector",
        )
        selected_code = lang_codes[lang_display.index(selected_display)]
        if selected_code != st.session_state.language:
            st.session_state.language = selected_code
            t.set_language(selected_code)
            st.rerun()

    with st.expander(t.get("ui.sidebar.provider"), expanded=False):
        provider = st.selectbox(
            t.get("ui.sidebar.provider"),
            options=["ollama", "gemini", "openai"],
            index=["ollama", "gemini", "openai"].index(st.session_state.provider),
            key="provider_selector",
            help=t.get("ui.sidebar.provider_help"),
        )
        if provider != st.session_state.provider:
            st.session_state.provider = provider
            models = get_available_models(provider)
            if models:
                st.session_state.model = models[0]
            st.rerun()

        models = get_available_models(st.session_state.provider)
        model = st.selectbox(
            t.get("ui.sidebar.model"),
            options=models,
            index=models.index(st.session_state.model)
            if st.session_state.model in models
            else 0,
            key="model_selector",
            help=t.get("ui.sidebar.model_help"),
        )
        if model != st.session_state.model:
            st.session_state.model = model

        if st.session_state.provider in ("gemini", "openai"):
            api_key = st.text_input(
                t.get("ui.sidebar.google_key")
                if st.session_state.provider == "gemini"
                else t.get("ui.sidebar.openai_key"),
                type="password",
                value=st.session_state.api_key,
                key="api_key_input",
                help=t.get("ui.sidebar.google_help")
                if st.session_state.provider == "gemini"
                else t.get("ui.sidebar.openai_help"),
            )
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key

        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            st.session_state.temperature,
            0.05,
            key="temp_slider",
        )
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature

        max_tokens = st.number_input(
            "Max Tokens",
            100,
            4096,
            st.session_state.max_tokens,
            100,
            key="max_tokens_input",
        )
        if max_tokens != st.session_state.max_tokens:
            st.session_state.max_tokens = max_tokens

    with st.expander(t.get("ui.sidebar.filters"), expanded=False):
        book_filter = st.text_input(
            t.get("ui.sidebar.book"),
            value=st.session_state.book_filter,
            key="book_filter",
            help=t.get("ui.sidebar.book_help"),
        )
        if book_filter != st.session_state.book_filter:
            st.session_state.book_filter = book_filter

        testament_filter = st.selectbox(
            t.get("ui.sidebar.testament"),
            options=["", "OT", "NT"],
            index=["", "OT", "NT"].index(st.session_state.testament_filter),
            key="testament_filter",
            help=t.get("ui.sidebar.testament_help"),
        )
        if testament_filter != st.session_state.testament_filter:
            st.session_state.testament_filter = testament_filter

    with st.expander(t.get("ui.debug.title"), expanded=False):
        st.session_state.show_retrieval_details = st.checkbox(
            "Show retrieval details",
            value=st.session_state.show_retrieval_details,
        )
        if st.button(t.get("ui.sidebar.reset_conversation", "Reset Conversation")):
            reset_conversation()
            st.rerun()


# --- Main Chat Area ---
st.title(t.get("app.title"))
st.caption(t.get("app.subtitle"))

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]
            if sources and st.session_state.show_retrieval_details:
                with st.expander(t.get("ui.debug.title", "📚 Sources")):
                    for r in sources:
                        st.write(f"{r.reference} (RRF: {r.rrf_score:.4f})")
            elif sources:
                refs = ", ".join([r.reference for r in sources[:5]])
                if refs:
                    st.caption(f"📖 Sources: {refs}")

# Chat input
if prompt := st.chat_input(
    t.get("ui.search.placeholder", "Ask AskRhema about Scripture...")
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        sources = []

        api_key = (
            st.session_state.api_key if st.session_state.provider != "ollama" else None
        )
        chat_service = st.session_state.chat_service

        # Use the language-specific system prompt from config
        system_prompt = get_system_prompt(st.session_state.language)

        try:
            stream = chat_service.process_message(
                user_message=prompt,
                conversation_history=st.session_state.messages[:-1],
                provider=st.session_state.provider,
                model_name=st.session_state.model,
                api_key=api_key,
                book_filter=st.session_state.book_filter or None,
                testament_filter=st.session_state.testament_filter or None,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                system_prompt_override=system_prompt,  # ✅ Pass override
            )
            for chunk, current_sources in stream:
                full_response += chunk
                sources = current_sources
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"{t.get('ui.ai.error')}: {str(e)}")
            full_response = f"Sorry, I encountered an error: {str(e)}"
            response_placeholder.markdown(full_response)
            sources = []

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            }
        )
        if sources and not st.session_state.show_retrieval_details:
            refs = ", ".join([r.reference for r in sources[:5]])
            if refs:
                st.caption(f"📖 Sources: {refs}")
        st.rerun()

# Welcome message when empty
if not st.session_state.messages:
    st.info(
        f"""
        ### {t.get("app.title")}
        {t.get("app.subtitle")}

        {t.get("ui.results.search_examples")}:
        - What does Romans 8:28 mean?
        - Explain the context of John 15.
        - What does Scripture say about forgiveness?
        - How does the Old Testament point toward Christ?
        """
    )
