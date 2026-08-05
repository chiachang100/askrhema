"""AskRhema – Conversational Bible search and AI exegesis assistant with i18n."""

import logging
import uuid
from pathlib import Path
import tomllib

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

logger = logging.getLogger(__name__)

@st.cache_data
def get_app_version() -> str:
    """Return the application version from pyproject.toml."""
    pyproject_path = Path(__file__).parent / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    return str(data["project"]["version"])


APP_VERSION = get_app_version()

def get_secret(key: str, default: str = "") -> str:
    """
    Safely retrieve a Streamlit secret.

    Returns the default value when no Streamlit secrets file exists.
    """
    try:
        value = st.secrets.get(key, default)
        return str(value) if value is not None else default
    except Exception:
        return default


def get_api_key_for_provider(provider: str) -> str:
    """Return the configured API key for the selected provider."""
    key_names = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    secret_name = key_names.get(provider)
    if not secret_name:
        return ""

    return get_secret(secret_name)

def reset_conversation() -> None:
    """Reset the current conversation history."""
    st.session_state.messages = []


st.set_page_config(
    page_title="AskRhema",
    page_icon="📖",
    layout="wide",
)

# --------------------------------------------------------------------------- 
# # Chat styling 
# # --------------------------------------------------------------------------- 
st.markdown( 
    """ 
    <style> 
    /* User message: avatar and content move to the right. */
    [data-testid="stChatMessage"]:has( 
        [data-testid="stChatMessageAvatarUser"] 
    ) { 
        flex-direction: row-reverse; 
    } 

    /* User message content area. */
    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) [data-testid="stChatMessageContent"] {
        align-items: flex-end;
        text-align: right;
    }
    
    /* User bubble: target the actual markdown/content wrapper. */
    [data-testid="stChatMessage"]:has( 
        [data-testid="stChatMessageAvatarUser"] 
    ) [data-testid="stChatMessageContent"] > div { 
        background: rgba(100, 116, 139, 0.12); 
        border-radius: 18px 18px 4px 18px; 
        padding: 0.7rem 1rem; 
        width: fit-content;
        max-width: 75%; 
        margin-left: auto;
    } 

    /* Make paragraphs inside the user bubble right-aligned. */
    [data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) [data-testid="stChatMessageContent"] p {
        text-align: right;
    }
    
    /* Assistant messages remain left-aligned. */
    [data-testid="stChatMessage"]:has( 
        [data-testid="stChatMessageAvatarAssistant"] 
    ) [data-testid="stChatMessageContent"] {
        text-align: left;
    }
    
    /* Remove unnecessary bottom margin from final paragraph. */
    [data-testid="stChatMessageContent"] p:last-child { 
        margin-bottom: 0; 
    } 
    </style> 
    """, 
    unsafe_allow_html=True, 
)

t = get_translations()

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "provider" not in st.session_state:
    configured_provider = get_secret(
        "DEFAULT_LLM_PROVIDER",
        DEFAULT_LLM_CONFIG.default_provider,
    )
    if configured_provider not in ("ollama", "gemini", "openai"):
        configured_provider = DEFAULT_LLM_CONFIG.default_provider
    st.session_state.provider = configured_provider

if "model" not in st.session_state:
    models = get_available_models(st.session_state.provider)
    st.session_state.model = models[0] if models else DEFAULT_LLM_CONFIG.ollama_model

if "api_key" not in st.session_state:
    st.session_state.api_key = get_api_key_for_provider(
        st.session_state.provider
    )

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


# ---------------------------------------------------------------------------
# Initialize search engine and chat service
# ---------------------------------------------------------------------------

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

t.set_language(st.session_state.language)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙ " + t.get("ui.sidebar.title"))

    # Language
    with st.expander("Language", expanded=False):
        lang_options = get_available_languages()
        lang_codes = list(lang_options.keys())
        lang_display = [
            f"{code} - {name}" for code, name in lang_options.items()
        ]

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

    # AI provider
    with st.expander(
        t.get("ui.sidebar.provider"),
        expanded=False,
    ):
        providers = ["ollama", "gemini", "openai"]
        current_provider_index = (
            providers.index(st.session_state.provider)
            if st.session_state.provider in providers
            else 0
        )

        provider = st.selectbox(
            t.get("ui.sidebar.provider"),
            options=providers,
            index=current_provider_index,
            key="provider_selector",
            format_func=lambda value: t.get(
                f"llm.providers.{value}",
                value.capitalize(),
            ),
            help=t.get("ui.sidebar.provider_help"),
        )

        if provider != st.session_state.provider:
            st.session_state.provider = provider

            models = get_available_models(provider)
            st.session_state.model = (
                models[0]
                if models
                else (
                    DEFAULT_LLM_CONFIG.ollama_model 
                    if provider == "ollama" 
                    else ""
                )
            )

            st.session_state.api_key = get_api_key_for_provider(provider)
            st.rerun()

        # Model
        models = get_available_models(st.session_state.provider)

        if not models:
            st.warning("No models available for this provider.")
        else:
            current_model_index = (
                models.index(st.session_state.model)
                if st.session_state.model in models
                else 0
            )

            model = st.selectbox(
                t.get("ui.sidebar.model"),
                options=models,
                index=current_model_index,
                key="model_selector",
                help=t.get("ui.sidebar.model_help"),
            )

            # Do not mutate model_selector after widget creation.
            st.session_state.model = model

        # API key
        if st.session_state.provider in ("gemini", "openai"):
            if st.session_state.provider == "gemini":
                api_label = t.get("ui.sidebar.google_key")
                api_help = t.get("ui.sidebar.google_help")
            else:
                api_label = t.get("ui.sidebar.openai_key")
                api_help = t.get("ui.sidebar.openai_help")

            api_key = st.text_input(
                api_label,
                type="password",
                value=st.session_state.api_key,
                key="api_key_input",
                help=api_help,
            )

            # Different key from the widget, so this is safe.
            st.session_state.api_key = api_key

        # Temperature
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.temperature),
            step=0.05,
            key="temp_slider",
        )

        # Different widget key, so this is safe.
        st.session_state.temperature = temperature

        # Maximum tokens
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=100,
            max_value=4096,
            value=int(st.session_state.max_tokens),
            step=100,
            key="max_tokens_input",
        )

        # Different widget key, so this is safe.
        st.session_state.max_tokens = int(max_tokens)

    # Search filters
    with st.expander(
        t.get("ui.sidebar.filters"),
        expanded=False,
    ):
        book_filter = st.text_input(
            t.get("ui.sidebar.book"),
            value=st.session_state.book_filter,
            key="book_filter_input",
            help=t.get("ui.sidebar.book_help"),
        )

        # IMPORTANT:
        # Do not use key="book_filter" here and then assign
        # st.session_state.book_filter afterward. Streamlit forbids
        # modifying a widget's own session-state key after creation.
        st.session_state.book_filter = book_filter

        testament_options = ["", "OT", "NT"]
        current_testament_index = (
            testament_options.index(st.session_state.testament_filter)
            if st.session_state.testament_filter in testament_options
            else 0
        )

        testament_filter = st.selectbox(
            t.get("ui.sidebar.testament"),
            options=testament_options,
            index=current_testament_index,
            key="testament_filter_input",
            format_func=lambda value: (
                t.get("bible.testaments.all")
                if value == ""
                else t.get(
                    f"bible.testaments.{value.lower()}",
                    value,
                )
            ),
            help=t.get("ui.sidebar.testament_help"),
        )

        st.session_state.testament_filter = testament_filter

    # Debug
    with st.expander(
        t.get("ui.debug.title", "🔧 Debug Info"),
        expanded=False,
    ):
        show_retrieval_details = st.checkbox(
            "Show retrieval details",
            value=st.session_state.show_retrieval_details,
            key="show_retrieval_details_input",
        )

        st.session_state.show_retrieval_details = show_retrieval_details

        if st.button(
            t.get(
                "ui.sidebar.reset_conversation",
                "Reset Conversation",
            )
        ):
            reset_conversation()
            st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title(t.get("app.title"))
st.caption(f"{t.get("app.subtitle")} - v{APP_VERSION}")

# ---------------------------------------------------------------------------
# Display previous chat messages
# ---------------------------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"]

            if sources and st.session_state.show_retrieval_details:
                with st.expander(t.get("ui.debug.title", "📚 Sources")):
                    for result in sources:
                        st.write(f"{result.reference} (RRF: {result.rrf_score:.4f})")

            elif sources:
                refs = ", ".join(result.reference for result in sources[:5])

                if refs:
                    st.caption(f"📖 Sources: {refs}")


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input(
    t.get(
        "ui.search.placeholder",
        "Ask AskRhema about Scripture...",
    )
):
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        sources = []

        if st.session_state.provider == "ollama":
            api_key = None
        else:
            api_key = st.session_state.api_key.strip()

        chat_service = st.session_state.chat_service
        system_prompt = get_system_prompt(st.session_state.language)

        try:
            stream = chat_service.process_message(
                user_message=prompt,
                conversation_history=st.session_state.messages[:-1],
                provider=st.session_state.provider,
                model_name=st.session_state.model,
                api_key=api_key,
                book_filter=(st.session_state.book_filter or None),
                testament_filter=(st.session_state.testament_filter or None),
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                system_prompt_override=system_prompt,
            )

            for chunk, current_sources in stream:
                full_response += chunk
                sources = current_sources

                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception:
            error_id = uuid.uuid4().hex[:8]

            logger.exception(
                "AI response generation failed [error_id=%s]",
                error_id,
            )

            user_error = t.get(
                "ui.ai.error",
                "Sorry, we couldn't generate a response.",
            )

            st.error(f"{user_error} (Error ID: {error_id})")

            full_response = user_error
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
            refs = ", ".join(result.reference for result in sources[:5])

            if refs:
                st.caption(f"📖 Sources: {refs}")

        st.rerun()


# ---------------------------------------------------------------------------
# Welcome message
# ---------------------------------------------------------------------------

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
