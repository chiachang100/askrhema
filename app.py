"""Main Streamlit UI application for SeekRhema."""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
import streamlit as st
import pandas as pd
from datetime import datetime
import logging

from config import (
    DEFAULT_SEARCH_CONFIG, 
    DEFAULT_LLM_CONFIG, 
    get_system_prompt,
    DEFAULT_LANGUAGE_CONFIG
)
from engine import (
    load_bible_data,
    get_verse_reference,
    HybridSearchEngine,
    SearchResult,
    stream_llm_response,
    get_available_models,
    BibleDataError,
    LLMProviderError
)
from i18n import get_translations
import importlib.metadata

@st.cache_data
def get_app_version() -> str:
    """Retrieve package version dynamically from pyproject.toml / installed package."""
    try:
        # Uses your package name defined in pyproject.toml
        return f"v{importlib.metadata.version('seekrhema')}"
    except importlib.metadata.PackageNotFoundError:
        # Fallback if package is running unpackaged during initial dev
        return "v0.1.0"

# Get dynamic version
app_version = get_app_version()

# Setup logging
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SeekRhema",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize translations
translations = get_translations()

# Custom CSS (keep your existing CSS, but I'll add language-specific adjustments)
st.markdown("""
    <style>
/* -------------------------------------------------------------
       MAIN HEADER & SUB-HEADER (Theme-Adaptive)
    ------------------------------------------------------------- */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        
        /* Light Mode Gradient (Dark Navy) */
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .sub-header {
        font-size: 1.15rem;
        font-weight: 500;
        color: var(--text-color, #4a5568); /* Uses Streamlit theme color */
        opacity: 0.9;
        margin-bottom: 2rem;
    }

    /* -------------------------------------------------------------
       DARK MODE OVERRIDES (Triggers when Streamlit/Browser is Dark)
    ------------------------------------------------------------- */
    @media (prefers-color-scheme: dark) {
        .main-header {
            /* Bright Gold/Sky-Blue Gradient for Dark Mode Visibility */
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
        }

        .sub-header {
            color: #cbd5e1 !important; /* Soft light gray for high dark-mode contrast */
            opacity: 1 !important;
        }

        .sidebar-title, .ai-response h4 {
            color: #f6d365 !important;
        }

        .verse-card {
            background: rgba(255, 255, 255, 0.05) !important;
            border-left: 5px solid #f6d365 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
        }

        .verse-reference {
            color: #90caf9 !important;
        }

        .verse-text {
            color: #e2e8f0 !important;
        }

        .search-stats, .example-chip {
            background: rgba(255, 255, 255, 0.08) !important;
            color: #e2e8f0 !important;
            border-color: rgba(255, 255, 255, 0.15) !important;
        }

        .footer {
            color: #94a3b8 !important; /* Slate light gray for dark mode */
            border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
    }

    /* -------------------------------------------------------------
       CARDS & UI COMPONENTS
    ------------------------------------------------------------- */
    .verse-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1e3a5f;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .verse-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .verse-reference {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e3a5f;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .verse-text {
        font-size: 1.05rem;
        line-height: 1.7;
        margin-top: 0.5rem;
        color: #2c3e50;
    }
    .verse-meta {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }
    .score-badge {
        display: inline-block;
        background-color: rgba(108, 117, 125, 0.15);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: var(--text-color, #495057);
        font-weight: 500;
    }
    .score-badge.high {
        background-color: #d4edda;
        color: #155724;
    }
    .score-badge.medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .category-tag {
        display: inline-block;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #155724;
        font-weight: 500;
    }
    .testament-tag {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .testament-tag.ot {
        background-color: #cce5ff;
        color: #004085;
    }
    .testament-tag.nt {
        background-color: #ffe5cc;
        color: #854d00;
    }
    .ai-response {
        background: rgba(0, 123, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .ai-response h4 {
        color: #1e3a5f;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sidebar-logo {
        text-align: center;
        padding: 1rem 0;
        font-size: 3rem;
    }
    .sidebar-title {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 1rem;
    }
    .search-stats {
        background: #f8f9fa;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
    }
    .stat-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-color, #495057);
        font-size: 0.9rem;
    }

    /* Standard Footer Styling */
    .footer {
        margin-top: 3rem;
        padding: 1rem;
        text-align: center;
        color: var(--text-color, #6c757d);
        font-size: 0.85rem;
        border-top: 1px solid rgba(108, 117, 125, 0.2);
    }

    .example-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .example-chip {
        background: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        cursor: pointer;
        text-align: center;
        font-size: 0.9rem;
        border: 1px solid #e9ecef;
        transition: all 0.2s ease;
    }
    .example-chip:hover {
        background: #e9ecef;
        border-color: #1e3a5f;
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "search_engine": None,
        "search_results": [],
        "query": "",
        "ai_response": "",
        "ollama_api_key": "",
        "google_api_key": "",
        "openai_api_key": "",
        "selected_provider": "ollama",
        "selected_model": "llama2",
        "ai_mode": False,
        "fast_mode": False,
        "initialized": False,
        "last_search_time": None,
        "search_count": 0,
        "error_message": None,
        "search_triggered": False,
        "language": DEFAULT_LANGUAGE_CONFIG.default_language
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


@st.cache_data(ttl=3600)
def load_bible_data_cached(file_path: str) -> List[Dict[str, Any]]:
    """Load Bible data with caching."""
    return load_bible_data(file_path)


@st.cache_resource(ttl=3600)
def get_search_engine() -> HybridSearchEngine:
    """Get or create the search engine with caching."""
    return HybridSearchEngine()


def display_sidebar() -> tuple[str, str, str, bool, int, Optional[str], Optional[str], str]:
    """Display the sidebar and return configuration values."""
    t = translations
    
    st.sidebar.markdown('<div class="sidebar-logo">📖</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="sidebar-title">{t.get("ui.sidebar.title")}</div>', unsafe_allow_html=True)
    st.sidebar.divider()
    
    # Language selection
    st.sidebar.subheader("🌐 Language")
    lang_options = list(t.get_available_languages().keys())
    lang_display = list(t.get_available_languages().values())
    
    current_lang = st.session_state.get("language", "en")
    current_index = lang_options.index(current_lang) if current_lang in lang_options else 0
    
    language = st.sidebar.selectbox(
        label="Select Language",
        options=lang_options,
        format_func=lambda x: t.get_available_languages().get(x, x),
        key="language_select"
    )
    
    if language != st.session_state.get("language"):
        st.session_state.language = language
        # Reload translations for the new language
        translations.set_language(language)
    
    st.sidebar.divider()
    
    # Provider selection
    st.sidebar.subheader(t.get("ui.sidebar.provider"))
    provider = st.sidebar.selectbox(
        label=t.get("ui.sidebar.provider"),
        options=["ollama", "google", "openai"],
        format_func=lambda x: t.get(f"llm.providers.{x}", x),
        help=t.get("ui.sidebar.provider_help"),
        key="provider_select"
    )
    st.session_state.selected_provider = provider
    
    # Model selection
    st.sidebar.subheader(t.get("ui.sidebar.model"))
    models = get_available_models(provider)
    default_model = models[0] if models else ""
    model = st.sidebar.selectbox(
        label=t.get("ui.sidebar.model"),
        options=models,
        help=t.get("ui.sidebar.model_help"),
        key="model_select"
    )
    st.session_state.selected_model = model
    
    # API key inputs
    st.sidebar.subheader(t.get("ui.sidebar.api_keys"))
    api_key = None
    
    if provider == "ollama":
        st.sidebar.info(t.get("ui.sidebar.ollama_help"))
        if st.sidebar.button(t.get("ui.sidebar.ollama_status"), use_container_width=True):
            try:
                import httpx
                response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
                if response.status_code == 200:
                    st.sidebar.success("✅ Ollama is running!")
                    models_list = response.json().get("models", [])
                    if models_list:
                        model_names = [m.get('name', 'unknown') for m in models_list]
                        st.sidebar.info(f"Available: {', '.join(model_names)}")
                else:
                    st.sidebar.error("❌ Ollama returned an error")
            except Exception as e:
                st.sidebar.error(f"❌ Cannot connect: {str(e)}")
    elif provider == "google":
        api_key = st.sidebar.text_input(
            label=t.get("ui.sidebar.google_key"),
            value=st.session_state.google_api_key,
            type="password",
            help=t.get("ui.sidebar.google_help"),
            placeholder="AIza...",
            key="google_api_input"
        )
        st.session_state.google_api_key = api_key
        if not api_key:
            st.sidebar.warning("⚠️ Please enter your Google API key")
    elif provider == "openai":
        api_key = st.sidebar.text_input(
            label=t.get("ui.sidebar.openai_key"),
            value=st.session_state.openai_api_key,
            type="password",
            help=t.get("ui.sidebar.openai_help"),
            placeholder="sk-...",
            key="openai_api_input"
        )
        st.session_state.openai_api_key = api_key
        if not api_key:
            st.sidebar.warning("⚠️ Please enter your OpenAI API key")
    
    st.sidebar.divider()
    
    # Search settings
    st.sidebar.subheader("🔍 Search Settings")
    
    top_k = st.sidebar.slider(
        label=t.get("ui.search.depth"),
        min_value=1,
        max_value=10,
        value=DEFAULT_SEARCH_CONFIG.top_k,
        help=t.get("ui.search.depth_help"),
        key="top_k_slider"
    )
    
    fast_mode = st.sidebar.toggle(
        label=t.get("ui.search.fast_mode"),
        value=st.session_state.fast_mode,
        help=t.get("ui.search.fast_mode_help"),
        key="fast_mode_toggle"
    )
    st.session_state.fast_mode = fast_mode
    
    # Filters
    st.sidebar.subheader(t.get("ui.sidebar.filters"))
    
    testament_options = ["All", "OT", "NT"]
    testament_filter = st.sidebar.selectbox(
        label=t.get("ui.sidebar.testament"),
        options=testament_options,
        format_func=lambda x: t.get(f"bible.testaments.{x.lower()}", x) if x != "All" else t.get("bible.testaments.all"),
        help=t.get("ui.sidebar.testament_help"),
        key="testament_filter_select"
    )
        
    books = ["All"] + ["Genesis", "Psalms", "Isaiah", "Matthew", "John", "Romans"]
    book_display = [t.get("bible.books.all")] + [
        t.get(f"bible.books.{book.lower()}", book) for book in books[1:]
    ]
    
    book_filter = st.sidebar.selectbox(
        label=t.get("ui.sidebar.book"),
        options=books,
        format_func=lambda x: t.get(f"bible.books.{x.lower()}", x) if x != "All" else t.get("bible.books.all"),
        help=t.get("ui.sidebar.book_help"),
        key="book_filter_select"
    )

    st.sidebar.divider()
    
    # AI Mode toggle
    st.sidebar.subheader("🧠 AI Features")
    ai_mode = st.sidebar.toggle(
        label=t.get("ui.sidebar.ai_mode"),
        value=st.session_state.ai_mode,
        help=t.get("ui.sidebar.ai_mode_help"),
        key="ai_mode_toggle"
    )
    st.session_state.ai_mode = ai_mode
    
    # Display stats
    if st.session_state.search_count > 0:
        st.sidebar.divider()
        st.sidebar.subheader(t.get("ui.sidebar.statistics"))
        st.sidebar.metric(t.get("ui.sidebar.searches"), st.session_state.search_count)
        if st.session_state.last_search_time:
            st.sidebar.caption(f"{t.get('ui.sidebar.last_search')}: {st.session_state.last_search_time}")
    
    return provider, model, api_key, ai_mode, top_k, book_filter, testament_filter, language


def display_verse_card(result: SearchResult, index: int) -> None:
    """Display a single verse card with enhanced styling."""
    t = translations
    
    # Determine score badge class
    score_class = "medium"
    if result.fused_score > 0.5:
        score_class = "high"
    
    # Determine testament class
    testament_class = "ot" if result.testament == "OT" else "nt"
    
    with st.container():
        st.markdown(f"""
        <div class="verse-card">
            <div class="verse-reference">
                <span>{result.get_reference()}</span>
                <span class="category-tag">{result.category}</span>
                <span class="testament-tag {testament_class}">{result.testament}</span>
            </div>
            <div class="verse-text">"{result.text}"</div>
            <div class="verse-meta">
                <span class="score-badge {score_class}">🎯 RRF: {result.fused_score:.3f}</span>
                <span class="score-badge">📊 Dense: {result.dense_score:.3f}</span>
                <span class="score-badge">📝 Sparse: {result.sparse_score:.3f}</span>
                <span class="score-badge">#️⃣ Rank: {index + 1}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def generate_ai_response(
    provider: str,
    model: str,
    query: str,
    api_key: Optional[str],
    results: List[SearchResult],
    system_prompt: str,
    language: str = "en"
) -> None:
    """Generate and stream AI response based on search results."""
    t = translations
    
    if not results:
        st.warning(t.get("ui.results.no_results"))
        return
    
    # Prepare context from search results
    context_verses = [
        {
            "book": r.book,
            "chapter": r.chapter,
            "verse": r.verse,
            "text": r.text
        }
        for r in results[:5]
    ]
    
    # Generate prompt based on language
    if language.startswith("zh"):
        prompt = f"""请根据以下查询 "{query}" 找到的圣经经文提供全面的解经分析：

上下文经文：
{chr(10).join([f"- {v['book']} {v['chapter']}:{v['verse']}: {v['text']}" for v in context_verses])}

请按以下结构组织您的回答：

1. **摘要**：简要概述这些经文及其与查询的关联
2. **神学主题**：这些经文中呈现的关键神学概念和教义
3. **历史背景**：重要的历史或文化背景
4. **实际应用**：这些经文如何应用于现代基督徒生活
5. **关联经文**：这些经文与其他相关圣经章节的互联关系

请用清晰的部分组织您的回答，提供适当的引用，保持学术性但易于理解的语气。保持全面但简洁。所有见解都基于所提供的经文。"""
    else:
        prompt = f"""Please provide a thorough exegetical analysis of the following Bible passages found for the query: "{query}"

Context Passages:
{chr(10).join([f"- {v['book']} {v['chapter']}:{v['verse']}: {v['text']}" for v in context_verses])}

Please structure your response as follows:

1. **Summary**: Brief overview of the passages and their connection to the query
2. **Theological Themes**: Key theological concepts and doctrines present in these passages
3. **Historical Context**: Important historical or cultural background
4. **Practical Application**: How these passages apply to modern Christian living
5. **Connections**: Interconnections between these passages and other related scriptures

Format your response with clear sections, proper citations, and maintain a scholarly yet accessible tone.
Be thorough but concise. Ground all insights in the provided scripture passages."""

    # Stream the response
    response_container = st.empty()
    full_response = ""
    
    try:
        with st.spinner(t.get("ui.ai.generating")):
            response_container.markdown(f"""
            <div class="ai-response">
                <h4>{t.get("ui.ai.exegesis")} <span style="display:inline-block;animation:spin 1s linear infinite;">⏳</span></h4>
                <p><em>{t.get("ui.ai.generating")}</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Use language-specific system prompt
            lang_system_prompt = get_system_prompt(language)
            
            for chunk in stream_llm_response(
                provider=provider,
                model_name=model,
                prompt=prompt,
                system_prompt=lang_system_prompt,
                api_key=api_key,
                context_verses=context_verses,
                temperature=0.7,
                max_tokens=1500
            ):
                full_response += chunk
                response_container.markdown(f"""
                <div class="ai-response">
                    <h4>{t.get("ui.ai.exegesis")}</h4>
                    <div style="margin-top: 1rem; line-height: 1.8;">
                        {full_response}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    except LLMProviderError as e:
        st.error(f"{t.get('ui.ai.provider_error')}: {str(e)}")
        response_container.empty()
    except Exception as e:
        st.error(f"{t.get('ui.ai.error')}: {str(e)}")
        response_container.empty()


def handle_search(
    search_engine: HybridSearchEngine,
    query: str,
    top_k: int,
    book_filter: str,
    testament_filter: str,
    fast_mode: bool
) -> List[SearchResult]:
    """Execute the search with error handling."""
    try:
        # Prepare filters
        book = book_filter if book_filter != "All" else None
        testament = testament_filter if testament_filter != "All" else None
        
        # Perform search based on mode
        if fast_mode:
            results = search_engine.search_fast(
                query=query,
                top_k=top_k,
                book=book,
                testament=testament
            )
        else:
            results = search_engine.search(
                query=query,
                top_k=top_k,
                book=book,
                testament=testament
            )
        
        # Update session stats
        st.session_state.search_count += 1
        st.session_state.last_search_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.error_message = None
        
        return results
        
    except Exception as e:
        st.session_state.error_message = str(e)
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return []


def main() -> None:
    """Main application entry point."""
    t = translations
    initialize_session_state()
    
    # Set language from session state
    current_lang = st.session_state.get("language", "en")
    translations.set_language(current_lang)
    
    # Load Bible data
    data_path = Path(__file__).parent / "data" / "sample_bible.json"
    try:
        verses = load_bible_data_cached(str(data_path))
    except FileNotFoundError:
        st.error(f"❌ {t.get('ui.errors.bible_data')}: {data_path}")
        st.info(t.get("ui.errors.bible_data_help"))
        st.stop()
    except BibleDataError as e:
        st.error(f"❌ Bible data validation error: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error loading Bible data: {str(e)}")
        st.stop()
    
    # Initialize search engine
    search_engine = get_search_engine()
    if not st.session_state.initialized:
        with st.spinner(t.get("ui.status.initializing")):
            try:
                search_engine.initialize(verses)
                st.session_state.search_engine = search_engine
                st.session_state.initialized = True
                st.success(f"✅ {t.get('ui.status.initialized')}")
            except Exception as e:
                st.error(f"❌ {t.get('ui.errors.init_failed')}: {str(e)}")
                st.stop()
    
    # Sidebar configuration
    provider, model, api_key, ai_mode, top_k, book_filter, testament_filter, language = display_sidebar()
    
    # Main content
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f'<div class="main-header">{t.get("app.title")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">{t.get("app.subtitle")}</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.search_count > 0:
            st.metric(t.get("ui.sidebar.searches"), st.session_state.search_count)
    
    st.divider()
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            label="Search",
            placeholder=t.get("ui.search.placeholder"),
            value=st.session_state.query,
            label_visibility="collapsed",
            key="search_input"
        )
        st.session_state.query = query
    
    with col2:
        search_button = st.button(
            label=t.get("ui.search.button"),
            type="primary",
            use_container_width=True,
            key="search_button"
        )
    
    # Handle search
    if search_button and query:
        st.session_state.search_triggered = True
    
    if st.session_state.search_triggered and query:
        st.session_state.search_triggered = False
        with st.spinner(t.get("ui.status.searching")):
            results = handle_search(
                search_engine,
                query,
                top_k,
                book_filter,
                testament_filter,
                st.session_state.fast_mode
            )
            st.session_state.search_results = results
    
    # Display search stats
    if st.session_state.search_results:
        results = st.session_state.search_results
        st.markdown(f"""
        <div class="search-stats">
            <div class="stat-item">
                <span>📊 {t.get('ui.results.found').format(count=len(results))}</span>
            </div>
            <div class="stat-item">
                <span>🔍 Query</span>
                <span class="stat-value">"{query}"</span>
            </div>
            <div class="stat-item">
                <span>⚡ Mode</span>
                <span class="stat-value">{'Fast' if st.session_state.fast_mode else 'Hybrid'}</span>
            </div>
            {f'<div class="stat-item"><span>📖 Filter</span><span class="stat-value">{book_filter}</span></div>' if book_filter != "All" else ''}
            {f'<div class="stat-item"><span>📜 Testament</span><span class="stat-value">{testament_filter}</span></div>' if testament_filter != "All" else ''}
        </div>
        """, unsafe_allow_html=True)
    
    # Display results
    results = st.session_state.search_results
    
    if results:
        # Display results
        for idx, result in enumerate(results):
            display_verse_card(result, idx)
            
        # AI Response
        if ai_mode and results:
            st.divider()
            with st.expander(t.get("ui.ai.exegesis"), expanded=True):
                generate_ai_response(
                    provider=provider,
                    model=model,
                    query=query,
                    api_key=api_key if provider in ["google", "openai"] else None,
                    results=results,
                    system_prompt=get_system_prompt(language),
                    language=language
                )
    
    elif st.session_state.get("error_message"):
        st.error(f"❌ {st.session_state.error_message}")
        st.session_state.error_message = None
    
    elif search_button and query:
        st.info(t.get("ui.results.no_results"))
    
    elif not query:
        # Show helpful examples
        st.markdown(f"""
        ### {t.get('ui.results.search_examples')}
        <div class="example-grid">
            <div class="example-chip">"God created"</div>
            <div class="example-chip">"faith salvation"</div>
            <div class="example-chip">"love thy neighbor"</div>
            <div class="example-chip">"for God so loved"</div>
            <div class="example-chip">"blessed are the poor"</div>
            <div class="example-chip">"shepherd"</div>
        </div>
        
        ### {t.get('ui.results.tips')}
        - {t.get('ui.results.tips_quotes')}
        - {t.get('ui.results.tips_ai')}
        - {t.get('ui.results.tips_fast')}
        - {t.get('ui.results.tips_filters')}
        """, unsafe_allow_html=True)
    
    # Footer
    # &nbsp;•&nbsp; {t.get('app.powered_by')}
    st.divider()
    st.markdown(f"""
    <div class="footer">
        <strong>{t.get('app.title')}</strong> {app_version}
        &nbsp;•&nbsp; {datetime.now().strftime('%Y')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()