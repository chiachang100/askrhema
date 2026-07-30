# app.py (Minimal changes for Streamlit 1.60+)
"""Main Streamlit UI application for TBS (Tuixiu Bible Search)."""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator
import streamlit as st
import pandas as pd
from datetime import datetime

from config import DEFAULT_SEARCH_CONFIG, DEFAULT_LLM_CONFIG, get_system_prompt
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


# Page configuration
st.set_page_config(
    page_title="TBS - Tuixiu Bible Search",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS (keep your existing CSS here)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
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
        background-color: #e9ecef;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        color: #495057;
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
        background: linear-gradient(135deg, #e8f4f8 0%, #d4e8f0 100%);
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
        color: #495057;
        font-size: 0.9rem;
    }
    .stat-item .stat-value {
        font-weight: 600;
        color: #1e3a5f;
    }
    .footer {
        margin-top: 3rem;
        padding: 1rem;
        text-align: center;
        color: #6c757d;
        font-size: 0.85rem;
        border-top: 1px solid #e9ecef;
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
        "search_triggered": False
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


def display_sidebar() -> tuple[str, str, str, bool, int, Optional[str], Optional[str]]:
    """Display the sidebar and return configuration values."""
    st.sidebar.markdown('<div class="sidebar-logo">📖</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-title">TBS Config</div>', unsafe_allow_html=True)
    st.sidebar.divider()
    
    # Provider selection
    st.sidebar.subheader("🤖 AI Provider")
    provider = st.sidebar.selectbox(
        label="Select Provider",
        options=["ollama", "google", "openai"],
        format_func=lambda x: {
            "ollama": "🏠 Local Ollama",
            "google": "☁️ Google Gemini",
            "openai": "☁️ OpenAI"
        }.get(x, x),
        help="Select the LLM provider for AI exegesis",
        key="provider_select"
    )
    st.session_state.selected_provider = provider
    
    # Model selection
    st.sidebar.subheader("🧠 Model")
    models = get_available_models(provider)
    default_model = models[0] if models else ""
    model = st.sidebar.selectbox(
        label="Select Model",
        options=models,
        help="Select the model to use for AI responses",
        key="model_select"
    )
    st.session_state.selected_model = model
    
    # API key inputs
    st.sidebar.subheader("🔑 API Keys")
    api_key = None
    
    if provider == "ollama":
        st.sidebar.info("🔗 Using local Ollama at http://localhost:11434")
        if st.sidebar.button("🔄 Check Ollama Status", use_container_width=True):
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
            label="Google API Key",
            value=st.session_state.google_api_key,
            type="password",
            help="Enter your Google Gemini API key from Google AI Studio",
            placeholder="AIza...",
            key="google_api_input"
        )
        st.session_state.google_api_key = api_key
        if not api_key:
            st.sidebar.warning("⚠️ Please enter your Google API key")
    elif provider == "openai":
        api_key = st.sidebar.text_input(
            label="OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            help="Enter your OpenAI API key from platform.openai.com",
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
        label="📊 Results Depth",
        min_value=1,
        max_value=10,
        value=DEFAULT_SEARCH_CONFIG.top_k,
        help="Number of search results to return",
        key="top_k_slider"
    )
    
    fast_mode = st.sidebar.toggle(
        label="⚡ Fast Mode",
        value=st.session_state.fast_mode,
        help="Use only dense vector search (faster but less comprehensive)",
        key="fast_mode_toggle"
    )
    st.session_state.fast_mode = fast_mode
    
    # Filters
    st.sidebar.subheader("🎯 Filters")
    
    books = ["All", "Genesis", "Psalms", "Isaiah", "Matthew", "John", "Romans"]
    
    book_filter = st.sidebar.selectbox(
        label="📖 Book",
        options=books,
        help="Filter by book of the Bible",
        key="book_filter_select"
    )
    
    testament_filter = st.sidebar.selectbox(
        label="📜 Testament",
        options=["All", "OT", "NT"],
        help="Filter by Old or New Testament",
        key="testament_filter_select"
    )
    
    st.sidebar.divider()
    
    # AI Mode toggle
    st.sidebar.subheader("🧠 AI Features")
    ai_mode = st.sidebar.toggle(
        label="🤖 AI Exegesis Mode",
        value=st.session_state.ai_mode,
        help="Enable AI-powered analysis of the search results",
        key="ai_mode_toggle"
    )
    st.session_state.ai_mode = ai_mode
    
    # Display stats
    if st.session_state.search_count > 0:
        st.sidebar.divider()
        st.sidebar.subheader("📊 Statistics")
        st.sidebar.metric("Searches", st.session_state.search_count)
        if st.session_state.last_search_time:
            st.sidebar.caption(f"Last search: {st.session_state.last_search_time}")
    
    return provider, model, api_key, ai_mode, top_k, book_filter, testament_filter


def display_verse_card(result: SearchResult, index: int) -> None:
    """Display a single verse card with enhanced styling."""
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
    system_prompt: str
) -> None:
    """Generate and stream AI response based on search results."""
    if not results:
        st.warning("No search results to analyze")
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
    
    # Generate a comprehensive prompt
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
        with st.spinner("Generating AI exegesis..."):
            response_container.markdown("""
            <div class="ai-response">
                <h4>🤖 AI Exegesis <span style="display:inline-block;animation:spin 1s linear infinite;">⏳</span></h4>
                <p><em>Generating insights...</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            for chunk in stream_llm_response(
                provider=provider,
                model_name=model,
                prompt=prompt,
                system_prompt=system_prompt,
                api_key=api_key,
                context_verses=context_verses,
                temperature=0.7,
                max_tokens=1500
            ):
                full_response += chunk
                response_container.markdown(f"""
                <div class="ai-response">
                    <h4>🤖 AI Exegesis</h4>
                    <div style="margin-top: 1rem; line-height: 1.8;">
                        {full_response}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
    except LLMProviderError as e:
        st.error(f"AI Provider Error: {str(e)}")
        response_container.empty()
    except Exception as e:
        st.error(f"Error generating AI response: {str(e)}")
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
        return []


def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    
    # Load Bible data
    data_path = Path(__file__).parent / "data" / "sample_bible.json"
    try:
        verses = load_bible_data_cached(str(data_path))
    except FileNotFoundError:
        st.error(f"❌ Bible data file not found: {data_path}")
        st.info("Please ensure the data directory contains sample_bible.json")
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
        with st.spinner("🔮 Initializing search engine..."):
            try:
                search_engine.initialize(verses)
                st.session_state.search_engine = search_engine
                st.session_state.initialized = True
            except Exception as e:
                st.error(f"❌ Failed to initialize search engine: {str(e)}")
                st.stop()
    
    # Sidebar configuration
    provider, model, api_key, ai_mode, top_k, book_filter, testament_filter = display_sidebar()
    
    # Main content
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<div class="main-header">📖 TBS — Tuixiu Bible Search</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Hybrid Bible Search with AI-Powered Exegesis</div>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.search_count > 0:
            st.metric("Searches", st.session_state.search_count)
    
    st.divider()
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            label="🔍 Search the Bible",
            placeholder="Enter your search query (e.g., 'God created', 'faith in action', 'for God so loved')",
            value=st.session_state.query,
            label_visibility="collapsed",
            key="search_input"
        )
        st.session_state.query = query
    
    with col2:
        search_button = st.button(
            label="🔍 Search",
            type="primary",
            use_container_width=True,
            key="search_button"
        )
    
    # Handle search
    if search_button and query:
        st.session_state.search_triggered = True
    
    if st.session_state.search_triggered and query:
        st.session_state.search_triggered = False
        with st.spinner("🔍 Searching..."):
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
                <span>📊 Found</span>
                <span class="stat-value">{len(results)} results</span>
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
            with st.expander("🤖 AI Exegesis", expanded=True):
                generate_ai_response(
                    provider=provider,
                    model=model,
                    query=query,
                    api_key=api_key if provider in ["google", "openai"] else None,
                    results=results,
                    system_prompt=get_system_prompt()
                )
    
    elif st.session_state.get("error_message"):
        st.error(f"❌ {st.session_state.error_message}")
        st.session_state.error_message = None
    
    elif search_button and query:
        st.info("🔍 No results found. Try adjusting your search query or filters.")
    
    elif not query:
        # Show helpful examples
        st.markdown("""
        ### 💡 Search Examples
        <div class="example-grid">
            <div class="example-chip">"God created"</div>
            <div class="example-chip">"faith salvation"</div>
            <div class="example-chip">"love thy neighbor"</div>
            <div class="example-chip">"for God so loved"</div>
            <div class="example-chip">"blessed are the poor"</div>
            <div class="example-chip">"shepherd"</div>
        </div>
        
        ### 🎯 Tips
        - Use quotes for exact phrases
        - Enable AI Exegesis for deep analysis
        - Use Fast Mode for quicker results
        - Filter by book or testament for focused searches
        """, unsafe_allow_html=True)
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div class="footer">
        <strong>📖 TBS — Tuixiu Bible Search</strong> v1.0.0
        &nbsp;•&nbsp; Powered by Hybrid Search &amp; AI
        &nbsp;•&nbsp; {datetime.now().strftime('%Y')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()