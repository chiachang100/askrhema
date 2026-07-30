# TBS — Tuixiu Bible Search

A hybrid, local-first Bible search and exegesis tool built with Python 3.14+, Streamlit, Qdrant, BM25, and AI integration.

## 🚀 Features

- **Hybrid Search**: Combines dense vector search (semantic) with sparse BM25 search (keyword) using Reciprocal Rank Fusion (RRF)
- **AI-Powered Exegesis**: Get AI-generated insights from multiple providers (Ollama, Google Gemini, OpenAI)
- **Local-First**: Uses in-memory Qdrant for vector search and local Ollama support
- **Fast Mode**: Option for quick semantic-only search
- **Filters**: Filter by Bible book and testament
- **Beautiful UI**: Modern Streamlit interface with verse cards and real-time streaming

## 📋 Prerequisites

- **Python 3.14+** (required)
- **uv** package manager
- **Optional**: **Ollama** for local LLM support

## 🛠️ Installation

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

### 2. Clone/Initialize Project
```bash
# Create project directory
uv init tuixiu-bible-search --app
cd tuixiu-bible-search

# Copy all source files into the directory
# Ensure the following structure:
# tuixiu-bible-search/
# ├── app.py
# ├── config.py
# ├── engine/
# │   ├── __init__.py
# │   ├── hybrid_search.py
# │   ├── indexer.py
# │   └── llm_provider.py
# └── data/
#     └── sample_bible.json
```

### 3. Pin Python Version
```bash
uv python pin 3.14
```

### 4. Install Dependencies
```bash
# Install all dependencies
uv add "streamlit@latest" \
       "qdrant-client@latest" \
       "sentence-transformers@latest" \
       "rank-bm25@latest" \
       "google-genai@latest" \
       "openai@latest" \
       "httpx@latest" \
       "pydantic@latest" \
       "numpy@latest" \
       "torch@latest" \
       "transformers@latest" \
       "tokenizers@latest"

# Generate lockfile
uv lock
```

## 🚀 Running the Application
Start the Application
```bash
uv run streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## ⚙️ Configuration
### Ollama Setup
Install Ollama from `ollama.ai`

Pull a model:

```bash
ollama pull llama2
# or
ollama pull llama3
```

Ensure Ollama is running:

```bash
ollama serve
```

### Google Gemini Setup
Get API key from Google AI Studio

Enter the API key in the app sidebar

OpenAI Setup
Get API key from OpenAI Platform

Enter the API key in the app sidebar

## 🔍 Using the Application
- Search: Enter a query in the search box and click "Search"
- Filter: Use sidebar filters to narrow by book or testament
- Fast Mode: Toggle for quick semantic-only search
- AI Mode: Enable AI exegesis for analyzed results
- Provider: Choose between Ollama, Google Gemini, or OpenAI

## 🛠️ Development
### Code Quality
```bash
# Run formatter
uv run black .

# Run linter
uv run ruff check .

# Run type checker
uv run mypy .
```

### Testing
```bash
# Run tests
uv run pytest
```

Updating Dependencies
```bash
# Update to latest versions
uv sync --upgrade
```

## 📊 Architecture
- Frontend: Streamlit with custom CSS
- Vector Search: Qdrant in-memory with sentence-transformers
- Sparse Search: BM25Okapi from rank-bm25
- Fusion: Reciprocal Rank Fusion (RRF) with k=60
- AI: Streaming via Ollama, Google, or OpenAI

## 🐛 Troubleshooting
### Common Issues
Qdrant in-memory issues:

No configuration needed - runs entirely in memory

### Ollama connection errors:
- Ensure Ollama is running: ollama serve
- Check port: `http://localhost:11434`

### API key errors:

Verify API keys are entered correctly in the sidebar

Ensure your Google/OpenAI account has access to the selected models

### Memory issues:

The embedding model uses memory; consider using smaller models

Reduce batch sizes in config.py

## Logs
Check Streamlit logs for detailed error messages:

```bash
uv run streamlit run app.py --logger.level=debug
```

---

## License

RustVerdict is licensed under either of the following licenses, at your option:

- MIT License
- Apache License 2.0

You may choose either license when using, modifying, or distributing RustVerdict.

See:

- [LICENSE-MIT](./LICENSE-MIT)
- [LICENSE-APACHE](./LICENSE-APACHE)

---

## 🙏 Acknowledgments
Built with Streamlit, Qdrant, and sentence-transformers

BM25 implementation from rank-bm25

AI integration with Ollama, Google Gemini, and OpenAI

**Happy Bible Searching! 📖✨**

---
