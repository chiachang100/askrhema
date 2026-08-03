# AskRhema - Chat-first Bible Search and Exegesis Assistant

**AskRhema** is a chat-first Bible search and exegesis assistant. It combines hybrid search (BM25 + dense vectors) with large language models to provide grounded, conversational answers to your biblical questions.

> "Ask and it will be given to you; seek and you will find; knock and the door will be opened to you. For everyone who asks receives; the one who seeks finds; and to the one who knocks, the door will be opened." - Matthew 7:7-8, NIV.

---
## Overview

**AskRhema** feels like a modern AI chatbot, but behind the scenes it retrieves relevant Bible passages using a hybrid search engine (sparse **BM25** and dense vector embeddings) and fuses them with **Reciprocal Rank Fusion (RRF)**. The top results are supplied to a language model (**Ollama**, **Google Gemini**, or **OpenAI**) to generate a citation-rich, context‑aware response.

---
## Requirements

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) (project and dependency manager)
- Optional: [Ollama](https://ollama.com/) running locally (if you want to use Ollama)
- Google API key (for Gemini)
- OpenAI API key (for OpenAI)

---
## 🚀 Features

- **Hybrid Search**: Combines dense vector search (semantic) with sparse BM25 search (keyword) using Reciprocal Rank Fusion (RRF)

- **AI-Powered Exegesis**: Get AI-generated insights from multiple providers (Ollama, Google Gemini, OpenAI)

- **Local-First**: Uses in-memory Qdrant for vector search and local Ollama support

- **Fast Mode**: Option for quick semantic-only search

- **Filters**: Filter by Bible book and testament

- **Beautiful UI**: Modern Streamlit interface with verse cards and real-time streaming

---
## 📋 Prerequisites

- **Python 3.14+** (required)
- **uv** package manager
- **Optional**: **Ollama** for local LLM support

---
## 📊 Architecture
- Frontend: Streamlit with custom CSS
- Vector Search: Qdrant in-memory with sentence-transformers
- Sparse Search: BM25Okapi from rank-bm25
- Fusion: Reciprocal Rank Fusion (RRF) with k=60
- AI: Streaming via Ollama, Google, or OpenAI

```text
Streamlit Chat UI
        ↓
Chat Orchestrator (chat.py)
        ↓
Hybrid Retrieval (hybrid_search.py)
   ↙           ↘
BM25          Qdrant (dense)
   ↘           ↙
    RRF Fusion
        ↓
LLM Provider (llm_provider.py)
        ↓
Streaming Response
```

---
## 🛠️ Installation

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
pip install uv
```

### Clone/Initialize Project
```bash
# Create project directory
uv init askrhema --app
cd askrhema
uv python pin 3.14
```

- Install dependencies:

```bash
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
uv add --dev "pytest@latest" "black@latest" "ruff@latest" "mypy@latest"
uv lock
```

- Copy all source files into the directory
- Ensure the following structure:
```text
# askrhema/
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

---

## 🚀 Running the Application
Start the Application
```bash
uv run streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---
## LLM Providers
- **Ollama**: runs locally; ensure Ollama is running and the chosen model is available (e.g., llama3). Set the Ollama URL in the settings.
- **Google Gemini**: enter your API key and select gemini-2.5-flash (default) or another model.
- **OpenAI**: enter your API key and select gpt-4o-mini (default) or another model.

---
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

---
## 🔍 Using the Application
- Search: Enter a query in the search box and click "Search"
- Filter: Use sidebar filters to narrow by book or testament
- Fast Mode: Toggle for quick semantic-only search
- AI Mode: Enable AI exegesis for analyzed results
- Provider: Choose between Ollama, Google Gemini, or OpenAI

---
## 🛠️ Development
- Update: `uv sync --upgrade`
- Running the app: `uv run streamlit run app.py`
- Clean and run the app: `pyclean . --debris && uv run streamlit run app.py`
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format: `uv run black .`
- Type check: `uv run mypy .`

---
## 🐛 Troubleshooting

### Common Issues
- uv not found: install via pip install uv or see uv installation.
- Memory issues: the embedding model may be large; consider using bge-small-en-v1.5 (lighter) by changing EmbeddingConfig.model_name.
- Qdrant in-memory: No configuration needed - runs entirely in memory
- Qdrant in‑memory: index is rebuilt on each app restart; for persistence, switch to disk mode.

### Ollama connection errors:
- Ensure Ollama is running: ollama serve
- Check port: `http://localhost:11434`
- Ollama not responding: ensure Ollama is running (ollama serve) and the model is pulled.

### API key errors:
- Verify API keys are entered correctly in the sidebar
- Ensure your Google/OpenAI account has access to the selected models
- API key errors: verify keys are entered correctly in the Settings sidebar.

### Memory issues:
- The embedding model uses memory; consider using smaller models
- Reduce batch sizes in config.py

---
## Logs
- Check Streamlit logs for detailed error messages:
- 
`uv run streamlit run app.py --logger.level=debug`

---
## License

AskRhema is licensed under either of the following licenses, at your option:

- MIT License
- Apache License 2.0

You may choose either license when using, modifying, or distributing the project.

See:

- [LICENSE-MIT](./LICENSE-MIT)
- [LICENSE-APACHE](./LICENSE-APACHE)

---
## 🙏 Acknowledgments
- Built with Streamlit, Qdrant, and sentence-transformers
- BM25 implementation from rank-bm25
- AI integration with Ollama, Google Gemini, and OpenAI

**Happy Using AskRhema - Chat-first Bible Search and Exegesis Assistant! 📖✨**

---
