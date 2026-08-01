MASTER PROMPT: AskRhema [Python 3.14+ & Native uv Project Setup]
You are an expert Principal AI Software Engineer. You are tasked with writing the complete, production-ready codebase for SeekRheAskRhema-first Bible search and exegesis tool built with Python 3.14+, Streamlit, Qdrant, BM25, Ollama, and Cloud LLM fallbacks (Google Gemini and OpenAI).

All project setup, dependency management, environment isolation, and execution MUST strictly use uv and standard pyproject.toml. Do NOT use or generate a requirements.txt file. The project directory and distribution package name MUST be askrhema.

You must write ALL code files completely—do not use placeholders like # TODO: implement rest or ....

1. TECHNICAL STACK & RUNTIME REQUIREMENTS
Python Runtime: Python 3.14+ (uv python pin 3.14)

Package & Project Manager: uv (PEP 621 compliant pyproject.toml and uv.lock ONLY — no requirements.txt)

Frontend / UI: Streamlit (latest version using st.session_state and real-time st.write_stream)

Vector Index & Engine: qdrant-client (running in local in-memory mode :memory:)

Sparse Keyword Search: rank-bm25 (using BM25Okapi)

Dense Vector Embeddings: sentence-transformers (all-MiniLM-L6-v2 or bge-small-en-v1.5)

LLM Integrations:

Local: Ollama API via httpx (streaming HTTP POST to http://localhost:11434/api/generate)

Google Cloud: google-genai SDK (from google import genai using model gemini-2.5-flash)

OpenAI: openai v1.x+ client (from openai import OpenAI using model gpt-4o-mini)

2. UV WORKFLOW & COMMAND PROTOCOL
Execute project initialization, Python version pinning, dependency installation, and application execution strictly using uv:

bash
# 1. Initialize project directory named askrhema
uv init askrhema --app
cd askrhema

# 2. Pin Python version to 3.14
uv python pin 3.14

# 3. Add core dependencies with @latest to get the most recent releases
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

# 4. Add development dependencies (optional)
uv add --dev "pytest@latest" "black@latest" "ruff@latest" "mypy@latest"

# 5. Generate lock file
uv lock

# 6. Run application
uv run streamlit run app.py
Alternative approach using version constraints (for stability):

bash
# Add dependencies with minimum version constraints
uv add "streamlit>=1.40.0" \
       "qdrant-client>=1.12.0" \
       "sentence-transformers>=3.0.0" \
       "rank-bm25>=0.2.2" \
       "google-genai>=0.1.0" \
       "openai>=1.50.0" \
       "httpx>=0.27.0" \
       "pydantic>=2.9.0" \
       "numpy>=1.26.0" \
       "torch>=2.0.0" \
       "transformers>=4.30.0" \
       "tokenizers>=0.13.0"
To update dependencies to latest versions later:

bash
uv sync --upgrade
3. FILE SYSTEM & MODULE ARCHITECTURE
Generate complete source code for every file in the following structure:

text
askrhema/
├── pyproject.toml           # Native uv dependency manifest (PEP 621) targeting Python 3.14+
├── uv.lock                  # Auto-generated lockfile
├── README.md                # Execution guide using `uv run`
├── app.py                   # Main Streamlit UI layout and session coordinator
├── config.py                # Configuration defaults, constants, and system prompts
├── data/
│   └── sample_bible.json    # Initial dataset schema containing biblical verses
└── engine/
    ├── __init__.py          # Module initialization
    ├── indexer.py           # Loads JSON dataset into memory structures
    ├── hybrid_search.py     # Executes Sparse + Dense search & RRF fusion
    └── llm_provider.py      # Stream generator for Ollama, Google GenAI, and OpenAI
4. DETAILED SPECIFICATION PER FILE
File 1: pyproject.toml
Project name: "askrhema"

Must specify requires-python = ">=3.14"

Uses standard PEP 621 [project] metadata with dependencies array managed by uv

Include all dependencies from the uv add commands above

Include tool configurations for Black, Ruff, and MyPy for code quality

Explicitly omit requirements.txt

File 2: data/sample_bible.json
Provide a representative JSON array with at least 8 distinct verses covering Genesis, Psalms, Isaiah, Matthew, John, and Romans. Structure each entry as:

json
{
  "id": 1,
  "book": "Genesis",
  "chapter": 1,
  "verse": 1,
  "text": "In the beginning God created the heavens and the earth.",
  "testament": "OT",
  "category": "Creation"
}
File 3: config.py
Define configuration classes/dataclasses for:

SearchConfig: Search parameters (top_k, rrf_k_constant, vector_size, collection_name)

LLMConfig: LLM settings (system prompts, model names, URLs, temperature, max_tokens)

EmbeddingConfig: Embedding model settings (model_name, device, batch_size)

Store default system prompts for the AI exegesis engine:

text
"You are AskRhema, an expert biblical assistant. Ground your answer strictly in the provided Bible passages. Always cite the Book, Chapter, and Verse for every passage reference."
File 4: engine/indexer.py
Function load_bible_data(file_path: str) -> list[dict] to parse and validate the JSON data

Function get_verse_reference(verse: dict) -> str to format book/chapter/verse references

Validate all required fields: id, book, chapter, verse, text, testament, category

Raise appropriate exceptions for invalid data

File 5: engine/hybrid_search.py
Class HybridSearchEngine:

Uses @st.cache_resource semantics to keep SentenceTransformer and Qdrant in-memory client persistent across Streamlit user interactions

Implements indexing logic for both Qdrant (PointStruct with vectors and JSON payload) and BM25 (BM25Okapi)

Implements search with filtering capabilities (filtering by book or testament inside Qdrant)

Implements Reciprocal Rank Fusion (RRF):

R
R
F
_
S
c
o
r
e
(
d
)
=
∑
m
∈
M
1
k
+
r
m
(
d
)
(
k
=
60
)
RRF_Score(d)=∑ 
m∈M
​
  
k+r 
m
​
 (d)
1
​
 (k=60)

Returns a list of SearchResult objects containing metadata, individual dense/sparse ranks, and fused RRF score

Include proper type hints and docstrings

File 6: engine/llm_provider.py
Implements a unified streaming function:

python
def stream_llm_response(
    provider: str, 
    model_name: str, 
    prompt: str, 
    system_prompt: str, 
    api_key: str | None = None,
    context_verses: list[dict] | None = None
) -> Generator[str, None, None]:
Google GenAI Handler: Uses from google import genai; instantiates client = genai.Client(api_key=api_key) and iterates over client.models.generate_content_stream()

OpenAI Handler: Uses from openai import OpenAI; instantiates client = OpenAI(api_key=api_key) and streams chat.completions.create()

Ollama Handler: Uses httpx.stream("POST", "http://localhost:11434/api/generate", json={"model": model_name, "prompt": prompt, "system": system_prompt}) to yield decoded response fields

Include get_available_models(provider: str) -> list[str] function

File 7: app.py
Modern Streamlit layout with:

Sidebar:

Provider Selectbox ("Local Ollama", "Google Gemini", "OpenAI")

Model Selectbox dynamically based on Provider

API key input fields (using st.text_input(type="password")) with session state retention

Search depth slider (Top-K: 1 to 10)

Category / Testament filter selectbox

Toggle for AI Mode

Main Screen:

Header for AskRhema

Search input box with search button

Toggle between Fast Passages Only mode and AI Exegesis Mode

Displays retrieved passage cards (highlighting Book, Chapter:Verse, text, category tag, and RRF score)

If AI Exegesis Mode is active, streams the LLM's structured commentary using st.write_stream()

Proper session state management

File 8: README.md
Provide step-by-step setup and execution instructions including:

Prerequisites (Python 3.14+, uv, optional Ollama)

Installation steps using uv init, uv python pin, uv add @latest

Running the application with uv run streamlit run app.py

Configuration instructions for Ollama, Google Gemini, and OpenAI

Troubleshooting common issues

Development notes

1. ADDITIONAL REQUIREMENTS
Code Quality
Use Python 3.14 type hint syntax throughout

Include comprehensive docstrings for all public functions and classes

Implement proper error handling with try/except blocks

Use dataclasses for configuration objects

Follow PEP 8 style guidelines

Performance
Use @st.cache_resource for expensive operations (model loading, indexing)

Use @st.cache_data for data loading

Implement lazy loading for models and indices

Use batch processing where appropriate

Security
API keys should be stored in session state, not hardcoded

Use password input fields for API keys

Never log or print API keys

User Experience
Show loading spinners during long operations

Display clear error messages

Use consistent styling and branding

Provide helpful tooltips and labels

6. GENERATION MANDATE
Begin generating the codebase. Output every file completely with all imports, full implementations, and proper Python 3.14 type hint syntax. Do not use ellipsis or TODO placeholders—every function must be fully implemented.

Important Notes:

The uv add commands use @latest to get the most recent releases of each package

Include numpy, torch, transformers, and tokenizers as dependencies (required by sentence-transformers)

The project uses Python 3.14+ features - ensure all code is compatible

All Streamlit caching should use the appropriate decorator for the use case

The UI should be responsive and work well on different screen sizes

