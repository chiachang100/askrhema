# AskRhema Master Prompt

# MASTER PROMPT: AskRhema — Chat-First Bible Search & Exegesis Assistant

You are an expert Principal AI Software Engineer.

Your task is to write the complete, production-ready codebase for **AskRhema**, a chat-first Bible search and exegesis assistant built with **Python 3.14+, Streamlit, Qdrant, BM25, Sentence Transformers, Ollama, Google Gemini, and OpenAI**.

AskRhema should feel like a modern conversational AI application rather than a traditional Bible search interface.

The user should primarily interact with AskRhema through **one conversational chat interface**. Bible retrieval, hybrid search, source selection, prompt construction, and LLM provider orchestration should happen automatically behind the scenes.

---

# 1. PRODUCT VISION

AskRhema is a conversational biblical research assistant.

The primary interaction should be:

```text
User asks a question
        ↓
AskRhema understands the conversation
        ↓
AskRhema retrieves relevant Bible passages
        ↓
BM25 + dense vector search are combined
        ↓
Relevant Scripture is supplied to the LLM
        ↓
LLM generates a grounded response
        ↓
Response streams into the chat
        ↓
Relevant Scripture citations/sources are displayed
```

The user should NOT need to understand or manually operate the search engine.

The application should feel similar to a modern AI chatbot, while specializing in Bible search, biblical context, and exegesis.

Examples of natural user interactions:

* "What does Romans 8:28 mean?"
* "Explain the context of this passage."
* "What does the Bible say about forgiveness?"
* "Compare what Jesus says about prayer in Matthew and Luke."
* "Show me passages about God's faithfulness."
* "What is happening in John 15?"
* "How does this connect to the Old Testament?"
* "What does Scripture say about loving your enemies?"

The user should be able to ask follow-up questions naturally without having to repeat the context.

---

# 2. CORE PRODUCT PRINCIPLES

## Chat first

The chat interface is the primary UI.

Do not design AskRhema as a search dashboard with a chat feature added to it.

Instead, design it as a chatbot with intelligent Bible retrieval underneath.

## Retrieval is automatic

Do not expose search mechanics in the primary interface.

The user should not normally have to select:

* Top-K
* Search mode
* BM25
* Vector search
* RRF
* Testament
* Category
* Search filters
* Retrieval strategy

These are internal implementation details.

## Sources remain visible

Although retrieval is automatic, AskRhema must make the biblical sources behind its response easy to inspect.

Responses should include clear Book, Chapter, and Verse references.

Where appropriate, display a compact "Sources" or "Scripture references" section underneath the assistant response.

## Conversation matters

The assistant must preserve conversation history using Streamlit session state.

Follow-up questions should be interpreted in context.

For example:

```text
User:
What does Romans 8:28 mean?

AskRhema:
...

User:
What about verse 29?

AskRhema:
...
```

The second question should be understood as referring to Romans 8.

## Streaming

LLM responses should stream into the chat using Streamlit's streaming capabilities, including `st.write_stream` where appropriate.

---

# 3. TECHNICAL STACK & RUNTIME

Use the following technology stack:

* Python 3.14+
* `uv`
* `pyproject.toml`
* Streamlit
* Qdrant Client using local in-memory mode (`:memory:`)
* `rank-bm25`
* Sentence Transformers
* Ollama
* Google GenAI SDK
* OpenAI SDK
* HTTPX
* Pydantic
* NumPy
* PyTorch
* Transformers
* Tokenizers

Python version:

```text
Python 3.14+
```

The project must be compatible with Python 3.14.

---

# 4. PROJECT MANAGEMENT — UV ONLY

All project setup, dependency management, environment isolation, and execution MUST use `uv`.

Do NOT create or use `requirements.txt`.

Use standard PEP 621 `pyproject.toml`.

The project directory and distribution package name MUST be:

```text
askrhema
```

Initialization:

```bash
uv init askrhema --app
cd askrhema
uv python pin 3.14
```

Add dependencies:

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
```

Development dependencies:

```bash
uv add --dev "pytest@latest" \
              "black@latest" \
              "ruff@latest" \
              "mypy@latest"
```

Generate/update the lock file:

```bash
uv lock
```

Run the application:

```bash
uv run streamlit run app.py
```

Do not generate a `requirements.txt` file.

---

# 5. PROJECT STRUCTURE

Generate the complete project using this architecture:

```text
askrhema/
├── pyproject.toml
├── uv.lock
├── README.md
├── app.py
├── config.py
├── data/
│   └── sample_bible.json
└── engine/
    ├── __init__.py
    ├── indexer.py
    ├── hybrid_search.py
    ├── llm_provider.py
    └── chat.py
```

The new `chat.py` module is intentional.

The architecture should separate:

```text
app.py
    UI and Streamlit session state

chat.py
    Conversation orchestration and retrieval/LLM coordination

hybrid_search.py
    Dense + sparse Bible retrieval

llm_provider.py
    Ollama / Gemini / OpenAI abstraction

indexer.py
    Bible data loading and validation

config.py
    Configuration and prompts
```

---

# 6. FILE: pyproject.toml

Create a complete PEP 621-compatible `pyproject.toml`.

Requirements:

* Project name: `askrhema`
* Python requirement: `>=3.14`
* All dependencies listed
* Development tools configured
* Black configuration
* Ruff configuration
* MyPy configuration
* No `requirements.txt`

Use dependencies corresponding to the UV commands above.

---

# 7. FILE: data/sample_bible.json

Provide a representative JSON array containing at least 8 distinct Bible verses.

The sample dataset must cover:

* Genesis
* Psalms
* Isaiah
* Matthew
* John
* Romans

Each entry must contain:

```json
{
  "id": 1,
  "book": "Genesis",
  "chapter": 1,
  "verse": 1,
  "text": "In the beginning God created the heavens and the earth.",
  "testament": "OT",
  "category": "Creation"
}
```

Required fields:

* `id`
* `book`
* `chapter`
* `verse`
* `text`
* `testament`
* `category`

The sample data is only a representative initial dataset.

---

# 8. FILE: config.py

Create dataclasses/configuration classes for:

## SearchConfig

Include:

* `top_k`
* `rrf_k_constant`
* `vector_size`
* `collection_name`

Default RRF constant:

```text
60
```

## LLMConfig

Include:

* system prompt
* model names
* Ollama URL
* temperature
* max tokens
* provider configuration

Default models should support:

```text
Ollama
Google Gemini: gemini-2.5-flash
OpenAI: gpt-4o-mini
```

## EmbeddingConfig

Include:

* model name
* device
* batch size

Support an embedding model such as:

```text
all-MiniLM-L6-v2
```

or:

```text
bge-small-en-v1.5
```

## Default system prompt

Use a grounding-first biblical assistant prompt along these lines:

```text
You are AskRhema, an expert biblical assistant.

Ground your answer strictly in the provided Bible passages.

Always cite the Book, Chapter, and Verse for every passage reference.

Do not invent biblical quotations or references.

When the retrieved passages do not adequately support an answer, clearly say so rather than presenting unsupported claims as Scripture.

Distinguish clearly between what the biblical text says and interpretive explanation.
```

The prompt may be expanded where necessary to support conversational context, but biblical grounding and citation requirements must remain explicit.

---

# 9. FILE: engine/indexer.py

Implement complete Bible data loading and validation.

Required function:

```python
def load_bible_data(file_path: str) -> list[dict]: ...
```

Responsibilities:

* Read JSON
* Parse the dataset
* Validate required fields
* Return validated Bible records
* Raise useful exceptions for malformed or invalid data

Required fields:

```text
id
book
chapter
verse
text
testament
category
```

Also implement:

```python
def get_verse_reference(verse: dict) -> str: ...
```

Return a formatted reference such as:

```text
John 3:16
```

or:

```text
Romans 8:28
```

Use proper type hints and comprehensive docstrings.

---

# 10. FILE: engine/hybrid_search.py

Implement the hybrid Bible retrieval engine.

Create:

```text
HybridSearchEngine
```

The engine must combine:

1. Dense vector search
2. Sparse BM25 search
3. Reciprocal Rank Fusion

Use:

* Sentence Transformers
* Qdrant
* BM25Okapi

Qdrant should operate locally using in-memory mode:

```text
:memory:
```

Use Streamlit caching appropriately so expensive models and indexes are not unnecessarily recreated across interactions.

Use `@st.cache_resource` for expensive resources such as:

* embedding model
* Qdrant client
* initialized search engine

Use `@st.cache_data` for suitable data-loading operations.

---

# 11. HYBRID SEARCH PROCESS

When a user submits a question:

### Dense search

Generate an embedding for the query and retrieve semantically relevant passages from Qdrant.

### Sparse search

Tokenize the query and use BM25Okapi to retrieve keyword-relevant passages.

### Fusion

Combine results using Reciprocal Rank Fusion:

```text
RRF_Score(d) = Σ 1 / (k + r_m(d))
```

with:

```text
k = 60
```

Return results ranked by fused score.

Each search result should contain useful metadata such as:

* Bible book
* chapter
* verse
* text
* category
* testament
* dense rank
* sparse rank
* fused RRF score

Create an appropriate `SearchResult` dataclass/model.

---

# 12. SEARCH FILTERING

The search engine should support filtering by:

* Bible book
* testament

Filtering should be implemented internally and should not dominate the primary chatbot UI.

The application may expose advanced filtering through Settings, but the default experience should be automatic retrieval.

---

# 13. FILE: engine/llm_provider.py

Implement a unified streaming interface:

```python
def stream_llm_response(
    provider: str,
    model_name: str,
    prompt: str,
    system_prompt: str,
    api_key: str | None = None,
    context_verses: list[dict] | None = None,
) -> Generator[str, None, None]: ...
```

Implement all three providers.

## Ollama

Use HTTPX streaming against:

```text
http://localhost:11434/api/generate
```

Use a streaming POST request with the model, prompt, and system prompt.

Yield response text incrementally.

## Google Gemini

Use:

```python
from google import genai
```

Instantiate:

```python
client = genai.Client(api_key=api_key)
```

Use streaming content generation.

Default model:

```text
gemini-2.5-flash
```

## OpenAI

Use:

```python
from openai import OpenAI
```

Instantiate:

```python
client = OpenAI(api_key=api_key)
```

Use streamed chat completions.

Default model:

```text
gpt-4o-mini
```

Handle provider errors gracefully.

Never expose API keys in error messages or logs.

Also implement:

```python
def get_available_models(provider: str) -> list[str]: ...
```

---

# 14. FILE: engine/chat.py

Create a conversational orchestration layer.

This module is responsible for connecting:

```text
conversation
    ↓
retrieval
    ↓
context construction
    ↓
LLM
    ↓
streamed response
```

Implement an appropriate chat service/class.

Responsibilities should include:

* Accepting the current user message
* Receiving conversation history
* Retrieving relevant Bible passages
* Constructing the contextual prompt
* Passing retrieved passages to the LLM
* Streaming the response
* Returning/retaining source references
* Handling provider errors

The conversation layer should distinguish between:

1. User conversation history
2. Retrieved Scripture context
3. System instructions

Do not blindly send unlimited conversation history to the model.

Use a sensible bounded history strategy.

---

# 15. CONVERSATIONAL RETRIEVAL

AskRhema should use the user's conversational context when appropriate.

For example:

```text
User:
What does John 15 teach about abiding?

Assistant:
...

User:
How does that relate to fruit?

Assistant:
...
```

The second message should be interpreted using the conversation context.

However, retrieval should remain focused on the current question and relevant context rather than blindly embedding the entire conversation.

The system should be capable of reformulating or contextualizing a follow-up question internally before retrieval when necessary.

---

# 16. FILE: app.py — CHAT-FIRST UI

The UI is the most important redesign.

Do NOT build the main interface around a Bible search form.

Do NOT put the primary search controls in the sidebar.

Do NOT require the user to choose a provider, model, search depth, category, testament, or AI mode before asking a question.

The default UI should resemble a modern AI chatbot.

---

# 17. PRIMARY UI

The application should contain:

## Header

Display:

```text
AskRhema
```

with a concise description such as:

```text
A conversational Bible research and exegesis assistant.
```

Keep branding clean and understated.

## Conversation

Use Streamlit chat components such as:

```python
st.chat_message()
```

Display the conversation in chronological order.

User messages should appear as user messages.

Assistant responses should appear as assistant messages.

## Input

Use:

```python
st.chat_input()
```

The placeholder should be something similar to:

```text
Ask AskRhema about Scripture...
```

The user should be able to submit questions directly.

---

# 18. RESPONSE EXPERIENCE

Assistant responses must stream progressively.

Use:

```python
st.write_stream()
```

or an equivalent Streamlit streaming mechanism.

The interaction should feel immediate and conversational.

Avoid forcing the user to wait for an entire response before anything appears.

---

# 19. SCRIPTURE SOURCES

Every response that relies upon retrieved Scripture should clearly cite the relevant biblical references.

For example:

```text
Romans 8:28
Romans 8:29
Genesis 50:20
```

Do not overwhelm the user with internal retrieval metadata by default.

Do NOT normally display:

```text
RRF score: 0.0314
Dense rank: 2
Sparse rank: 4
```

These are implementation details.

Instead, present a concise source section such as:

```text
Sources

Romans 8:28
Romans 8:29
Genesis 50:20
```

The UI may optionally provide an expandable details area for debugging/development purposes.

---

# 20. CHAT SESSION STATE

Use Streamlit session state to maintain:

* conversation messages
* provider selection
* model selection
* API keys
* optional configuration
* source metadata where appropriate

The conversation should survive Streamlit reruns during the current session.

Use a message structure that can retain both:

```text
role
content
```

and optional source metadata.

---

# 21. SETTINGS

The primary interface should remain simple.

Advanced configuration should be hidden behind a settings mechanism, such as:

```text
⚙ Settings
```

or a compact sidebar/drawer.

Settings may contain:

* LLM provider
* model
* API key
* search depth
* optional Bible filters
* temperature
* max tokens
* optional "show retrieval details" setting

These settings must not clutter the primary chat experience.

API keys must use password inputs.

Do not hardcode API keys.

Never log or print API keys.

---

# 22. PROVIDER DEFAULTS

The application should support:

### Local Ollama

Use when available.

### Google Gemini

Use:

```text
gemini-2.5-flash
```

### OpenAI

Use:

```text
gpt-4o-mini
```

The user may select a provider through Settings.

The provider selection should not be the central interaction model of the application.

---

# 23. INITIAL / EMPTY CHAT STATE

When no conversation exists, display a welcoming empty state.

For example:

```text
AskRhema

Explore Scripture through conversation.

Ask a question about a passage, biblical theme,
person, doctrine, or context.
```

Optionally show a few clickable example prompts:

```text
What does Romans 8:28 mean?

Explain the context of John 15.

What does Scripture say about forgiveness?

How does the Old Testament point toward Christ?
```

These should be suggestions, not mandatory workflows.

---

# 24. ERROR HANDLING

Provide friendly user-facing errors.

Examples:

* Ollama is unavailable
* Invalid API key
* Provider request failed
* Bible dataset failed to load
* Embedding model failed to initialize
* Search/index initialization failed

Do not expose stack traces or secrets in the normal user interface.

Developer-friendly logging may be used where appropriate, but never log API keys.

---

# 25. PERFORMANCE

Use appropriate Streamlit caching.

Use:

```python
@st.cache_resource
```

for expensive persistent resources such as:

* Sentence Transformer models
* Qdrant client
* Hybrid search engine
* Other expensive initialized resources

Use:

```python
@st.cache_data
```

for suitable immutable/reusable data loading.

Implement lazy loading where appropriate.

Use batch processing when indexing Bible data.

Do not repeatedly initialize the embedding model on every chat message.

Do not rebuild the entire search index for every user interaction.

---

# 26. SECURITY

API keys must:

* Never be hardcoded
* Never be committed
* Never be printed
* Never be logged
* Be entered through password fields
* Be stored only in appropriate runtime/session state

Handle external API errors safely.

Do not leak provider credentials into prompts, source displays, or exception messages.

---

# 27. BIBLICAL GROUNDING

AskRhema must prioritize the provided Bible dataset as the source of biblical textual grounding.

The LLM must not fabricate:

* Bible verses
* Bible references
* quotations
* claims that are presented as direct Scripture

When making interpretive statements, distinguish interpretation from direct biblical wording.

When relevant Scripture is not available in the supplied dataset, the assistant should acknowledge the limitation rather than pretending the dataset contains a passage it does not contain.

The assistant should cite Book, Chapter, and Verse wherever biblical passages are referenced.

---

# 28. UX PRINCIPLES

The interface should be:

* Clean
* Minimal
* Conversational
* Responsive
* Easy to understand
* Suitable for desktop and smaller screens

Avoid unnecessary controls.

The user should be able to open AskRhema and immediately type a question.

The main mental model should be:

```text
"Ask AskRhema a question."
```

not:

```text
"Configure a search and then execute it."
```

---

# 29. WHAT NOT TO BUILD

Do NOT make the primary interface:

* A search form
* A dashboard
* A Bible database browser
* A collection of search filters
* A provider configuration screen
* A technical retrieval-debugging interface

Do NOT require:

* Search button
* Top-K slider
* AI mode toggle
* Fast Passages Only toggle
* Manual category selection
* Manual Testament selection
* Manual provider selection before first message

These may exist under Settings where appropriate, but they must not dominate the experience.

---

# 30. FILE: README.md

Provide a complete README containing:

## Overview

Explain AskRhema as a chat-first Bible research and exegesis assistant.

## Requirements

* Python 3.14+
* uv
* Optional Ollama
* Google API key if using Gemini
* OpenAI API key if using OpenAI

## Setup

Use the UV workflow.

Example:

```bash
uv init askrhema --app
cd askrhema
uv python pin 3.14
```

Install dependencies using `uv add`.

## Running

```bash
uv run streamlit run app.py
```

## Ollama

Explain how to configure/use local Ollama.

## Google Gemini

Explain where the API key is configured.

## OpenAI

Explain where the API key is configured.

## Architecture

Explain:

```text
Streamlit Chat UI
        ↓
Chat Orchestrator
        ↓
Hybrid Retrieval
   ↙           ↘
BM25          Qdrant
   ↘           ↙
    RRF Fusion
        ↓
LLM Provider
        ↓
Streaming Response
```

## Troubleshooting

Include common issues involving:

* uv
* Python
* Streamlit
* Ollama
* API keys
* model initialization
* Qdrant
* Sentence Transformers

## Development

Include commands for:

```bash
uv run pytest
uv run ruff check .
uv run black .
uv run mypy .
```

---

# 31. CODE QUALITY

All generated code must be complete and production-oriented.

Requirements:

* Python 3.14 type syntax
* Full type hints
* Comprehensive docstrings for public functions/classes
* PEP 8 compliance
* Sensible module boundaries
* Clear naming
* Proper exception handling
* No unnecessary duplication
* No dead code
* No placeholder implementations

Use dataclasses where configuration/data models are appropriate.

Use modern Python syntax.

---

# 32. TESTING

Include meaningful tests where appropriate.

At minimum, test:

* Bible JSON loading
* Bible record validation
* Verse reference formatting
* BM25/search behavior
* RRF score calculation
* Search result ordering
* Chat prompt construction
* Provider-independent chat orchestration where practical

Tests should not require external API calls unless explicitly mocked.

---

# 33. GENERATION MANDATE

Generate the complete AskRhema codebase.

You MUST provide every required file completely.

Do not use:

```text
TODO
FIXME
...
pass
# implement later
# omitted for brevity
```

Do not provide partial implementations.

Do not leave functions unfinished.

Every import must be present.

Every required function must be implemented.

Every required class must be implemented.

The generated application must be internally consistent across all modules.

The code must target Python 3.14+.

---

# 34. FINAL ARCHITECTURAL REQUIREMENT

The most important product requirement is:

**AskRhema must feel like a chatbot first and a search engine second.**

The user should be able to:

```text
Open AskRhema
    ↓
Read a welcoming prompt
    ↓
Ask a Bible question
    ↓
Receive a streamed answer
    ↓
See Scripture sources
    ↓
Ask a follow-up question
    ↓
Continue the conversation naturally
```

All sophisticated functionality—BM25, embeddings, Qdrant, RRF, source retrieval, context construction, and LLM provider selection—should support this conversational experience without making the user manage the underlying machinery.

The result should be a clean, modern, production-ready **AskRhema Bible chatbot** built on the existing hybrid search and multi-provider architecture.

---
