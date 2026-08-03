# AskRhema Master Prompt Notes

Absolutely. I’ve kept the original technical stack, uv workflow, hybrid BM25 + dense retrieval, RRF, three LLM providers, security requirements, and full-code generation mandate, while redesigning the product around a chat-first experience. The original prompt’s technical foundation is preserved here.

Here is the replacement master prompt:

- [AskRhema Master Prompt](./AskRhema-Master-Prompt.md)

This version should be a much better master prompt for generating the actual application because it changes the design philosophy rather than merely hiding a few controls. The important shift is that chat.py now owns the conversational orchestration while app.py becomes primarily the chat UI.

---
I would modify the existing app rather than start over.

Because you already have a working implementation, the safest path is to treat this as a UI/UX refactor, not a new application. Your existing master prompt already has the pieces we want to preserve: the hybrid search engine, Qdrant/BM25 retrieval, LLM provider abstraction, streaming, caching, and configuration.

My recommendation

Keep the existing GitHub repo and workspace. Create a feature branch.

Something like:

main
  │
  └── feature/chat-first-ui

Then refactor toward:

                    EXISTING APP
                         │
          ┌──────────────┴──────────────┐
          │                             │
     KEEP / IMPROVE                 REPLACE
          │                             │
   Hybrid search                   Old Streamlit UI
   Qdrant                          Search box
   BM25                            Search button
   Embeddings                      Top-K controls
   RRF                             AI toggle
   LLM providers                   Provider clutter
   Streaming                       Passage-card workflow
   Bible indexing
          │
          └──────────────┬──────────────┘
                         ↓
                  CHAT-FIRST UI
                         │
                  st.chat_message
                  st.chat_input
                  conversation history
                  streamed answers
                  Scripture sources
                  hidden retrieval
Why I wouldn't start from scratch

Starting over creates unnecessary risk.

You would have to recreate and retest things that are already working:

Bible ingestion
embeddings
Qdrant indexing
BM25
RRF
LLM integrations
Ollama
Gemini
OpenAI
streaming
caching
error handling

The existing architecture already separates those concerns reasonably well.

The biggest change is app.py, plus adding the conversational orchestration layer I proposed.

The one thing I would not do

Don't simply tell an AI coding agent:

"Change the UI to a chatbot."

That's likely to result in it rewriting working pieces unnecessarily.

Instead, I'd give it a migration/refactoring prompt with explicit instructions:

Preserve all working backend functionality. Do not rewrite the search/indexing/LLM infrastructure unless necessary. Replace the existing Streamlit interaction model with a chat-first interface.

Then make the changes incrementally.

Recommended migration sequence

Phase 1 — Protect the existing app

Create a branch and make sure the current application runs.

git checkout -b feature/chat-first-ui
uv run streamlit run app.py

Commit the working baseline.

Phase 2 — Refactor the UI

Change app.py from:

Sidebar configuration
      ↓
Search input
      ↓
Search button
      ↓
Passage results
      ↓
Optional AI response

to:

Chat history
      ↓
st.chat_input()
      ↓
retrieve automatically
      ↓
LLM automatically
      ↓
st.write_stream()
      ↓
Sources

Phase 3 — Add engine/chat.py

This is the important architectural improvement.

It becomes the bridge between your existing search and LLM layers:

app.py
  │
  │ user message
  ↓
chat.py
  │
  ├── hybrid_search.py
  │       ↓
  │    Scripture
  │
  └── llm_provider.py
          ↓
       response

Phase 4 — Hide advanced controls

Move provider/model/API-key/search controls into Settings.

Don't delete the functionality—just get it out of the user's face.

Phase 5 — Test the new experience

Test actual conversations rather than just individual searches:

"What does Romans 8:28 mean?"

"Explain verse 29."

"How does that connect to verse 18?"

"Show me another passage with a similar theme."

"Compare this with Genesis 50:20."

That's where we'll discover whether the new conversational architecture actually works.

What I'd do in your position

Existing GitHub repo → new feature branch → incremental refactor → test → merge.

I would not create a new GitHub repository unless the current codebase has become so tangled that the backend cannot reasonably be separated from the UI.

And since you said the application is already running, I'd want to inspect the actual code before changing anything. The master prompt tells us what was intended, but the running code tells us what you actually have.

If you upload the current project (or the relevant source files), I can help you map it into:

"keep / modify / move / delete / add"

and then we can produce a migration prompt specifically for your existing codebase, rather than asking an AI coding agent to rebuild AskRhema from scratch.

---
