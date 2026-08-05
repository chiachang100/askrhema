# AskRhema from Search to Chat

Yes. The current master prompt is very much designed around a search application UI—sidebar controls, provider/model selectors, search depth, filters, AI-mode toggle, passage cards, etc.

We can simplify it into a conversational Bible search and AI exegesis assistant while keeping the underlying hybrid search + exegesis architecture.

I’d recommend this UX:

```text
┌─────────────────────────────────────────────────────────┐
│  AskRhema                                      ⚙ Settings │
│  Biblical research & exegesis assistant                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  User: What does Romans 8:28 mean in context?           │
│                                                         │
│  AskRhema:                                              │
│  Romans 8:28 says...                                    │
│                                                         │
│  📖 Sources                                             │
│  Romans 8:18–30                                         │
│  Genesis 50:20                                          │
│                                                         │
│                                                         │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  Ask about Scripture...                          ➤      │
└─────────────────────────────────────────────────────────┘
```

What changes

Remove from the primary UI:

Search box + separate Search button
Top-K slider
Category/Testament selectors
AI Mode toggle
Provider selector
Model selector
API-key fields
Separate "Fast Passages" vs "AI Exegesis" screens
Passage cards as the primary interaction

The existing prompt currently explicitly asks for all of these in the sidebar/main screen.

Replace them with:

One chat interface
Persistent conversation history
Streaming assistant responses
Automatic Bible retrieval behind the scenes
Automatic citation/source display
Optional settings drawer/sidebar for advanced configuration
Conversation-aware follow-up questions

So the user simply asks:

"Explain the significance of the wilderness in Scripture."

And AskRhema internally does:

```text
User question
     ↓
Conversation context
     ↓
Hybrid Bible retrieval
(BM25 + embeddings + RRF)
     ↓
Relevant passages
     ↓
LLM
     ↓
Streaming response
     ↓
Answer + citations
```

That actually fits your existing architecture very well. The hybrid engine already combines sparse/dense search using RRF, while the LLM layer already supports streaming across Ollama, Gemini, and OpenAI.

I'd also change the architecture slightly

Instead of making app.py responsible for all the conversational behavior, I'd introduce something like:

```text
askrhema/
├── app.py
├── config.py
├── data/
│   └── sample_bible.json
└── engine/
    ├── __init__.py
    ├── indexer.py
    ├── hybrid_search.py
    ├── llm_provider.py
    └── chat.py              ← new
```

chat.py becomes the orchestration layer:

```text
chat message
    ↓
retrieve relevant Scripture
    ↓
build contextual prompt
    ↓
stream LLM response
    ↓
return answer + sources

This gives you a much cleaner separation:
```

app.py → chatbot UI
chat.py → conversational orchestration
hybrid_search.py → Bible retrieval
llm_provider.py → model abstraction
config.py → configuration
indexer.py → Bible data/indexing
One important UX decision

I would not eliminate the settings completely. Put them behind a small ⚙ Settings control.

There you can expose:

Local Ollama / Gemini / OpenAI
Model
API key
Search depth
Testament/book filtering
Temperature
Maybe "Show retrieved passages"

But 90% of users should never need to see those controls.

That turns AskRhema from:

> "A Bible search interface with an AI mode"

into:

> "A Bible chatbot that happens to have sophisticated retrieval underneath."

I think that's the stronger product direction.

If you want, I can next 
rewrite the attached master prompt into a new chatbot-first master prompt, preserving the technical requirements but replacing the current UI specification and adjusting the architecture accordingly.

---
