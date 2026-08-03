"""Conversational orchestration layer for AskRhema."""

from typing import Any, Dict, Generator, List, Optional

from config import LLMConfig, SearchConfig
from engine.hybrid_search import HybridSearchEngine, SearchResult
from engine.llm_provider import stream_llm_response


class ChatService:
    """
    Orchestrates conversation, retrieval, and LLM streaming.
    """

    def __init__(
        self,
        search_engine: HybridSearchEngine,
        search_config: SearchConfig,
        llm_config: LLMConfig,
    ) -> None:
        self.search_engine = search_engine
        self.search_config = search_config
        self.llm_config = llm_config

    def process_message(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        book_filter: Optional[str] = None,
        testament_filter: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt_override: Optional[str] = None,   # NEW
    ) -> Generator[tuple[str, List[SearchResult]], None, None]:
        """
        Process a user message: retrieve, build prompt, stream response.

        Args:
            user_message: Current user input.
            conversation_history: List of messages (role, content).
            provider: LLM provider.
            model_name: Model name.
            api_key: API key if needed.
            book_filter: Optional book filter for retrieval.
            testament_filter: Optional testament filter.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            system_prompt_override: Optional custom system prompt to use instead of the default.

        Yields:
            Tuples of (text_chunk, sources_so_far).
        """
        # 1. Retrieve relevant verses
        results = self.search_engine.search(
            query=user_message,
            top_k=self.search_config.top_k,
            book_filter=book_filter,
            testament_filter=testament_filter,
        )

        # 2. Build prompt with context verses
        prompt = self._build_prompt(user_message, conversation_history, results)

        # 3. Stream the LLM response
        full_response = ""
        sources = results
        # Use the override if provided, else the default
        system_prompt = system_prompt_override or self.llm_config.system_prompt

        try:
            for chunk in stream_llm_response(
                provider=provider,
                model_name=model_name,
                prompt=prompt,
                system_prompt=system_prompt,
                api_key=api_key,
                context_verses=[r.verse for r in results],
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                full_response += chunk
                yield chunk, sources
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            yield error_msg, sources
            return

    def _build_prompt(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        results: List[SearchResult],
    ) -> str:
        """
        Build the full prompt for the LLM, including conversation history and retrieved verses.
        """
        verses_text = ""
        if results:
            verses_text = "Here are the relevant Bible passages (with citations):\n\n"
            for r in results:
                ref = r.reference
                text = r.verse["text"]
                verses_text += f"{ref}: {text}\n"
        else:
            verses_text = "No specific Bible passages were retrieved for this query."

        history_text = ""
        if history:
            recent = history[-5:]
            for msg in recent:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    history_text += f"User: {content}\n"
                elif role == "assistant":
                    history_text += f"Assistant: {content}\n"

        prompt = f"""Conversation so far:
{history_text}

Current user query: {user_message}

Retrieved Scripture passages:
{verses_text}

Based on the above conversation and Scripture, please answer the user's query. Provide a clear, grounded response and cite references.
"""
        return prompt