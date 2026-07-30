"""LLM provider module for streaming responses from multiple AI services."""

import json
from typing import Generator, List, Dict, Any, Optional
import httpx
import streamlit as st

from config import DEFAULT_LLM_CONFIG, LLMConfig


class LLMProviderError(Exception):
    """Exception raised for LLM provider errors."""
    pass


def get_available_models(provider: str) -> List[str]:
    """
    Get available models for a given provider.
    
    Args:
        provider: The provider name ('ollama', 'google', 'openai')
        
    Returns:
        List of model names
    """
    if provider == "ollama":
        return ["llama2", "llama3", "mixtral", "gemma", "phi"]
    elif provider == "google":
        return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    elif provider == "openai":
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    else:
        return []


def _format_context_verses(context_verses: List[Dict[str, Any]]) -> str:
    """Format context verses for the LLM prompt."""
    if not context_verses:
        return "No specific Bible passages provided."
    
    formatted = []
    for verse in context_verses:
        reference = f"{verse['book']} {verse['chapter']}:{verse['verse']}"
        formatted.append(f"{reference} - {verse['text']}")
    
    return "\n".join(formatted)


def stream_llm_response(
    provider: str,
    model_name: str,
    prompt: str,
    system_prompt: str,
    api_key: Optional[str] = None,
    context_verses: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> Generator[str, None, None]:
    """
    Stream a response from the specified LLM provider.
    
    Args:
        provider: The provider name ('ollama', 'google', 'openai')
        model_name: The model name to use
        prompt: The user prompt
        system_prompt: The system prompt
        api_key: API key for cloud providers
        context_verses: List of context verses
        temperature: Temperature for generation (uses default if None)
        max_tokens: Maximum tokens to generate (uses default if None)
        
    Yields:
        Chunks of the response as strings
        
    Raises:
        LLMProviderError: If there's an error with the provider
        ValueError: If provider is unsupported or missing required config
    """
    config = DEFAULT_LLM_CONFIG
    temperature = temperature or config.temperature
    max_tokens = max_tokens or config.max_tokens
    
    # Format context
    context_text = _format_context_verses(context_verses) if context_verses else ""
    
    # Build the full prompt
    full_prompt = f"{system_prompt}\n\nContext passages:\n{context_text}\n\nUser query: {prompt}"
    
    if provider == "ollama":
        yield from _stream_ollama(model_name, full_prompt, system_prompt, temperature, max_tokens)
    elif provider == "google":
        if not api_key:
            raise ValueError("API key required for Google Gemini")
        yield from _stream_google(model_name, full_prompt, system_prompt, api_key, temperature, max_tokens)
    elif provider == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI")
        yield from _stream_openai(model_name, full_prompt, system_prompt, api_key, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _stream_ollama(
    model_name: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int
) -> Generator[str, None, None]:
    """Stream response from Ollama."""
    config = DEFAULT_LLM_CONFIG
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    
    try:
        with httpx.Client(timeout=config.timeout_seconds) as client:
            with client.stream(
                "POST",
                config.ollama_url,
                json=payload
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                            
    except httpx.TimeoutException:
        raise LLMProviderError("Ollama request timed out. Ensure Ollama is running and responsive.")
    except httpx.ConnectError:
        raise LLMProviderError(f"Failed to connect to Ollama. Ensure Ollama is running at {config.ollama_url}")
    except httpx.HTTPStatusError as e:
        raise LLMProviderError(f"Ollama returned error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        raise LLMProviderError(f"Ollama error: {str(e)}")


def _stream_google(
    model_name: str,
    prompt: str,
    system_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int
) -> Generator[str, None, None]:
    """Stream response from Google Gemini."""
    try:
        from google import genai
        
        client = genai.Client(api_key=api_key)
        
        # Combine system prompt with user prompt
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = client.models.generate_content_stream(
            model=model_name,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
                
    except ImportError:
        raise LLMProviderError("google-genai package not installed. Run: uv add google-genai")
    except Exception as e:
        raise LLMProviderError(f"Google Gemini error: {str(e)}")


def _stream_openai(
    model_name: str,
    prompt: str,
    system_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int
) -> Generator[str, None, None]:
    """Stream response from OpenAI."""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except ImportError:
        raise LLMProviderError("openai package not installed. Run: uv add openai")
    except Exception as e:
        raise LLMProviderError(f"OpenAI error: {str(e)}")