"""Multi-provider streaming LLM interface."""

import json
from collections.abc import Generator

import httpx
from google import genai
from openai import OpenAI

from config import LLMConfig


class LLMProviderError(Exception):
    """Raised when an LLM provider fails."""

    pass


def stream_llm_response(
    provider: str,
    model_name: str,
    prompt: str,
    system_prompt: str,
    api_key: str | None = None,
    context_verses: list[dict] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Generator[str]:
    """Stream a response from the configured LLM provider."""
    if provider == "ollama":
        yield from _stream_ollama(
            model_name, prompt, system_prompt, temperature, max_tokens
        )
    elif provider == "gemini":
        if not api_key:
            raise ValueError("Gemini API key is required")
        yield from _stream_gemini(
            model_name, prompt, system_prompt, api_key, temperature, max_tokens
        )
    elif provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key is required")
        yield from _stream_openai(
            model_name, prompt, system_prompt, api_key, temperature, max_tokens
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _stream_ollama(
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str]:
    url = f"{LLMConfig.ollama_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": True,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    with (
        httpx.Client(timeout=60.0) as client,
        client.stream("POST", url, json=payload) as response,
    ):
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


def _stream_gemini(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str]:
    client = genai.Client(api_key=api_key)
    full_prompt = f"{system_prompt}\n\n{prompt}"
    response = client.models.generate_content_stream(
        model=model,
        contents=full_prompt,
        config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def _stream_openai(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str]:
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def get_available_models(provider: str) -> list[str]:
    """Return the models available from the specified provider."""
    if provider == "ollama":
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{LLMConfig.ollama_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    if models:
                        return models
        except Exception:
            pass
        return [LLMConfig.ollama_model]
    elif provider == "gemini":
        return ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    elif provider == "openai":
        return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    else:
        return []
