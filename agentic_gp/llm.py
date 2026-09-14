"""Chat-model factory shared by the baseline and the student starter code.

The provider is chosen with environment variables so the same agent code runs
against a local open-source model (Hermes via Ollama) or a hosted API:

    RACE_LLM_PROVIDER = ollama | anthropic | openai        (default: ollama)
    RACE_LLM_MODEL    = model name for that provider       (default: see DEFAULT_MODELS)
    OLLAMA_BASE_URL   = http://localhost:11434/v1          (ollama only)
    ANTHROPIC_API_KEY / OPENAI_API_KEY                     (hosted providers)

Usage:
    from agentic_gp.llm import get_chat_model
    llm = get_chat_model()                         # provider + model from env
    llm = get_chat_model(provider="anthropic")     # planner on Claude Opus 5
    llm = get_chat_model(provider="anthropic", model="claude-haiku-4-5")  # cheap executor
"""

from __future__ import annotations

import os

DEFAULT_MODELS = {
    "ollama": "hermes3",            # Nous Research Hermes 3 (Llama 3.1 based), `ollama pull hermes3`
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
}


def get_chat_model(model: str | None = None, provider: str | None = None, temperature: float = 0.0):
    """Return a LangChain chat model that supports `.bind_tools()`."""
    provider = (provider or os.getenv("RACE_LLM_PROVIDER", "ollama")).lower()
    model = model or os.getenv("RACE_LLM_MODEL") or DEFAULT_MODELS.get(provider)
    if model is None:
        raise ValueError(f"Unknown provider {provider!r}; choose one of {sorted(DEFAULT_MODELS)}")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # Claude 5-family models reason adaptively by default and reject sampling
        # parameters such as temperature, so none are passed here.
        return ChatAnthropic(model=model, max_tokens=2048)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "ollama":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        )

    raise ValueError(f"Unknown provider {provider!r}; choose one of {sorted(DEFAULT_MODELS)}")


def describe_model() -> str:
    provider = os.getenv("RACE_LLM_PROVIDER", "ollama").lower()
    model = os.getenv("RACE_LLM_MODEL") or DEFAULT_MODELS.get(provider, "?")
    return f"{provider}:{model}"
