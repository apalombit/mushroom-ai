"""
Provider-agnostic LLM client using LiteLLM + Instructor.

Usage:
    from llm.client import structured_completion, completion

    # Structured output (preferred — validated Pydantic model)
    result = structured_completion("...", response_model=MySchema, system="...")

    # Plain text
    text = completion("...")

Provider switching: change LLM_PROVIDER and LLM_MODEL in .env — no code changes.
"""

import logging
from functools import lru_cache

import instructor
import litellm
from litellm import completion as _litellm_completion

from config import settings

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True

if settings.llm_base_url:
    litellm.api_base = settings.llm_base_url


def _get_model_string() -> str:
    """Build LiteLLM model string from provider + model name."""
    provider = settings.llm_provider.lower()
    model = settings.llm_model

    if provider == "ollama":
        return f"ollama_chat/{model}"
    elif provider == "anthropic":
        return f"anthropic/{model}"
    elif provider == "openai":
        return model
    else:
        return f"{provider}/{model}"


@lru_cache(maxsize=1)
def get_instructor_client() -> instructor.Instructor:
    """
    Get Instructor-patched client for structured LLM outputs.

    Uses JSON mode for Ollama (local models don't reliably follow tool-call format).
    Uses TOOLS mode for cloud providers (Anthropic, OpenAI) which support it natively.
    """
    provider = settings.llm_provider.lower()
    mode = instructor.Mode.MD_JSON if provider == "ollama" else instructor.Mode.TOOLS
    client = instructor.from_litellm(_litellm_completion, mode=mode)
    logger.info(
        "Instructor client initialized: provider=%s model=%s mode=%s",
        settings.llm_provider,
        settings.llm_model,
        mode,
    )
    return client


def _ollama_kwargs() -> dict:
    """Return Ollama-specific params (num_ctx) when provider is Ollama."""
    if settings.llm_provider.lower() == "ollama":
        return {"num_ctx": settings.ollama_num_ctx}
    return {}


def _api_key_kwargs() -> dict:
    """Return api_key kwarg for the configured provider (if set)."""
    provider = settings.llm_provider.lower()
    if provider == "ollama" and settings.ollama_api_key:
        return {"api_key": settings.ollama_api_key}
    if provider == "anthropic" and settings.anthropic_api_key:
        return {"api_key": settings.anthropic_api_key}
    if provider == "openai" and settings.openai_api_key:
        return {"api_key": settings.openai_api_key}
    return {}


def completion(
    prompt: str,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 2048,
    **kwargs,
) -> str:
    """Simple text completion (unstructured)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    model = _get_model_string()
    temp = temperature if temperature is not None else settings.llm_temperature

    response = _litellm_completion(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        **_api_key_kwargs(),
        **_ollama_kwargs(),
        **kwargs,
    )
    return response.choices[0].message.content


def structured_completion(
    prompt: str,
    response_model: type,
    system: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 2,
    **kwargs,
):
    """
    Structured completion — returns a validated Pydantic model.

    Instructor handles JSON schema injection, parsing, and retry on validation failure.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = get_instructor_client()
    model = _get_model_string()

    return client.chat.completions.create(
        model=model,
        response_model=response_model,
        messages=messages,
        temperature=temperature,
        max_retries=max_retries,
        **_api_key_kwargs(),
        **_ollama_kwargs(),
        **kwargs,
    )
