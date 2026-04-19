"""
Provider-agnostic LLM client using LiteLLM + Instructor.

Usage:
    from llm.client import structured_completion, completion

    # Structured output (preferred — validated Pydantic model)
    result = structured_completion("...", response_model=MySchema, system="...")

    # Plain text
    text = completion("...")

    # Vision (multi-modal) — same surface, plus image_paths
    text = vision_completion("describe", image_paths=["a.jpg"])
    obj = structured_vision_completion("...", image_paths=["a.jpg"], response_model=MySchema)

Provider switching: change LLM_PROVIDER and LLM_MODEL in .env — no code changes.
"""

import base64
import logging
from functools import lru_cache
from pathlib import Path

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


_MIME_BY_EXT = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".gif": "gif",
    ".bmp": "bmp",
}


def _encode_image_to_data_url(path: str | Path) -> str:
    """Read an image file and return a base64-encoded data URL.

    MIME type is inferred from the file extension. Used to embed images
    inline in multi-modal LiteLLM messages payloads.
    """
    p = Path(path)
    mime = _MIME_BY_EXT.get(p.suffix.lower(), "jpeg")
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def _build_vision_messages(
    prompt: str,
    image_paths: list[str | Path],
    system: str | None,
) -> list[dict]:
    """Build a multi-modal `messages` payload (one text block + N image blocks)."""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _encode_image_to_data_url(img)},
            }
        )

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def vision_completion(
    prompt: str,
    image_paths: list[str | Path],
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 2048,
    model: str | None = None,
    **kwargs,
) -> str:
    """Multi-modal text completion: prompt + one or more images.

    `model` overrides the configured LLM_MODEL for this call only — useful when
    the default text model isn't a vision model. Provider routing is identical
    to `completion()`.
    """
    messages = _build_vision_messages(prompt, image_paths, system)
    model_str = _get_model_string() if model is None else _override_model_string(model)
    temp = temperature if temperature is not None else settings.llm_temperature

    response = _litellm_completion(
        model=model_str,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        **_api_key_kwargs(),
        **_ollama_kwargs(),
        **kwargs,
    )
    return response.choices[0].message.content


def structured_vision_completion(
    prompt: str,
    image_paths: list[str | Path],
    response_model: type,
    system: str | None = None,
    temperature: float = 0.1,
    max_retries: int = 2,
    model: str | None = None,
    **kwargs,
):
    """Structured multi-modal completion — returns a validated Pydantic model.

    Reuses `get_instructor_client()` so Ollama uses MD_JSON mode and cloud
    providers use TOOLS mode automatically.
    """
    messages = _build_vision_messages(prompt, image_paths, system)
    client = get_instructor_client()
    model_str = _get_model_string() if model is None else _override_model_string(model)

    return client.chat.completions.create(
        model=model_str,
        response_model=response_model,
        messages=messages,
        temperature=temperature,
        max_retries=max_retries,
        **_api_key_kwargs(),
        **_ollama_kwargs(),
        **kwargs,
    )


def _override_model_string(model: str) -> str:
    """Build a LiteLLM model string for an ad-hoc model name (per-call override).

    Uses the configured provider's routing prefix.
    """
    provider = settings.llm_provider.lower()
    if provider == "ollama":
        return f"ollama_chat/{model}"
    elif provider == "anthropic":
        return f"anthropic/{model}"
    elif provider == "openai":
        return model
    else:
        return f"{provider}/{model}"
