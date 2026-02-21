"""
LLM health check — verifies the configured provider is reachable and responding.

Run manually:
    pytest tests/integration/test_llm_health.py -v -s

The -s flag is required to see the printed prompts and responses.
"""

import pprint

import pytest
from pydantic import BaseModel, Field

from config import settings
from llm.client import completion, structured_completion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _section(label: str, content: str) -> None:
    print(f"\n  [{label}]")
    print(f"  {content}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_llm_provider_config():
    """Print the active LLM configuration so it's visible in test output."""
    _banner("LLM Configuration")
    _section("Provider", settings.llm_provider)
    _section("Model", settings.llm_model)
    _section("Base URL", settings.llm_base_url or "(default)")
    _section("API key set", "yes" if settings.ollama_api_key or settings.anthropic_api_key or settings.openai_api_key else "no")
    print()

    assert settings.llm_provider, "LLM_PROVIDER must be set in .env"
    assert settings.llm_model, "LLM_MODEL must be set in .env"


def test_plain_completion():
    """
    Send a minimal plain-text prompt and print the response.
    Verifies basic connectivity + that the model produces any output.
    """
    prompt = "Reply with exactly one word: hello"
    system = "You are a concise assistant. Follow instructions precisely."

    _banner("Plain Completion Health Check")
    _section("System", system)
    _section("Prompt", prompt)

    response = completion(prompt=prompt, system=system, max_tokens=32)

    _section("Response", repr(response))
    print()

    assert isinstance(response, str), "completion() must return a string"
    assert len(response.strip()) > 0, "Response must not be empty"
    print(f"  ✓ Got {len(response)} chars back from {settings.llm_provider}/{settings.llm_model}")


def test_structured_completion():
    """
    Send a minimal structured prompt and verify Instructor can parse the response.
    Uses a trivial two-field schema to avoid stressing the model.
    """

    class PingResponse(BaseModel):
        status: str = Field(description="Always the string 'ok'")
        model_name: str = Field(description="The name of the AI model responding")

    prompt = (
        'Respond with a JSON object containing exactly two fields: '
        '"status" (always the string "ok") and '
        '"model_name" (your model name or "unknown" if unsure).'
    )
    system = "You are a health-check endpoint. Return only the requested JSON."

    _banner("Structured Completion Health Check")
    _section("Schema", "PingResponse(status: str, model_name: str)")
    _section("System", system)
    _section("Prompt", prompt)

    result = structured_completion(
        prompt=prompt,
        response_model=PingResponse,
        system=system,
        temperature=0.0,
        max_retries=2,
    )

    _section("Parsed response", pprint.pformat(result.model_dump()))
    print()

    assert isinstance(result, PingResponse), "structured_completion() must return PingResponse"
    assert result.status == "ok", f"Expected status='ok', got {result.status!r}"
    assert isinstance(result.model_name, str) and len(result.model_name) > 0
    print(f"  ✓ Structured output OK — model identified as: {result.model_name!r}")
