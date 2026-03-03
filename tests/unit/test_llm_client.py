"""Unit tests for llm/client.py — Ollama num_ctx injection."""

from unittest.mock import MagicMock, patch

from llm.client import completion


@patch("llm.client._litellm_completion")
@patch("llm.client.settings")
def test_ollama_injects_num_ctx(mock_settings, mock_completion):
    """When provider is ollama, num_ctx should be passed to litellm."""
    mock_settings.llm_provider = "ollama"
    mock_settings.llm_model = "gemma3:27b"
    mock_settings.llm_base_url = "http://localhost:11434"
    mock_settings.llm_temperature = 0.2
    mock_settings.ollama_num_ctx = 32768
    mock_settings.ollama_api_key = ""
    mock_settings.anthropic_api_key = ""
    mock_settings.openai_api_key = ""

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "hello"
    mock_completion.return_value = mock_response

    result = completion("test prompt")

    assert result == "hello"
    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["num_ctx"] == 32768


@patch("llm.client._litellm_completion")
@patch("llm.client.settings")
def test_anthropic_omits_num_ctx(mock_settings, mock_completion):
    """When provider is anthropic, num_ctx should NOT be passed."""
    mock_settings.llm_provider = "anthropic"
    mock_settings.llm_model = "claude-sonnet-4-20250514"
    mock_settings.llm_base_url = ""
    mock_settings.llm_temperature = 0.2
    mock_settings.anthropic_api_key = "sk-ant-test"
    mock_settings.ollama_api_key = ""
    mock_settings.openai_api_key = ""

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "hello"
    mock_completion.return_value = mock_response

    result = completion("test prompt")

    assert result == "hello"
    call_kwargs = mock_completion.call_args
    assert "num_ctx" not in call_kwargs.kwargs
