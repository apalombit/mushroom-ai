"""Unit tests for vision support in llm/client.py."""

import base64
import struct
import zlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm.client import _encode_image_to_data_url, vision_completion


def _write_tiny_png(path: Path) -> None:
    """Write a 1x1 transparent PNG (no external deps)."""
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)  # 1x1, 8-bit RGBA
    raw = b"\x00\x00\x00\x00\x00"  # filter byte + RGBA pixel
    idat = zlib.compress(raw)
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def test_encode_image_to_data_url_png(tmp_path):
    """PNG file is base64-encoded into a valid data URL with correct prefix."""
    img = tmp_path / "tiny.png"
    _write_tiny_png(img)

    url = _encode_image_to_data_url(img)

    assert url.startswith("data:image/png;base64,")
    payload = url.split(",", 1)[1]
    decoded = base64.b64decode(payload)
    assert decoded == img.read_bytes()
    assert decoded.startswith(b"\x89PNG")


def test_encode_image_to_data_url_jpeg_extension(tmp_path):
    """A .jpg extension maps to image/jpeg MIME."""
    img = tmp_path / "fake.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    url = _encode_image_to_data_url(img)
    assert url.startswith("data:image/jpeg;base64,")


@patch("llm.client._litellm_completion")
@patch("llm.client.settings")
def test_vision_completion_builds_multimodal_messages(mock_settings, mock_completion, tmp_path):
    """vision_completion should build a one-text + N-image content payload."""
    mock_settings.llm_provider = "ollama"
    mock_settings.llm_model = "gemma4:31b-cloud"
    mock_settings.llm_base_url = "http://localhost:11434"
    mock_settings.llm_temperature = 0.2
    mock_settings.ollama_num_ctx = 32768
    mock_settings.ollama_api_key = "test-key"
    mock_settings.anthropic_api_key = ""
    mock_settings.openai_api_key = ""

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "a red mushroom"
    mock_completion.return_value = mock_response

    img1 = tmp_path / "a.png"
    img2 = tmp_path / "b.png"
    _write_tiny_png(img1)
    _write_tiny_png(img2)

    result = vision_completion(
        "What do you see?",
        image_paths=[img1, img2],
        system="You are a mycologist.",
    )

    assert result == "a red mushroom"
    kwargs = mock_completion.call_args.kwargs

    assert kwargs["model"] == "ollama_chat/gemma4:31b-cloud"
    assert kwargs["num_ctx"] == 32768
    assert kwargs["api_key"] == "test-key"

    messages = kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are a mycologist."}
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    content = user_msg["content"]
    assert content[0] == {"type": "text", "text": "What do you see?"}
    assert len(content) == 3  # 1 text + 2 images
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2]["type"] == "image_url"
    assert content[2]["image_url"]["url"].startswith("data:image/png;base64,")


@patch("llm.client._litellm_completion")
@patch("llm.client.settings")
def test_vision_completion_model_override(mock_settings, mock_completion, tmp_path):
    """Per-call model override should rewrite the model string with provider prefix."""
    mock_settings.llm_provider = "ollama"
    mock_settings.llm_model = "llama3.1:8b"  # default text model
    mock_settings.llm_base_url = ""
    mock_settings.llm_temperature = 0.1
    mock_settings.ollama_num_ctx = 16384
    mock_settings.ollama_api_key = ""
    mock_settings.anthropic_api_key = ""
    mock_settings.openai_api_key = ""

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "ok"
    mock_completion.return_value = mock_response

    img = tmp_path / "a.png"
    _write_tiny_png(img)

    vision_completion("describe", image_paths=[img], model="gemma4:31b-cloud")

    kwargs = mock_completion.call_args.kwargs
    assert kwargs["model"] == "ollama_chat/gemma4:31b-cloud"
