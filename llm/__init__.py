"""Provider-agnostic LLM layer (LiteLLM + Instructor)."""

from llm.client import completion, get_instructor_client, structured_completion

__all__ = ["completion", "structured_completion", "get_instructor_client"]
