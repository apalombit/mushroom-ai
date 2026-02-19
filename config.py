"""Shared configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM — provider-agnostic via LiteLLM
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_base_url: str | None = "http://localhost:11434"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # API keys — only needed for cloud providers
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "mushroom_ai"
    postgres_user: str = "mushroom"
    postgres_password: str = "mushroom_dev"

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Similarity weights
    weight_morphological: float = 0.60
    weight_ecological: float = 0.25
    weight_taxonomic: float = 0.15

    # App
    log_level: str = "INFO"
    environment: str = "development"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
