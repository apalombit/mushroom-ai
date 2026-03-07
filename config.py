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
    ollama_num_ctx: int = 32_768

    # API keys — only needed for cloud providers
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    ollama_api_key: str = ""

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "mushroom_ai"
    postgres_user: str = "mushroom"
    postgres_password: str = "mushroom_dev"

    # Embeddings
    embedding_model: str = "all-mpnet-base-v2"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Similarity weights (7 groups + body-form filter)
    weight_macro_visual: float = 0.69
    weight_structural: float = 0.12
    weight_flesh_sensory: float = 0.07
    weight_microscopic_lab: float = 0.02
    weight_ecological: float = 0.01
    weight_taxonomic: float = 0.01
    weight_numeric: float = 0.10
    weight_body_form_filter: bool = False

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
