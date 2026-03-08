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
    grouping_profile: str = "default"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # Similarity weights
    weight_numeric: float = 0.10
    weight_body_form_filter: bool = False
    # Per-group embedding weights (None = use equal-split default)
    weight_global: float | None = None
    weight_cap_shape: float | None = None
    weight_cap_top: float | None = None
    weight_cap_margin: float | None = None
    weight_cap_viz: float | None = None
    weight_hymenium: float | None = None
    weight_gills_arrange: float | None = None
    weight_gills_distrib: float | None = None
    weight_gills_viz: float | None = None
    weight_pores: float | None = None
    weight_stem: float | None = None
    weight_stem_surf: float | None = None
    weight_stem_viz: float | None = None
    weight_veil: float | None = None
    weight_volva: float | None = None
    weight_flesh_visual: float | None = None
    weight_flesh_inner: float | None = None
    weight_uniformity: float | None = None
    weight_latex: float | None = None
    weight_flesh_perceptive: float | None = None
    weight_spore_vis: float | None = None
    weight_spore: float | None = None
    weight_microscopic: float | None = None
    weight_chemical: float | None = None
    weight_ecological: float | None = None
    weight_taxonomic: float | None = None

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
