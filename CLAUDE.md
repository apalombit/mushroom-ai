# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment
cp .env.example .env
make setup          # create .venv and install requirements.txt
make up             # docker compose up -d (Postgres+pgvector, MLflow, Ollama)
make test-setup     # smoke test: verify all services connect

# Database
make db-init        # create tables (python -m scripts.init_db)

# Development
make serve          # uvicorn api.main:app --reload --port 8001
make ui             # streamlit run ui/app.py --server.port 8501

# Quality
make lint           # ruff check + ruff format --check
make format         # ruff check --fix + ruff format
make test           # pytest -v (all tests)
pytest tests/unit/ --tb=short -q   # unit tests only (no Docker needed)
pytest tests/integration/          # integration tests (requires Docker services)
pytest tests/unit/test_foo.py -k "test_name"  # single test
```

CI runs `ruff check`, `ruff format --check`, and `pytest tests/unit/` only (no external services).

## Architecture

### Three-Layer Data Model

All species data flows through three layers, all in PostgreSQL:

| Layer | Table | What it holds |
|-------|-------|---------------|
| 1 | `source_observations` | One row per species per source — raw LLM extraction, never mutated |
| 2 | `reconciled_species` | One canonical row per species — merged from Layer 1 |
| 3 | pgvector columns on Layer 2 | Three 384-dim vectors per species (`embedding_morphological`, `embedding_ecological`, `embedding_taxonomic`) |

### Feature Groups and the Rubric

`ingestion/rubric.py` is the system's schema contract. It defines which fields belong to each of the three similarity groups (`MORPHOLOGICAL_FIELDS`, `ECOLOGICAL_FIELDS`, `TAXONOMIC_FIELDS`) and the `features_to_text()` bridge that converts structured features into text for embedding. Every component (extraction schema, reconciliation, embedding, similarity) must stay aligned with this rubric.

### Query Pipeline (similarity/)

1. Look up query species embeddings from `reconciled_species`
2. Run three independent pgvector cosine similarity queries (one per group)
3. Merge candidates, compute weighted score — default weights: morph 60%, eco 25%, taxon 15% (tunable via `.env` or request params)
4. Apply user context (region, season) as filter/boost
5. Return top-K with per-group breakdown + LLM-generated explanation

### LLM Layer (llm/)

Provider-agnostic via LiteLLM + Instructor. Switch provider/model with `LLM_PROVIDER` and `LLM_MODEL` in `.env` — no code changes. Use `structured_completion()` (returns a validated Pydantic model) over `completion()` (plain text) whenever the output needs structure.

### Schemas

Two separate schema namespaces:
- `llm/schemas.py` — LLM input/output contracts (`ExtractedSpeciesFeatures`, `ReconciliationResult`, `LookalikeExplanation`). Instructor validates against these.
- `api/schemas.py` — FastAPI request/response models.

### Config

`config.py` exposes a singleton `settings` object (pydantic-settings, reads `.env`). Import it as `from config import settings`. The `settings.database_url` property builds the SQLAlchemy URL.

## Code Style

- **Minimalistic**: no extra abstractions, no defensive code for impossible cases, no unused helpers.
- **Every feature tested**: every implemented function must have a corresponding test in `tests/unit/` or `tests/integration/`. Stub/`NotImplementedError` functions are exempt until implemented.
- Line length: 99. Python 3.11+. Ruff rules: E, F, I, W, UP.
- `pytest-asyncio` with `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio`.
- Shared test fixtures live in `tests/conftest.py`.
