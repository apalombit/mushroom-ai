# 🍄 Mushroom Lookalikes Finder

> Given a mushroom name, find species that look similar — ranked by morphological,
> ecological, and taxonomic similarity with tunable weights and LLM-generated explanations.

⚠️ **Disclaimer**: This tool is for educational and research purposes only.
Never consume wild mushrooms based solely on automated identification. Always consult an expert mycologist.

---

## How It Works

```
User: "Amanita caesarea" + region: Northern Italy, autumn
                        │
    ┌───────────────────▼────────────────────┐
    │  1. Validate species name in database  │
    └───────────────────┬────────────────────┘
                        │
    ┌───────────────────▼────────────────────┐
    │  2. Retrieve feature profile +         │
    │     precomputed embeddings (pgvector)   │
    └───────────────────┬────────────────────┘
                        │
    ┌───────────────────▼────────────────────┐
    │  3. Similarity search per feature      │
    │     group (morphological, ecological,  │
    │     taxonomic) — 3 independent         │
    │     pgvector queries                   │
    └───────────────────┬────────────────────┘
                        │
    ┌───────────────────▼────────────────────┐
    │  4. Merge & rerank with tunable        │
    │     weights (default: 60/25/15)        │
    │     Apply user context as filter/boost │
    └───────────────────┬────────────────────┘
                        │
    ┌───────────────────▼────────────────────┐
    │  5. Build feature comparison table     │
    │     (query species vs top-K matches)   │
    └───────────┬───────────┬────────────────┘
                │           │
    ┌───────────▼──┐  ┌─────▼────────────────┐
    │ Structured   │  │ 6. LLM "at a glance" │
    │ table output │  │    explanation of     │
    │              │  │    why they match     │
    └───────────┬──┘  └─────┬────────────────┘
                │           │
    ┌───────────▼───────────▼────────────────┐
    │  Streamlit UI: summary + ranked list   │
    │  + expandable feature comparison       │
    └────────────────────────────────────────┘
```

## Data Architecture (Three-Layer Model)

| Layer | Purpose | Storage |
|-------|---------|---------|
| **Layer 1 — Raw Sources** | One row per species per source, exactly as extracted by LLM | PostgreSQL |
| **Layer 2 — Reconciled Profiles** | One canonical feature row per species, merged from Layer 1 | PostgreSQL |
| **Layer 3 — Embeddings** | Per-group vectors for similarity search | pgvector columns on Layer 2 |

## Tech Stack

- **Database**: PostgreSQL 16 + pgvector (structured features + vector similarity)
- **LLM**: Provider-agnostic via LiteLLM + Instructor (Ollama local → cloud later)
- **Similarity**: Per-group pgvector queries, merged with tunable weights in Python
- **API**: FastAPI
- **Frontend**: Streamlit
- **Experiment Tracking**: MLflow
- **Infrastructure**: Docker Compose (local), Terraform + EC2 (cloud — later)
- **CI/CD**: GitHub Actions

## Quick Start

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

```bash
cp .env.example .env
docker compose up -d          # Postgres + pgvector, MLflow, Ollama
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.test_setup  # verify everything connects
```

## Repository Structure

```
mushroom-ai/
├── llm/                 # Provider-agnostic LLM layer (LiteLLM + Instructor)
│   ├── client.py        # completion() and structured_completion()
│   └── schemas.py       # Pydantic models for LLM extraction + explanation
├── db/                  # Database models, connection, migrations
│   ├── models.py        # SQLAlchemy models (Layer 1, 2, 3)
│   ├── connection.py    # Engine + session factory
│   └── migrations/      # Alembic or raw SQL migrations
├── ingestion/           # Multi-source feature extraction pipeline
│   ├── extract.py       # LLM-based feature extraction from text
│   ├── reconcile.py     # Multi-source reconciliation → Layer 2
│   ├── embed.py         # Feature group embedding → Layer 3
│   ├── sources/         # Source-specific fetchers (Wikipedia, etc.)
│   └── rubric.py        # Feature rubric definition + controlled vocabularies
├── similarity/          # Similarity engine
│   ├── search.py        # Per-group pgvector queries + merge/rerank
│   ├── weights.py       # Default weights + user override logic
│   └── explain.py       # LLM explanation generation from comparison table
├── api/                 # FastAPI application
│   ├── main.py          # App factory + lifespan
│   ├── routes.py        # Endpoints
│   └── schemas.py       # API request/response models
├── ui/                  # Streamlit dashboard
│   └── app.py
├── scripts/             # Utility scripts
├── tests/               # Unit + integration tests
├── data/seed/           # Seed species list + ground truth pairs
├── docs/                # Documentation
├── infra/               # Terraform, nginx (cloud deployment — later)
└── docker-compose.yml
```

## License

MIT
