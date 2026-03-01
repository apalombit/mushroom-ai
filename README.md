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

## Ingestion Pipeline

```
"Amanita muscaria"
        │
   ┌────▼─────────────────────────────────┐
   │  FETCH                               │
   │  Wikipedia/sources → raw text        │
   └────┬─────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────┐
   │  EXTRACT (2-pass, 6 LLM calls)      │
   │  Pass 1: identity, taxonomy, body    │
   │  Pass 2: 5 parallel detail groups    │
   │  → 114 structured fields (JSON)      │
   │  Stored as source_observations (L1)  │
   └────┬─────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────┐
   │  RECONCILE                           │
   │  Multiple sources per species →      │
   │  LLM merges into one canonical row   │
   │  Stored as reconciled_species (L2)   │
   └────┬─────────────────────────────────┘
        │
   ┌────▼─────────────────────────────────┐
   │  EMBED                               │
   │  Rubric splits 114 fields into       │
   │  3 semantic groups:                  │
   │    morphological / ecological / taxon│
   │  Each → text → sentence-transformer  │
   │  → 3 × 384-dim vectors per species  │
   │  Stored on reconciled_species (L2)   │
   └──────────────────────────────────────┘
```

The 2-pass extraction groups (cap, hymenium, stem/veil, flesh/chem, spore/eco)
are purely an extraction-time split to help smaller LLMs handle the 114-field
schema. Once merged and stored, the rubric's 3-group split (morphological,
ecological, taxonomic) takes over for embedding and search.

## Data Architecture (Three-Layer Model)

| Layer | Purpose | Storage |
|-------|---------|---------|
| **Layer 1 — Raw Sources** | One row per species per source, exactly as extracted by LLM | PostgreSQL |
| **Layer 2 — Reconciled Profiles** | One canonical feature row per species, merged from Layer 1 | PostgreSQL |
| **Layer 3 — Embeddings** | Per-group vectors for similarity search | pgvector columns on Layer 2 |

## Similarity Search

At query time, the 3 embedding vectors drive lookalike discovery:

1. Look up the query species' 3 embeddings from Layer 2
2. Run 3 independent pgvector cosine similarity searches (one per group)
3. Merge candidates into a single ranked list with tunable weights
   (default: 60% morphological, 25% ecological, 15% taxonomic)
4. Filter/boost by user context (region, season) if provided
5. Return top-K with per-group score breakdown + LLM-generated explanation

Morphology dominates because lookalikes are primarily a visual confusion risk.
Ecology and taxonomy help refine: species sharing the same habitat and lineage
are more likely to be encountered together.

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
make setup
docker compose up -d          # Postgres + pgvector, MLflow, Ollama
source .venv/bin/activate
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
