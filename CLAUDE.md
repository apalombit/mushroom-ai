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

# Evaluation
python -m scripts.eval                          # single eval with default weights
python -m scripts.eval --benchmark              # sweep named configs from BENCHMARK_CONFIGS
python -m scripts.eval --sweep 5000 --top-k 5   # joint sweep: emb weights × alpha × gates
python -m scripts.optimize_jaccard --trials 100  # Optuna optimization of Jaccard field weights

# Ingestion (flags are strictly independent — no auto-chaining)
python -m scripts.ingest --fetch-only           # fetch raw HTML/text from sources
python -m scripts.ingest --extract              # LLM feature extraction → Layer 1
python -m scripts.ingest --reconcile            # merge sources → Layer 2
python -m scripts.ingest --embed                # generate embeddings → Layer 3
```

CI runs `ruff check`, `ruff format --check`, and `pytest tests/unit/` only (no external services).

## Architecture

### Directory Layout

```
api/              FastAPI routes + schemas
db/               SQLAlchemy models + connection (get_session/engine)
data/reference/   morphological_vocabulary.yaml, feature_usage.yaml (feature→method cross-reference)
data/seed/        species_list.yaml, known_lookalikes.yaml (merged GT + extracted lookalikes)
evaluation/       Recall metrics (recall.py), weight sweep optimizer (sweep.py), MLflow logging
ingestion/        5-stage pipeline: fetch → extract → reconcile → normalize → embed
  profiles/       Grouping profiles (default.yaml, finer_grps.yaml) defining EMBEDDING_GROUPS
  sources/        Per-source fetchers (mushroomexpert, firstnature, ultimatemushroom, funghiitaliani)
  rubric.py       Schema contract: EMBEDDING_GROUPS, NUMERIC_FIELDS, features_to_text()
  normalize.py    Alias resolution, canonical enforcement, load_vocabulary()
  embed.py        sentence-transformers embeddings per group
llm/              LiteLLM + Instructor wrapper (client.py), Pydantic schemas (schemas.py)
similarity/       Search engine core
  search.py       Main entry: search_lookalikes(), search_by_group(), build_comparison_table()
  aggregation.py  Score aggregation strategies (weighted_avg, rrf, contrastive_gate, blended)
  jaccard.py      Soft-Jaccard scorer using vocabulary similarity matrices
  morphotype.py   Coarse morphotype signature matching (body_form|hymenium|size|cap_shape|gills)
  numeric.py      Numeric field comparison (range categories, single proximity, month overlap)
  weights.py      SimilarityWeights class (per-group weights, filters, alpha)
  explain.py      LLM-generated lookalike explanations
scripts/          CLI entry points (ingest.py, eval.py, init_db.py, optimize_jaccard.py)
ui/               Streamlit front-end
tests/            unit/ (no services needed) + integration/ (requires Docker)
```

### Three-Layer Data Model

All species data flows through three layers in PostgreSQL + pgvector:

| Layer | Table | What it holds |
|-------|-------|---------------|
| 1 | `source_observations` | One row per species per source — raw LLM extraction, never mutated |
| 2 | `reconciled_species` | One canonical row per species — merged from Layer 1, stores `features_json` |
| 3 | pgvector columns on Layer 2 | One 768-dim vector per embedding group (`embedding_<group>`) + gating columns |

Gating columns on Layer 2: `overall_body_form`, `hymenium_type`, `overall_size_class`, `morphotype_signature` — populated during embedding for fast pre-filtering.

### Feature Groups and the Rubric

`ingestion/rubric.py` is the system's schema contract. It loads the active grouping profile (`GROUPING_PROFILE` in `.env`) which defines `EMBEDDING_GROUPS` — a dict mapping group names to their constituent feature fields. The active profile (e.g., `finer_grps`) determines how many embedding columns exist and how similarity is decomposed. The `features_to_text()` function converts structured features into text for embedding. Every component must stay aligned with this rubric.

### Query Pipeline (similarity/search.py)

1. Look up query species from `reconciled_species`
2. Run N independent pgvector cosine similarity queries (one per embedding group)
3. **Morpho pool priority**: candidates appearing in morphological groups get priority over eco/taxon-only hits
4. **Optional hard gates**: body_form_filter, hymenium_filter, size_class_filter (default: off — gates hurt recall)
5. **Optional morphotype pre-filter**: filter by morphotype signature match score
6. Compute **numeric similarity** from structured range/single/month fields
7. **Aggregate** per-group scores using selected strategy (weighted_avg | rrf | contrastive_gate)
8. **Blend** with Jaccard similarity: `alpha * embedding_score + (1-alpha) * jaccard_score`
9. Apply region/season context boosts
10. Return top-K with per-group breakdown + optional LLM explanation

### Similarity Scoring Methods

**Embedding similarity** — Per-group cosine similarity from pgvector, aggregated via weights in `SimilarityWeights`. Weights are normalized to sum to 1.0. Optimized via `scripts/eval.py --sweep`.

**Soft-Jaccard similarity** (`similarity/jaccard.py`) — Compares discrete feature values using similarity matrices from `data/reference/morphological_vocabulary.yaml`. Three field types: vocabulary scalars (via matrices), color fields (bidirectional best-match), boolean fields (exact match). Per-field weights optimizable via `scripts/optimize_jaccard.py`. `DEFAULT_JACCARD_WEIGHTS` in jaccard.py stores optimized weights.

**Blending** — `alpha` parameter (0.0–1.0) controls embedding vs Jaccard mix. `alpha=1.0` = pure embedding, `alpha=0.0` = pure Jaccard. Default: 0.5. Tunable via `.env` or request params.

**Numeric similarity** (`similarity/numeric.py`) — Range fields → size category (small/medium/large), single fields → proximity scoring, month fields → Jaccard of month sets.

**Feature coverage**: `data/reference/feature_usage.yaml` is the source of truth for which features are used by each scoring method. Most morphological features are shared between embedding and Jaccard, but each method has exclusives (e.g., ecology/taxonomy are embedding-only; gills.edge_texture, gills.texture, flesh.quantity are Jaccard-only).

### Evaluation & Optimization

**Ground truth**: `known_lookalikes.yaml` (~847 indexed edges, ~416 species). Loaded via `load_unified_lookalikes()` (filtered to DB species) or `load_lookalike_graph()` (raw edges).

**Evaluation** (`evaluation/recall.py`): Bidirectional Recall@1/3/5 — for each pair (A,B), check if B in top-K of A's results and vice versa. `BENCHMARK_CONFIGS` stores named weight presets for comparison.

**Weight sweep** (`evaluation/sweep.py`): Precompute-then-rescore pattern. `precompute_scores()` runs all DB queries once (embedding searches + Jaccard per candidate). `rescore()` is pure arithmetic — called 200K+ times per sweep. Jointly optimizes: embedding group weights × alpha × gate filter combinations.

**Jaccard optimization** (`scripts/optimize_jaccard.py`): Optuna TPE sampler over 52 per-field weights. Slow (~90s/trial) because it computes Jaccard of each edge species against all 498 species per trial.

### LLM Layer (llm/)

Provider-agnostic via LiteLLM + Instructor. Switch provider/model with `LLM_PROVIDER` and `LLM_MODEL` in `.env` — no code changes. Use `structured_completion()` (returns a validated Pydantic model) over `completion()` (plain text) whenever the output needs structure. Never import litellm/instructor directly.

### Config

`config.py` exposes a singleton `settings` object (pydantic-settings, reads `.env`). Import it as `from config import settings`. Key settings: `GROUPING_PROFILE` (embedding group layout), `weight_*` (per-group weights, filter flags, alpha), `LLM_*` (provider config), database connection params.

## Code Style

- **Minimalistic**: no extra abstractions, no defensive code for impossible cases, no unused helpers.
- **Every feature tested**: every implemented function must have a corresponding test in `tests/unit/` or `tests/integration/`. Stub/`NotImplementedError` functions are exempt until implemented.
- Line length: 99. Python 3.11+. Ruff rules: E, F, I, W, UP.
- `pytest-asyncio` with `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio`.
- Shared test fixtures live in `tests/conftest.py`.
- All LLM calls via `llm/client.py` only — never import litellm/instructor directly.
- All DB access via `db/connection.py` `get_session()` / `engine` — never create raw connections.
- Ingest flags are strictly independent — no auto-chaining between stages.

## Key Dependencies

- **LLM**: litellm + instructor (provider-agnostic structured output)
- **Embeddings**: sentence-transformers (all-mpnet-base-v2, 768-dim)
- **DB**: sqlalchemy 2.0+, psycopg2-binary, pgvector (IVFFlat indexes)
- **API/UI**: fastapi, uvicorn, streamlit
- **Optimization**: optuna (Jaccard weight search), numpy (Dirichlet sampling for sweeps)
- **Tracking**: mlflow
