# 🍄 Mushroom Lookalikes Finder

> Given a mushroom name, find species that look similar — ranked by morphological,
> ecological, and taxonomic similarity with tunable weights and LLM-generated explanations.

⚠️ **Disclaimer**: This tool is for educational and research purposes only.
Never consume wild mushrooms based solely on automated identification. Always consult an expert mycologist.

---

## Project Journey

The project grows along two tracks that meet at one goal: **snap a photo → get its dangerous lookalikes**.

| Track | What it does | Status |
|-------|--------------|--------|
| 🔤 **Text engine** | Name a species → ranked lookalikes with explanations | ✅ Usable end-to-end |
| 📷 **Photo reader** | Photo → the same morphological checklist the engine uses | 🚧 First classifiers trained |
| 🔗 **Integration** | Photo in → lookalikes out | 🔜 Next milestone |

### 🔤 Track 1 — Lookalike engine from text

Mushroom descriptions are turned into a structured checklist (cap, gills, stem, colours, habitat…),
then species are compared trait by trait.

| # | Stage | What changed | Result |
|---|-------|--------------|--------|
| 1 | **Foundations** | Postgres + pgvector, FastAPI, Streamlit; LLM reads a species page and fills a feature profile | First working search by name |
| 2 | **Better reading** | 2-pass LLM extraction into 114 structured fields; multiple sources merged per species (Wikipedia, MushroomExpert, First Nature, Ultimate Mushroom, Funghi Italiani); aliases normalised to a controlled vocabulary | Clean, comparable profiles |
| 3 | **Measuring it** | Ground truth of ~850 known lookalike pairs; bidirectional Recall@1/3/5; MLflow tracking | Every change is scored, not eyeballed |
| 4 | **Sharper comparison** | 3 coarse embedding groups split into finer per-area groups; soft-Jaccard on discrete traits via vocabulary similarity matrices; numeric + season matching; morphotype pre-filter; blended embedding/Jaccard score tuned by weight sweeps and Optuna | Trait-level similarity with tuned weights |
| 5 | **Learned ranking** | LightGBM re-ranker over 61 per-field similarity signals, with model fingerprinting to catch stale models | Ranking learned from known lookalike pairs |
| 6 | **User-facing polish** | "Dangerous lookalike" mode (opposite edibility), LLM explanations, reference links and images, Italian UI | ~500 species, in English and Italian |

### 📷 Track 2 — Reading features from photos

Goal: look at a photo and fill in the same checklist, so an unknown mushroom can be searched.

| # | Stage | What changed | Result |
|---|-------|--------------|--------|
| 1 | **Image pool** | ~14K photos from citizen-science sources (iNaturalist, Mushroom Observer), quality-checked, view-angle tagged, near-duplicates removed | Clean training set |
| 2 | **First classifiers** | Frozen DINOv2 image embeddings + small per-feature heads, trained on labels inherited from each photo's species | Structure learned well — hymenium **90.5%**, body form **85.6%**, cap shape **74.8%**; colour weak at **43.8%** |
| 3 | **A vision-language model as teacher** | Gemma 3 labels features *photo by photo*; 8 features tested with single-shot, few-shot and two-stage prompting | 3 trusted teachers (hymenium, cap colour, volva); 5 shelved |
| 4 | **Retraining on per-photo labels** | First head (volva) retrained on ~5.3K teacher labels instead of species-inherited ones | **82.7%** accuracy, macro-F1 **0.753** — best class balance of any head so far |
| 5 | **Scaling up** | Same recipe for cap colour and hymenium (colour has the most to gain) | ⏸ Larger labelling run deferred until compute is available |

**What the shelved features taught us**

- **A species fact is not always visible in a photo.** A species may "have a ring", but on this
  specimen it fell off or is hidden. Labels inherited from the species then punish a model that is
  correctly cautious. Features like this need a per-photo visibility check.
- **Some details sit below the visible floor.** Fine surface textures (silky, velvety, powdery)
  cannot be resolved at the model's 896×896 input. They would need higher resolution or close-up
  crops.

### 🔗 Next — photo in, lookalikes out

- [ ] Predict checklist features from uploaded photo(s), with confidence
- [ ] Feed confidence-weighted features into the existing similarity search
- [ ] Show ranked lookalikes with a visual + text comparison in the Streamlit UI

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
