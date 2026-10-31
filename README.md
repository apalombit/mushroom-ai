# 🍄 Mushroom Lookalikes Finder

> Given a mushroom, find the species it can be confused with — ranked trait by trait across
> morphology, ecology and taxonomy, with LLM-generated explanations. Photo input is in progress.

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
| 2 | **Better reading** | 2-pass LLM extraction into 100+ structured fields; multiple sources merged per species (Wikipedia, MushroomExpert, First Nature, Ultimate Mushroom, Funghi Italiani); aliases normalised to a controlled vocabulary | Clean, comparable profiles |
| 3 | **Measuring it** | Ground truth of ~850 known lookalike pairs; bidirectional Recall@1/3/5; MLflow tracking | Every change is scored, not eyeballed |
| 4 | **Sharper comparison** | 3 coarse embedding groups split into 46 fine-grained ones; soft-Jaccard on discrete traits via vocabulary similarity matrices; numeric + season matching; morphotype pre-filter; blended embedding/Jaccard score tuned by weight sweeps and Optuna | Trait-level similarity with tuned weights |
| 5 | **Learned ranking** | LightGBM re-ranker over 61 per-field similarity signals, with model fingerprinting to catch stale models | Ranking learned from known lookalike pairs |
| 6 | **User-facing polish** | "Dangerous lookalike" mode (opposite edibility), LLM explanations, reference links and images, Italian UI | ~500 species, in English and Italian |

### 📷 Track 2 — Reading features from photos

Goal: look at a photo and fill in the same checklist, so an unknown mushroom can be searched.

| # | Stage | What changed | Result |
|---|-------|--------------|--------|
| 1 | **Image pool** | ~14K photos from citizen-science sources (iNaturalist, Mushroom Observer), quality-checked, view-angle tagged, near-duplicates removed | Clean training set |
| 2 | **First classifiers** | Frozen DINOv2 image embeddings + small per-feature heads, trained on labels inherited from each photo's species | Structure learned well — hymenium **90.5%**, body form **85.6%**, cap shape **74.8%**; colour weak at **43.8%** |
| 3 | **A vision-language model as teacher** | Gemma 4 labels features *photo by photo*; 8 features tested with single-shot, few-shot and two-stage prompting | 3 trusted teachers (hymenium, cap colour, volva); 5 shelved |
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
User: "Amanita caesarea"  (+ optional region / season, dangerous-only mode)
                        │
    ┌───────────────────▼──────────────────────┐
    │ 1. Look up the species' feature profile  │
    │    + per-group embeddings (pgvector)     │
    └───────────────────┬──────────────────────┘
                        │
    ┌───────────────────▼──────────────────────┐
    │ 2. Retrieve candidates: one cosine query │
    │    per embedding group (46 groups: cap   │
    │    colour, gill attachment, ring, …)     │
    │    Morphological hits take priority      │
    └───────────────────┬──────────────────────┘
                        │
    ┌───────────────────▼──────────────────────┐
    │ 3. Optional gates: body form, hymenium,  │
    │    size class, morphotype, edibility     │
    └───────────────────┬──────────────────────┘
                        │
    ┌───────────────────▼──────────────────────┐
    │ 4. Score each candidate                  │
    │    • weighted per-group cosine + numeric │
    │    • soft-Jaccard on discrete traits     │
    │    blend: α·embedding + (1−α)·Jaccard    │
    │    — or the LightGBM learned ranker      │
    └───────────────────┬──────────────────────┘
                        │
    ┌───────────────────▼──────────────────────┐
    │ 5. Region / season boosts → top-K        │
    └────────────┬─────────────────┬───────────┘
                 │                 │
    ┌────────────▼─────┐   ┌───────▼───────────┐
    │ Feature          │   │ LLM "at a glance" │
    │ comparison table │   │ explanation       │
    └────────────┬─────┘   └───────┬───────────┘
                 │                 │
    ┌────────────▼─────────────────▼───────────┐
    │ Streamlit UI (EN / IT) or FastAPI JSON   │
    └──────────────────────────────────────────┘
```

## Ingestion Pipeline

Each stage is a separate flag on `python -m scripts.ingest` and never triggers the next one.

```
"Amanita muscaria"
        │
   ┌────▼──────────────────────────────────────┐
   │ FETCH               --fetch-only          │
   │ Wikipedia, MushroomExpert, First Nature,  │
   │ Ultimate Mushroom, Funghi Italiani        │
   │ → raw text + image links                  │
   └────┬──────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────┐
   │ EXTRACT             --extract             │
   │ Pass 1: identity, taxonomy, body plan     │
   │ Pass 2: 5 parallel detail groups          │
   │ → 100+ structured fields per source       │
   │ → source_observations (Layer 1)           │
   └────┬──────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────┐
   │ RECONCILE           --reconcile           │
   │ Normalise aliases to controlled vocab,    │
   │ LLM merges all sources into one profile   │
   │ → reconciled_species (Layer 2)            │
   └────┬──────────────────────────────────────┘
        │
   ┌────▼──────────────────────────────────────┐
   │ EMBED               --embed               │
   │ Grouping profile splits fields into       │
   │ groups (finer_grps: 46)                   │
   │ group → text → all-mpnet-base-v2 (768-d)  │
   │ + gating columns (body form, hymenium…)   │
   │ → pgvector columns (Layer 3)              │
   └───────────────────────────────────────────┘
```

The extraction groups (cap, hymenium, stem/veil, flesh/chem, spore/eco) only exist to help
smaller LLMs handle the large schema. After reconciliation, the active grouping profile in
`ingestion/profiles/` (set with `GROUPING_PROFILE`) decides how fields are embedded and compared.

## Data Architecture (Three-Layer Model)

| Layer | Purpose | Storage |
|-------|---------|---------|
| **Layer 1 — Raw Sources** | One row per species per source, exactly as extracted by the LLM | PostgreSQL |
| **Layer 2 — Reconciled Profiles** | One canonical feature row per species, merged from Layer 1 | PostgreSQL |
| **Layer 3 — Embeddings** | One 768-dim vector per embedding group + gating columns | pgvector columns on Layer 2 |

## Similarity Search

Lookalikes are scored with three complementary signals:

| Signal | What it compares | Where |
|--------|------------------|-------|
| **Embedding** | Cosine similarity per embedding group, combined with tuned per-group weights | `similarity/search.py`, `weights.py` |
| **Soft-Jaccard** | Discrete trait values, with partial credit from vocabulary similarity matrices (e.g. *gills* vs *ridges* = 0.7, *gills* vs *pores* = 0.1) | `similarity/jaccard.py` |
| **Numeric** | Size ranges, single measurements, fruiting-month overlap | `similarity/numeric.py` |

How the signals are combined:

1. Per-group embedding scores and the numeric score are aggregated by **weighted average**
   (API default), **RRF** or a **contrastive gate**.
2. The result is blended with Jaccard: `α · embedding + (1 − α) · Jaccard`. α defaults to 0.5
   and can be set per request.
3. Alternatively, the **learned ranker** replaces both steps. It is a LightGBM model trained on
   the known lookalike pairs over 61 per-field similarity features, and it is the default in
   the UI.

Optional gates (body form, hymenium, size class, morphotype) are off by default because they lower
recall. The **dangerous mode** keeps only candidates whose edibility is opposite to the query's,
for example edible vs toxic. Region and season act as boosts, not filters.

Morphology dominates because lookalikes are primarily a visual confusion risk. Ecology and
taxonomy refine the ranking, since species sharing habitat and lineage are more likely to be met
together.

## Evaluation

- **Ground truth**: ~850 known lookalike pairs across ~416 species (`data/seed/known_lookalikes.yaml`)
- **Metric**: bidirectional Recall@1/3/5. For each pair (A, B), is B in A's top-K, and A in B's?
- **Tuning**:
  - `scripts.eval --sweep`: joint sweep over group weights × α × gates
  - `scripts.optimize_jaccard`: Optuna over per-field Jaccard weights
  - `scripts.train_ranker`: trains the LightGBM ranker
- All runs are logged to MLflow.

## Vision Module (in progress)

`vision/` reads morphological features from photos: frozen DINOv2 embeddings feed small
per-feature classification heads. Labels are either inherited from the species profile or
produced photo by photo by a vision-language model teacher. The module is not yet connected to
search. See [vision/VISION_PROJECT_PLAN.md](vision/VISION_PROJECT_PLAN.md).

## Tech Stack

- **Database**: PostgreSQL 16 + pgvector
- **LLM**: Provider-agnostic via LiteLLM + Instructor (Ollama by default, cloud providers via `.env`)
- **Embeddings**: sentence-transformers `all-mpnet-base-v2` (768-dim)
- **Ranking & tuning**: LightGBM, Optuna, NumPy weight sweeps
- **Vision**: PyTorch, DINOv2, Gemma 4 VLM via Ollama
- **API / UI**: FastAPI, Streamlit
- **Experiment tracking**: MLflow
- **Infrastructure**: Docker Compose (local); Terraform + EC2 planned
- **CI**: GitHub Actions (ruff + unit tests)

## Quick Start

```bash
cp .env.example .env
make setup                 # create .venv and install requirements
make up                    # Postgres + pgvector, MLflow, Ollama
make test-setup            # verify all services connect
make db-init               # create tables

source .venv/bin/activate  # load species data, one stage at a time
python -m scripts.ingest --fetch-only
python -m scripts.ingest --extract
python -m scripts.ingest --reconcile
python -m scripts.ingest --embed

make serve                 # API → http://localhost:8001
make ui                    # UI  → http://localhost:8501
```

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for details.

## Repository Structure

```
mushroom-ai/
├── api/                 # FastAPI app: /lookalikes, /species, /associations
├── db/                  # SQLAlchemy models (Layers 1–3), connection, migrations
├── ingestion/           # fetch → extract → reconcile → embed
│   ├── sources/         # Per-source fetchers (Wikipedia, MushroomExpert, …)
│   ├── profiles/        # Grouping profiles defining the embedding groups
│   ├── rubric.py        # Schema contract: groups, numeric fields, features_to_text()
│   └── normalize.py     # Alias resolution against the controlled vocabulary
├── llm/                 # Provider-agnostic LLM layer (LiteLLM + Instructor)
├── similarity/          # Search engine
│   ├── search.py        # Retrieval, gating, scoring, context boosts
│   ├── aggregation.py   # weighted_avg / rrf / contrastive_gate + Jaccard blend
│   ├── jaccard.py       # Soft-Jaccard over vocabulary similarity matrices
│   ├── numeric.py       # Size, proximity and season-overlap comparison
│   ├── morphotype.py    # Coarse morphotype signature pre-filter
│   ├── ranker.py        # LightGBM learned re-ranker
│   ├── weights.py       # Per-group weights, filters, alpha
│   └── explain.py       # LLM lookalike explanations
├── evaluation/          # Recall@K, weight sweeps, MLflow logging
├── vision/              # Photo → features: DINOv2 heads, VLM labelling, dedup
├── ui/                  # Streamlit app + i18n (EN / IT)
├── scripts/             # CLIs: ingest, eval, optimize_jaccard, train_ranker, init_db
├── data/
│   ├── seed/            # Species list + known lookalike pairs (ground truth)
│   ├── reference/       # Morphological vocabulary + feature-usage map
│   └── models/          # Trained ranker model
├── tests/               # unit/ (no services) + integration/ (needs Docker services)
├── docs/                # Getting started + original implementation plan
├── infra/               # Terraform, nginx, MLflow (cloud deployment — planned)
└── docker-compose.yml
```

## License

MIT
