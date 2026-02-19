# Project 1: Lookalikes Finder — Implementation Plan

## Context

This is a mushroom lookalike finder. Given a species name, it finds similar-looking species
using **feature-based similarity** (not text-chunk RAG). The system has three data layers:

- **Layer 1** — Raw source observations (one row per species per source, LLM-extracted)
- **Layer 2** — Reconciled species profiles (one canonical row per species, merged from Layer 1)
- **Layer 3** — Per-group vector embeddings on Layer 2 for pgvector similarity search

At query time:
1. Validate species name
2. Retrieve its embeddings from Layer 3
3. Run 3 independent pgvector cosine similarity queries (morphological, ecological, taxonomic)
4. Merge and rerank with tunable weights (default 60/25/15)
5. Build feature comparison table
6. LLM generates "at a glance" explanation (the ONLY LLM call at query time)

## Architecture Rules (do not change these)

- **LLM calls**: Always go through `llm/client.py` (LiteLLM + Instructor). Never import litellm or instructor directly in other modules.
- **LLM provider**: Ollama local (`llama3.1:8b`), configured in `.env`. Do not change.
- **Structured outputs**: Pydantic schemas in `llm/schemas.py`, validated by Instructor.
- **Database**: PostgreSQL + pgvector. Models in `db/models.py`. Connection in `db/connection.py`.
- **Feature rubric**: Defined in `ingestion/rubric.py`. This is the single source of truth for what features exist and how they group.
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2` (384 dimensions, local, free).
- **API schemas**: `api/schemas.py` is the public contract (separate from LLM schemas and DB models).
- **Run from repo root** with venv activated. Docker services running via `docker compose up -d`.

## CRITICAL: Work incrementally

After EACH stage, verify it works before moving on. Every stage has a "Verify" section
with specific commands to run. Do not skip verification steps.

---

## Stage 1: Database Setup

**Goal**: Tables created in PostgreSQL, pgvector enabled, seed data loaded.

### Step 1.1: Implement `scripts/init_db.py`

The scaffold already has a working version. Extend it to also load ground truth pairs
from `data/seed/ground_truth_pairs.yaml` into the `ground_truth_pairs` table.

**Verify**:
```bash
python -m scripts.init_db
# Then connect and check:
# psql -h localhost -U mushroom -d mushroom_ai -c "\dt"
# Should show: source_observations, reconciled_species, ground_truth_pairs
```

### Step 1.2: Write a test for database connectivity

Create `tests/unit/test_db.py`:
- Test that `check_connection()` returns True
- Test that all three tables exist
- Test that ground truth pairs were loaded (count > 0)

**Verify**:
```bash
pytest tests/unit/test_db.py -v
```

---

## Stage 2: Wikipedia Source Fetcher

**Goal**: Fetch species page text from Wikipedia for each species in the seed list.

### Step 2.1: Implement `ingestion/sources/wikipedia.py`

Write a function `fetch_species_page(scientific_name: str) -> dict` that:
1. Calls the Wikipedia REST API to get the page content for the species
2. Returns `{"text": <page_text>, "url": <page_url>}` or `None` if not found
3. Handles redirects (e.g., common name → scientific name page)
4. Adds basic rate limiting (0.5s between requests)
5. Caches fetched pages to `data/cache/` to avoid re-fetching

### Step 2.2: Test the fetcher

Create `tests/unit/test_wikipedia.py`:
- Test fetching a known species ("Amanita caesarea") returns non-empty text
- Test fetching a non-existent species returns None
- Test that the returned text contains expected content (e.g., "Amanita" appears)

**Verify**:
```bash
pytest tests/unit/test_wikipedia.py -v
```

### Step 2.3: Fetch all seed species

Write a small script or add to `scripts/ingest.py` that iterates over
`data/seed/species_list.yaml` and fetches each species page. Print a summary
of how many succeeded vs failed.

**Verify**:
```bash
python -m scripts.ingest --fetch-only
# Should print: "Fetched 28/30 species" (or similar)
```

---

## Stage 3: LLM Feature Extraction (Layer 1)

**Goal**: For each fetched species page, extract structured features using the LLM
and store as Layer 1 rows in `source_observations`.

### Step 3.1: Write the extraction prompt

In `ingestion/extract.py`, write an extraction function that:
1. Takes species name + source text
2. Calls `structured_completion()` with `ExtractedSpeciesFeatures` as `response_model`
3. The system prompt should reference the rubric: what features to look for, how to
   handle uncertainty, to leave fields as null if not mentioned in the source
4. Returns the validated Pydantic model

The system prompt is critical. It should instruct the LLM:
- Extract only what is explicitly stated in the source text
- Use null for any feature not mentioned (do NOT guess)
- For colors, use descriptive natural language (e.g., "orange-red to yellow")
- For measurements, extract ranges where available
- Note any ambiguity in `extraction_notes`

### Step 3.2: Test extraction on a single species

Create `tests/integration/test_extraction.py`:
- Fetch the Wikipedia page for "Amanita muscaria" (well-documented species)
- Run extraction
- Verify the result has non-null values for cap.colors, cap.shape, edibility
- Verify it's a valid `ExtractedSpeciesFeatures` instance

**This test calls Ollama — it's an integration test, not a unit test.**

**Verify**:
```bash
pytest tests/integration/test_extraction.py -v
# May take 10-30 seconds per species (LLM inference)
```

### Step 3.3: Store extractions in Layer 1

Extend the extraction function to save results to `source_observations`:
- `features_json` = the Pydantic model dumped to dict
- `source_name` = "Wikipedia"
- `source_url` = the page URL
- `extraction_model` = the model string from config (e.g., "ollama/llama3.1:8b")
- Use the unique index on (scientific_name, source_name) to avoid duplicates on re-run

### Step 3.4: Run extraction for all seed species

Add to `scripts/ingest.py` a step that runs extraction for all fetched species.
Print progress and a summary of successes/failures.

**Verify**:
```bash
python -m scripts.ingest --extract
# Then check the database:
# psql -h localhost -U mushroom -d mushroom_ai -c "SELECT scientific_name, source_name FROM source_observations LIMIT 10"
```

---

## Stage 4: Reconciliation (Layer 2)

**Goal**: For each species with Layer 1 data, produce a reconciled Layer 2 profile.

### Step 4.1: Implement reconciliation for single-source species

Since v1 only has Wikipedia as a source, reconciliation is initially simple:
copy the single source observation's features as the reconciled profile. But
implement it properly so multi-source reconciliation can be added later:

In `ingestion/reconcile.py`:
1. For species with 1 source: copy features directly, confidence=1.0, needs_review=False
2. For species with 2+ sources (future): call `structured_completion()` with
   `ReconciliationResult` schema, passing all source observations
3. Store result in `reconciled_species` table
4. Handle incremental updates: only re-reconcile species whose sources changed

### Step 4.2: Test reconciliation

Create `tests/unit/test_reconcile.py`:
- Test single-source reconciliation produces a valid ReconciledSpecies row
- Test that features_json is populated and non-empty
- Test that edibility is set correctly

**Verify**:
```bash
python -m scripts.ingest --reconcile
pytest tests/unit/test_reconcile.py -v
# Check DB:
# psql ... -c "SELECT scientific_name, edibility, reconciliation_confidence FROM reconciled_species LIMIT 10"
```

---

## Stage 5: Embedding (Layer 3)

**Goal**: Convert reconciled features into per-group vector embeddings in pgvector.

### Step 5.1: Implement `ingestion/rubric.py` → `features_to_text()`

This is the bridge between structured data and embeddings. For each feature group,
convert the relevant fields into a natural language description.

Example for morphological features of Amanita muscaria:
"Cap convex to flat, 8-20 cm diameter, bright red to orange-red with white wart-like
scales. Gills free, crowded, white. Stem 8-20 cm tall, white, with a bulbous base.
Ring present, skirt-like, white, persistent. Volva present as rings of scales around
the bulbous base. Flesh white, no color change. Spore print white."

Rules:
- Skip null/missing fields (do not say "unknown" or "None")
- Use natural language, not key-value pairs
- Keep it concise (field-guide style)

### Step 5.2: Implement `ingestion/embed.py`

1. Load sentence-transformers model (cache it — only load once)
2. For each ReconciledSpecies row:
   a. Generate morphological text → embed → `embedding_morphological`
   b. Generate ecological text → embed → `embedding_ecological`
   c. Generate taxonomic text → embed → `embedding_taxonomic`
   d. Update the row in Postgres
3. Handle incremental: only re-embed species whose reconciled_at > embedded_at

### Step 5.3: Test embeddings

Create `tests/unit/test_embed.py`:
- Test `features_to_text()` produces non-empty strings for a sample feature dict
- Test that similar species produce more similar embeddings than dissimilar ones
  (e.g., cosine similarity of Amanita muscaria vs Amanita caesarea >
  cosine similarity of Amanita muscaria vs Boletus edulis)

**Verify**:
```bash
python -m scripts.ingest --embed
pytest tests/unit/test_embed.py -v
# Check DB:
# psql ... -c "SELECT scientific_name, embedding_morphological IS NOT NULL as has_emb FROM reconciled_species LIMIT 10"
# All should show has_emb = true
```

---

## Stage 6: Similarity Search Engine

**Goal**: Given a species name, find the top-K most similar species using pgvector.

### Step 6.1: Implement `similarity/search.py` → `search_by_group()`

Write a function that takes a species ID and a feature group name, and runs
a pgvector cosine similarity query against that group's embedding column.
Returns list of (species_id, similarity_score) tuples.

Use raw SQL via SQLAlchemy for the pgvector query:
```sql
SELECT id, scientific_name,
       1 - (embedding_morphological <=> :query_vector) AS similarity
FROM reconciled_species
WHERE id != :query_id
ORDER BY embedding_morphological <=> :query_vector
LIMIT :top_k
```

### Step 6.2: Implement `search_lookalikes()` — full merge/rerank pipeline

1. Look up query species by name (validate it exists)
2. Call `search_by_group()` for each of the 3 groups
3. Merge results: union all candidate species, compute weighted score
4. Sort by overall score descending
5. Return top-K candidates with per-group scores

### Step 6.3: Test against ground truth

Create `tests/integration/test_similarity.py`:
- For each ground truth pair (from `ground_truth_pairs` table):
  Query species_a → check that species_b appears in top-10 results
- Report pass rate (target: >70% of known pairs found in top-10 for v1)
- This is the core quality metric for the system

**Verify**:
```bash
pytest tests/integration/test_similarity.py -v
# Should print pass rate for ground truth pairs
```

### Step 6.4: Build feature comparison table

In `similarity/search.py`, add a function that takes the query species and
a list of candidate species, and builds a comparison structure:
- For each feature in the rubric, show the query value and candidate value side by side
- Flag which features are similar vs different (drove match vs distinguishing)

**Verify**: Call from a test script, print the comparison for "Amanita caesarea"
and verify it makes sense visually.

---

## Stage 7: LLM Explanation (query-time)

**Goal**: Generate a human-readable "at a glance" summary from the comparison table.

### Step 7.1: Implement `similarity/explain.py`

1. Takes the comparison table (query species + top candidates with features)
2. Formats it into a prompt for the LLM
3. Calls `structured_completion()` with `LookalikeExplanation` as response_model
4. Returns the explanation (summary, notable pairs, safety warning)

The prompt should:
- Present the comparison data as structured input
- Ask for a concise summary highlighting WHY these species are confused
- Demand a safety warning if any candidate is toxic/deadly
- Keep it short (2-3 sentences per notable pair)

### Step 7.2: Test explanation generation

Create `tests/integration/test_explain.py`:
- Run a full pipeline for "Amanita caesarea" (similarity search → explanation)
- Verify the explanation mentions A. phalloides or A. muscaria
- Verify a safety warning is present (since deadly lookalikes exist)

**Verify**:
```bash
pytest tests/integration/test_explain.py -v
```

---

## Stage 8: FastAPI Endpoints

**Goal**: Expose the full pipeline as REST API.

### Step 8.1: Wire up `POST /api/v1/lookalikes`

In `api/routes.py`:
1. Validate species exists in DB (return 404 if not)
2. Build `SimilarityWeights` from request (or defaults)
3. Call `search_lookalikes()` from similarity/search.py
4. Call `generate_explanation()` from similarity/explain.py
5. Map results to `LookalikeResponse` schema
6. Return response

### Step 8.2: Wire up `GET /api/v1/species/{name}` and `GET /api/v1/species`

Simple database lookups from `reconciled_species`.

### Step 8.3: Test API endpoints

Extend `tests/unit/test_api.py`:
- Test `/health` returns 200
- Test `/api/v1/species` returns a list
- Test `/api/v1/lookalikes` with a known species returns candidates
- Test `/api/v1/lookalikes` with unknown species returns 404
- Mock the LLM call for unit tests (only integration tests should call Ollama)

**Verify**:
```bash
pytest tests/unit/test_api.py -v
# Also manual test:
make serve
# Then in another terminal:
curl -X POST http://localhost:8001/api/v1/lookalikes \
  -H "Content-Type: application/json" \
  -d '{"species_name": "Amanita caesarea"}'
```

---

## Stage 9: Streamlit Dashboard

**Goal**: Build the user-facing UI.

### Step 9.1: Wire the UI to the FastAPI backend

In `ui/app.py` (scaffold exists with layout):
1. On search button click, POST to `http://localhost:8001/api/v1/lookalikes`
2. Display the LLM explanation summary at the top (prominent box)
3. Show safety warning in red if present
4. Render ranked candidates as a list/table with color-coded edibility:
   🔴 deadly, 🟠 toxic, 🟡 caution, 🟢 edible
5. Each candidate expands to show the feature comparison table

### Step 9.2: Add example species buttons

The scaffold already has example buttons. Make them functional — clicking one
fills the input and auto-triggers the search.

### Step 9.3: Add weight sliders

The scaffold has sliders. Wire them to the API request's weight overrides.

**Verify**:
```bash
# Terminal 1:
make serve
# Terminal 2:
make ui
# Open http://localhost:8501, search for "Amanita caesarea"
# Should see results with explanation and colored danger levels
```

---

## Stage 10: MLflow Tracking + Ground Truth Validation

**Goal**: Track ingestion quality and query performance; validate against known pairs.

### Step 10.1: MLflow in ingestion pipeline

In `scripts/ingest.py`, log to MLflow:
- Experiment: "ingestion"
- Per run: species count, extraction success rate, reconciliation conflict rate,
  embedding duration, model used

### Step 10.2: MLflow in query pipeline

In `similarity/search.py` or `api/routes.py`, log:
- Experiment: "queries"
- Per query: latency, top candidate similarity score, number of candidates above threshold

### Step 10.3: Ground truth validation script

Create `scripts/validate.py` that:
1. Loads all ground truth pairs
2. For each pair, runs the similarity search
3. Checks if the expected species appears in top-K
4. Logs pass rate to MLflow
5. Prints a report

**Verify**:
```bash
python -m scripts.validate
# Check MLflow UI at http://localhost:5000
```

---

## Summary of run commands per stage

| Stage | Key command | What it proves |
|-------|------------|----------------|
| 1 | `python -m scripts.init_db` | Tables exist, ground truth loaded |
| 2 | `python -m scripts.ingest --fetch-only` | Wikipedia pages fetched |
| 3 | `python -m scripts.ingest --extract` | Features extracted to Layer 1 |
| 4 | `python -m scripts.ingest --reconcile` | Layer 2 profiles created |
| 5 | `python -m scripts.ingest --embed` | Embeddings in pgvector |
| 6 | `pytest tests/integration/test_similarity.py` | Ground truth pairs found |
| 7 | `pytest tests/integration/test_explain.py` | LLM explanations work |
| 8 | `curl POST .../lookalikes` | Full API pipeline works |
| 9 | Browser at localhost:8501 | UI shows results |
| 10 | `python -m scripts.validate` | Quality metrics tracked |
