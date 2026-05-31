# Mushroom Visual Feature Extraction — Project Plan

> **Handover document for implementation.**
> Drop this file and `VISION_DATA_RETRIEVAL.md` into `vision/` in the
> mushroom-ai repo. Reference from `CLAUDE.md`.

---

## 1. Objective

Build a vision pipeline that extracts discrete morphological features from
mushroom photographs. Given one or more images of a specimen, predict specific
attributes from the existing `morphological_vocabulary.yaml` ontology — not
the species, but features like `hymenium_type: gills`, `cap_shape: convex`,
`stem_shape: bulbous`, etc.

Each feature is an independent classification task with a small, fixed set
of canonical values. The system outputs per-feature predictions with
confidence scores and flags features that cannot be determined from the
available image(s).

This is not a species classifier. It is a multi-attribute predictor over
a well-defined morphological vocabulary, designed to eventually feed into
the existing lookalike similarity search pipeline.

---

## 2. Assumptions & Constraints

- **Data scale:** 100+ labeled images per feature-value achievable by
  scraping iNaturalist, Mushroom Observer, GBIF, Wikimedia Commons.
- **Labeling shortcut:** The existing 543-species PostgreSQL database
  maps each species to its morphological features. Any image of a known
  species inherits all its feature labels automatically (weak supervision).
- **Image quality:** Mixed — curated species reference photos plus
  field observations of variable quality, lighting, framing.
- **Compute:** Apple Silicon MPS (M-series Mac) for development and
  training. DINOv2-B/14 (86M params frozen) and CLIP ViT-B/16 both
  run comfortably on MPS. Cloud APIs (Ollama, OpenAI) available if needed.
- **Framework:** PyTorch. HuggingFace `transformers` for model loading.
  `torchvision.transforms.v2` for image processing.
- **Tracking:** MLflow (already in the mushroom-ai stack).
- **Database:** PostgreSQL + pgvector (already in the stack). Extended
  with image registry and annotation tables.
- **Integration:** This pipeline lives as a `vision/` module inside the
  existing `mushroom-ai` repository. Shares the database and MLflow
  instance.

---

## 3. Architecture

### 3.1 Core Model

```
Image
  │
  ▼
Preprocessing (deterministic, versioned)
  │
  ▼
Frozen DINOv2-B/14 backbone (weights never change)
  → CLS embedding: 768-d vector (computed once, cached to disk)
  │
  │  All heads read the SAME cached 768-d embedding.
  │  One backbone, one embedding per image, shared across all heads.
  │
  ├──→ Head: hymenium_type    (Linear 768→7)   → "pores" (0.94)
  ├──→ Head: overall_body_form (Linear 768→11) → "boletoid" (0.91)
  ├──→ Head: cap_shape        (Linear 768→11)  → "convex" (0.87)
  ├──→ Head: cap_color        (Linear 768→14)  → "brown" (0.82)
  └──→ ... more heads added incrementally
```

Each head is a tiny linear classifier: 768 × num_classes + bias.
DINOv2 is frozen — the only trainable parameters are the heads.
The `hymenium_type` head (7 classes) has 5,383 trainable parameters.
Even 15 heads combined: ~150K trainable params against 86M frozen.

Since the backbone is frozen, the mapping `image → 768-d vector` is
deterministic. Extract all embeddings once, cache to disk, and head
training becomes logistic regression on 768-d vectors. Trains in
seconds on CPU.

### 3.2 Universal Output Contract

**Every head** — whether context or morphological — returns the same
output structure:

```python
{
    "status": "confident" | "uncertain" | "not_determinable",
    "value": "gills",              # predicted class (None if not_determinable)
    "confidence": 0.94,            # calibrated probability
    "runner_up": "ridges",         # second-best class
    "runner_up_confidence": 0.04,
    "reason": None,                # explanation if uncertain/not_determinable
}
```

| Status | Meaning | Trigger |
|--------|---------|---------|
| `confident` | Clear prediction | High softmax, above threshold |
| `uncertain` | Model ran but can't decide | Low softmax or top-2 too close |
| `not_determinable` | Feature not visible in image | Wrong view angle/framing (context gating, added later) |

Confidence thresholds are per-head, tuned on validation data via
temperature scaling.

### 3.3 Key Design Decisions

**Why frozen backbone + linear probes?**
With ~100 images per class, fine-tuning 86M params would overfit.
Linear probes on frozen features are the textbook approach at this
data scale. Fallback for underperforming heads: LoRA on last 2
transformer blocks.

**Why DINOv2 over CLIP as backbone?**
DINOv2 outperforms CLIP on fine-grained biological datasets
(iNaturalist: +8% linear probe accuracy over OpenCLIP ViT-G/14).
Patch-level features are richer for localizable attributes.
CLIP is used only as a zero-shot baseline and for bootstrapping
view-angle labels.

**Why independent heads?**
Each head has its own loss, class weights, and train/val split.
This allows incremental development (deploy `hymenium_type` while
still collecting data for `cap_shape`), handles missing labels
naturally, and keeps debugging clean. Joint training is a later
optimization once all heads are individually validated.

**Why not folder-per-class (ImageFolder)?**
One image carries labels for many features simultaneously. An image
of *Boletus edulis* is labeled for `hymenium_type`, `cap_shape`,
`cap_color`, `stem_shape`, etc. Images are stored once, indexed by
a manifest that maps each image to all its labels.

### 3.4 Feature Tiers

**Tier 1 — Start here (high visual signal, high impact):**

| Feature | Classes | Notes |
|---------|---------|-------|
| `hymenium_type` | 7 | gills/ridges/pores/teeth/smooth/gleba/alveolate |
| `overall_body_form` | 11 | agaricoid/boletoid/cantharelloid/... |
| `cap_shape` | 11 | convex/flat/conical/campanulate/... |
| `cap_color` | 14 | from color_palette canonical values |

**Tier 2 — Add next (moderate complexity):**

| Feature | Classes | Notes |
|---------|---------|-------|
| `surface_texture` | 14 | smooth/fibrillose/scaly/warty/... |
| `gill_attachment` | 8 | free/adnate/decurrent/... |
| `stem_shape` | 8 | equal/clavate/bulbous/... |
| `cap_surface_moisture` | 7 | dry/viscid/glutinous/... |
| `scales_or_warts` | 3+1 | scales/warts/both + implicit "none" |
| `cap_color_pattern` | 5 | uniform/darker_at_center/mottled/... |

**Tier 3 — Advanced (subtle or angle-dependent):**

`gill_spacing`, `gill_edge_texture`, `cap_margin`, `stem_reticulation`,
`stipe_attachment_position`, `stem_interior`, `overall_size_class`

**Not targetable from images:**

`flesh_odor`, `flesh_taste`, `flesh_texture`, `gill_texture` (waxy/brittle),
`stem_consistency` — these require physical interaction.

### 3.5 Context Heads & View Gating (deferred — not needed to start)

View angle and framing classification are defined but **not trained
upfront**. The initial approach trains morphological heads ungated
on all images. Context gating is added later only if error analysis
shows it would help.

**Context heads (train later if needed):**

| Head | Classes |
|------|---------|
| `view_angle` | top, side, underside, cross_section, detail, habitat |
| `framing` | full_specimen, upper_half, detail_macro, habitat_distant, cross_section |
| `subject_count` | single, few, cluster, with_hand |
| `maturity` | button, immature, mature, old |
| `condition` | fresh, wet_conditions, damaged, dried |

**Bootstrap strategy (no manual annotation needed):** CLIP zero-shot
is used in Phase 0 to auto-tag view angles across the entire dataset.
These labels are stored as `annotation_type='model_predicted'` and
spot-checked. If morphological heads underperform, a DINOv2
`view_angle` head is trained on the CLIP-bootstrapped labels, and
morphological heads are retrained with context gating.

**Feature applicability matrix** (used when gating is active):

```yaml
# config/feature_applicability.yaml
hymenium_type:
  view_angle:
    required: [underside, side]
    optional: [detail]
  framing: [full_specimen, upper_half, detail_macro]
  subject_count: [single, few]

cap_shape:
  view_angle:
    required: [side]
    optional: [top]
  framing: [full_specimen, upper_half]

cap_color:
  view_angle:
    required: [top, side]
    optional: [underside]
  framing: [full_specimen, upper_half, detail_macro]

overall_body_form:
  view_angle:
    required: [side, top, underside, habitat]
  framing: [full_specimen, habitat_distant]

stem_shape:
  view_angle:
    required: [side]
  framing: [full_specimen]

gill_attachment:
  view_angle:
    required: [underside]
    optional: [side]
  framing: [full_specimen, upper_half, detail_macro]

overall_size_class:
  view_angle:
    required: [side]
  framing: [full_specimen]
  subject_count: [single, with_hand]
```

---

## 4. Data Architecture

### 4.1 Three-Stage Image Storage

```
data/
├── images/
│   ├── raw/                              # Stage 1: fetched originals, untouched
│   │   ├── a3f8c1d2.jpg                  # content-hash named (SHA256[:12])
│   │   └── ...
│   │
│   └── processed/
│       ├── v1/                           # Stage 2: preprocessed (versioned)
│       │   ├── a3f8c1d2.jpg              # same ID, cleaned + resized
│       │   └── ...
│       └── v2/                           # later: different pipeline config
│
├── embeddings/
│   └── dinov2-base/
│       ├── processed-v1/                 # Stage 3: cached 768-d vectors
│       │   ├── a3f8c1d2.pt
│       │   └── ...
│       └── processed-v2/
│
└── pipeline_configs/
    ├── preprocess_v1.yaml
    └── ...
```

**Invariant:** The path `embeddings/{backbone}/{preprocess_version}/`
fully determines provenance. Changing either means a new directory.

**Raw images** — archival, never modified. Re-run from raw without
re-fetching if preprocessing changes.

**Processed images** — deterministic, viewable JPEGs (no float
normalization, no augmentation). Suitable for visual QA.

**Embeddings** — the actual training input. Heads never see raw pixels.

### 4.2 Preprocessing Pipeline (v1 — minimal)

All steps deterministic. No randomness, no augmentation.
Config stored in `pipeline_configs/preprocess_v1.yaml`.

```yaml
# pipeline_configs/preprocess_v1.yaml
version: "v1"
description: "Minimal viable preprocessing: format + resize + border removal"

format:
  target_colorspace: "RGB"
  output_format: "JPEG"
  jpeg_quality: 95

filtering:
  min_short_side_px: 100
  max_aspect_ratio: 3.0
  reject_corrupt: true

resize:
  strategy: "shortest_side"
  target_short_side: 256
  interpolation: "LANCZOS"

border_removal:
  enabled: true
  method: "constant_edge_detection"
  threshold_std: 5.0
  max_crop_fraction: 0.15
```

Steps:
1. Convert to RGB, save as JPEG q=95
2. Reject if < 100px shortest side or corrupt
3. Detect and crop constant-color letterbox borders
4. Resize shortest side → 256px, preserve aspect ratio, LANCZOS

**Not in v1 (deferred):**
- No augmentation (training-time only)
- No ImageNet float normalization (applied at embedding extraction)
- No center crop to 224 (done at embedding time)
- No white balance correction (add in v2 if color heads underperform)

### 4.3 Embedding Extraction

Applied at extraction time on processed images:

```python
EMBEDDING_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

Batched extraction on MPS. Idempotent (skips already-extracted).

### 4.4 Database Schema (extends existing mushroom-ai)

```sql
CREATE TABLE image_registry (
    image_id        TEXT PRIMARY KEY,       -- content hash (SHA256[:12])
    file_path       TEXT NOT NULL,          -- relative path in data/images/raw/
    species         TEXT,                   -- FK to reconciled_species if known
    source          TEXT NOT NULL,          -- 'inaturalist', 'mushroom_observer', etc.
    source_id       TEXT,                   -- original observation/photo ID
    source_url      TEXT,                   -- provenance link
    license         TEXT,                   -- CC-BY, CC0, etc.
    resolution_w    INTEGER,
    resolution_h    INTEGER,
    fetched_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE image_annotations (
    image_id        TEXT REFERENCES image_registry(image_id),
    annotation_type TEXT NOT NULL,          -- 'auto_from_species' | 'manual' | 'model_predicted'
    feature_name    TEXT NOT NULL,          -- 'hymenium_type', 'view_angle', etc.
    feature_value   TEXT NOT NULL,          -- 'gills', 'underside', etc.
    confidence      REAL,                  -- NULL for manual, model conf for predicted
    annotator       TEXT,                  -- 'species_db', 'clip-zero-shot', 'dinov2-head:v1'
    created_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (image_id, feature_name, annotation_type)
);

CREATE TABLE image_quality (
    image_id        TEXT REFERENCES image_registry(image_id),
    is_verified     BOOLEAN DEFAULT FALSE,
    quality_score   REAL,
    exclude_reason  TEXT,
    PRIMARY KEY (image_id)
);
```

Three annotation sources:
- **`auto_from_species`**: Bulk-populated from reconciled_species.
  All images of *Boletus edulis* get `hymenium_type: pores` etc.
- **`model_predicted`**: CLIP zero-shot view-angle tags, or later
  trained head predictions stored back for filtering.
- **`manual`**: Human corrections and additions.

### 4.5 Manifest for Training

Training manifests are built by querying the DB. Exported to parquet
for reproducibility and portability (e.g., Colab).

```python
manifest = build_manifest(
    db_connection,
    feature_name="hymenium_type",
    # view_filter=["underside", "side"],  # add later when gating is active
    exclude_unverified=False,
)
manifest.to_parquet("data/manifests/hymenium_type_v1.parquet")
```

### 4.6 Image Sources

See `VISION_DATA_RETRIEVAL.md` for full fetching specification
including iNaturalist API details, target counts, and species
selection strategy.

Priority: iNaturalist → Mushroom Observer → GBIF → Wikimedia Commons.

---

## 5. Module Structure

```
vision/
├── VISION_PROJECT_PLAN.md           # this file
├── VISION_DATA_RETRIEVAL.md         # data fetching specification
│
├── config/
│   ├── features.yaml                # feature registry: name, num_classes, tier
│   ├── feature_applicability.yaml   # view/framing requirements (for later gating)
│   ├── training.yaml                # hyperparams, paths, device settings
│   └── pipeline_configs/
│       └── preprocess_v1.yaml       # preprocessing pipeline config
│
├── data/
│   ├── fetch.py                     # image scraping from iNaturalist/MO/GBIF
│   ├── preprocess.py                # ImagePreprocessor: raw → processed
│   ├── dataset.py                   # MushroomFeatureDataset (PyTorch Dataset)
│   ├── labeling.py                  # auto-label from species DB
│   ├── manifest.py                  # build_manifest() from DB queries
│   ├── transforms.py                # training-time augmentations (NOT preprocessing)
│   └── splits.py                    # stratified train/val/test splitting
│
├── models/
│   ├── backbone.py                  # DINOv2 frozen feature extractor
│   ├── heads.py                     # FeatureHead: linear probe per feature
│   ├── multi_head.py                # MultiHeadPredictor wrapper
│   └── embed.py                     # EmbeddingExtractor: processed → .pt vectors
│
├── inference/
│   ├── predict.py                   # single-image prediction for all active heads
│   ├── context.py                   # context heads → applicability filter (later)
│   ├── multi_view.py                # aggregate predictions across views (later)
│   └── confidence.py                # temperature calibration + thresholding
│
├── training/
│   ├── trainer.py                   # per-head training loop on cached embeddings
│   ├── losses.py                    # CE + label smoothing + class weights
│   └── metrics.py                   # per-class accuracy, macro-F1, confusion matrix
│
├── tracking/
│   └── mlflow_utils.py              # experiment logging, model registry
│
├── evaluation/
│   ├── evaluate.py                  # full evaluation pipeline
│   └── report.py                    # per-feature performance reports
│
├── scripts/
│   ├── fetch_images.py              # CLI: scrape images for species list
│   ├── run_preprocess.py            # CLI: raw → processed (versioned)
│   ├── run_embed.py                 # CLI: processed → embeddings
│   ├── label_from_db.py             # CLI: auto-label from species DB
│   ├── clip_tag_views.py            # CLI: CLIP zero-shot view-angle tagging
│   ├── train_head.py                # CLI: train a single feature head
│   ├── train_all.py                 # CLI: train all active heads
│   ├── evaluate_head.py             # CLI: evaluate a single head
│   ├── predict_image.py             # CLI: run inference on image(s)
│   └── pipeline_status.py           # CLI: show counts at each stage
│
└── notebooks/
    ├── 01_clip_baseline.ipynb       # Phase 0: zero-shot probing
    ├── 02_data_exploration.ipynb    # image stats, class distributions
    ├── 03_training_analysis.ipynb   # learning curves, confusion matrices
    └── 04_error_analysis.ipynb      # failure cases, preprocessing impact
```

---

## 6. Current Status (2026-05-10)

### VLM labelling phase — closed 2026-05-10

VLM extractor pipeline ran across 8 candidate features. Three shipped as
project defaults; five shelved with documented reasons. Path A (single-shot
MCQA, 896×896 center-crop, no fewshot) is the project default; Path C
(staged) only helps when stage 1 is genuinely separable.

| Feature | Status | Result |
|---------|--------|--------|
| hymenium_type | **shipped** (Path C) | 89.1% non-null acc on hymenium leaves |
| cap_color | **shipped** (Path A + fuzzy) | 95% fuzzy match against multi-label GT |
| volva_presence | **shipped** (Path A v2) | 100% precision on `present` (Amanita-flag), 81% non-null acc |
| ring_presence | shelved | Species-vs-image GT gap (cortinas/fugacious) |
| substrate | shelved | Same GT gap (biology ≠ visibility) |
| surface_texture | shelved | 896×896 resolution ceiling |
| stem_shape | shelved | Equal-bias on a fuzzy threshold; Path C also failed |
| cap_shape | shelved | Convex-bias + maturity confound; ViT head from species labels already does 74.8% |

Memory note: `~/.claude/projects/.../memory/project_vlm_phase_close.md`.
Next phase plan: `~/.claude/plans/snoopy-zooming-chipmunk.md` (Phase 3 — ViT
head retraining with shipped VLM extractors as teachers).

### Original status snapshot (2026-04-19)

### What's implemented and working

**Data pipeline (Phase 0 + 1):** complete. Fetch → preprocess (v1) → embed
(DINOv2-base CLS token, 768-d) → auto-label from species DB. ~12K images
in registry, ~11K processed+embedded.

**VLM quality gating:** Gemini-based grading of image_quality,
subject_present, subject_dominance, and per-feature visibility
(cap_visible, hymenium_visible). Integrated into manifest building via
LEFT JOINs with graceful degradation (ungraded images pass through).

**Manual QA labeling tool:** `vision/labeling/manual_qa_app.py` — FastAPI
keyboard-driven labeling (y=good, n=bad, s=skip). JSONL crash-safe
persistence + DB upsert on shutdown. Integrated into manifest via
`manual_qa` LEFT JOIN. 2,622 images labeled for hymenium_type (2,038
good, 584 bad = 22% rejection rate).

**Training infrastructure (Phase 2a-2b):** complete. Per-head trainer
with species-level GroupShuffleSplit, sqrt-dampened WeightedRandomSampler,
CosineAnnealingLR, early stopping on val macro-F1, MLflow logging.

**Classification heads:** Two-layer MLP (768→256→128→num_classes) with
BatchNorm + Dropout, replacing original linear probes. Trained on
frozen DINOv2-base CLS embeddings.

### Tier 1 training results (2026-04-19)

| Feature | Images | Classes | Test Acc | Macro F1 | Weighted F1 | Best Epoch | Labels |
|---------|--------|---------|----------|----------|-------------|------------|--------|
| hymenium_type | 3,952 | 6 | 90.5% | 0.635 | 0.90 | 3 | species_propagated |
| overall_body_form | 11,655 | 10 | 85.6% | 0.505 | 0.85 | 3 | species_propagated |
| cap_shape | 9,176 | 9 | 74.8% | 0.314 | 0.74 | 1 | species_propagated |
| cap_color | 3,885 | 12 | 43.8% | 0.280 | 0.46 | 3 | species_propagated |
| volva_presence | 5,349 | 2 | 82.7% | 0.753 | 0.83 | 3 | vlm_labeled (2026-05-31) |

**Volva head — first VLM-teacher-labelled head (2026-05-31):** trained on 7,310 VLM sidecars
(0.98-dedup pool, 1,961 abstentions skipped → 5,349 usable labels across 632 species). Test
accuracy 82.7%, macro-F1 0.753 — highest macro-F1 of any head trained so far. Per-class:
`present` p/r/F1 = 0.64/0.60/0.62 (184 test), `absent` 0.88/0.90/0.89 (603 test). Species-propagated
baseline doesn't exist for volva (would have been ~99% "present" for all Amanitas regardless of
image — the exact noise the per-image VLM labels fix).

**Key observations:**
- hymenium_type and overall_body_form work well — structural features
  that DINOv2 CLS token captures. Main errors: ridges↔gills, teeth↔pores.
- cap_shape is marginal — dominant "convex" class OK (0.87 F1) but many
  shape classes too visually similar for a single CLS vector.
- cap_color is weak — color is species-variable (same species, different
  cap colors across specimens), lighting-dependent, and the CLS token
  compresses color information.

### Training bugs fixed (2026-04-19)

1. **Double balancing** — balanced sampler + class-weighted loss caused
   the model to never predict the dominant class. Fix: removed class
   weights from loss, kept sqrt-dampened sampler only.
2. **Phantom classes** — classes with 0 images in manifest (e.g. alveolate)
   stayed in class_names because `_filter_rare_classes` only catches classes
   *in* the manifest with too few species. Fix: always sync class_names
   with classes actually present in the manifest.

### Open questions (next steps)

1. **CLS token vs richer representations** — CLS token compresses the
   entire image into one 768-d vector. Patch tokens or intermediate
   layers could help for localized features (cap texture, gill detail).
   But this moves toward full ViT territory. Worth exploring intermediate
   DINOv2 layer concatenation or spatial attention pooling before
   abandoning the frozen backbone approach.
2. **Noisy labels** — species-propagated labels are inherently noisy
   (not every image of a species clearly shows the feature). Manual QA
   helps but is per-feature. Could explore: confidence-weighted training,
   curriculum learning, or VLM-based per-image annotation.
3. **Cap color** — likely needs preprocessing v2 with white balance
   correction and/or a different embedding strategy (color histograms,
   augmentation).
4. **Manual QA for other features** — currently only hymenium_type has
   manual QA labels. Other features would benefit from the same treatment.

---

## 7. Phases & TODOs

### Phase 0: CLIP Zero-Shot Baseline + View Tagging

**Goal:** (a) Establish free performance floor for morphological features.
(b) Auto-tag view angles across entire dataset using CLIP — no manual
annotation needed.

**Duration:** 1–2 days

#### 0a: Module skeleton

- [ ] Create `vision/` directory structure as shown in Section 5
- [ ] Create `config/features.yaml` — extract feature names, class lists,
      num_classes, and tier from `data/reference/morphological_vocabulary.yaml`
      for all Tier 1 features plus context heads
- [ ] Create `config/feature_applicability.yaml` as shown in Section 3.5
- [ ] Create `pipeline_configs/preprocess_v1.yaml` as shown in Section 4.2
- [ ] Create `config/training.yaml` with defaults:
  ```yaml
  device: "mps"
  backbone: "facebook/dinov2-base"
  clip_model: "openai/clip-vit-base-patch16"
  embedding_dim: 768
  default_lr: 1e-3
  default_weight_decay: 1e-4
  default_dropout: 0.1
  default_label_smoothing: 0.1
  max_epochs: 50
  early_stopping_patience: 10
  paths:
    raw_images: "data/images/raw"
    processed_images: "data/images/processed"
    embeddings: "data/embeddings"
    manifests: "data/manifests"
    pipeline_configs: "vision/pipeline_configs"
  ```

#### 0b: CLIP zero-shot baseline for morphological features

- [ ] Write `notebooks/01_clip_baseline.ipynb`:
  - [ ] Load `openai/clip-vit-base-patch16` via HuggingFace transformers
  - [ ] For each Tier 1 feature, construct text prompts from vocabulary
        definitions. Use descriptive prompts, e.g.:
    ```python
    hymenium_prompts = {
        "gills":     "a mushroom with blade-like gills under the cap",
        "pores":     "a mushroom with a sponge-like pore surface under the cap",
        "ridges":    "a mushroom with blunt forking ridges under the cap",
        "teeth":     "a mushroom with hanging spines or teeth under the cap",
        "smooth":    "a mushroom with a smooth fertile surface",
        "gleba":     "a puffball mushroom with enclosed internal spore mass",
        "alveolate": "a morel mushroom with deeply pitted honeycomb surface",
    }
    ```
  - [ ] Assemble a small test set: ~5-10 images per feature-value for
        Tier 1 features (~150-200 images total). Can use the first batch
        of fetched images or manually selected reference photos.
  - [ ] Run zero-shot classification, record per-feature accuracy
  - [ ] Produce results table: feature × CLIP accuracy
  - [ ] Document: which features >80% (may not need supervised heads),
        which <60% (priority targets for DINOv2 heads)

#### 0c: CLIP zero-shot view-angle tagging (bulk)

- [ ] Implement `vision/scripts/clip_tag_views.py`:
  - [ ] Load CLIP model
  - [ ] Define view-angle prompts:
    ```python
    view_prompts = {
        "top":           "a photograph looking down at the top of a mushroom cap",
        "side":          "a side view photograph of a mushroom showing the cap and stem in profile",
        "underside":     "a photograph showing the underside gills or pores of a mushroom",
        "habitat":       "a distant photograph of mushrooms growing in their natural habitat",
        "detail":        "a close-up macro photograph of mushroom surface texture",
        "cross_section": "a photograph of a mushroom cut in half showing the interior",
    }
    ```
  - [ ] Run over all processed images in the dataset
  - [ ] Store predictions in `image_annotations` table with:
        `annotation_type='model_predicted'`, `annotator='clip-zero-shot'`,
        `confidence=<softmax_prob>`
  - [ ] Print distribution summary (how many per view class)
- [ ] Spot-check accuracy: manually verify ~50 images across all view
      classes. Record accuracy. If >85%, labels are usable for later
      context gating without manual annotation.

**Output:** Baseline accuracy table for morpho features. CLIP-bootstrapped
view-angle labels across entire dataset. Evidence for which features need
supervised heads.

---

### Phase 1: Data Pipeline

**Goal:** Build the three-stage image pipeline (fetch → preprocess → embed)
and the database schema for image management.

**Duration:** 3–5 days

#### 1a: Database schema

- [ ] Create DB migration adding `image_registry`, `image_annotations`,
      `image_quality` tables (SQL in Section 4.4)
- [ ] Add indexes:
  ```sql
  CREATE INDEX idx_annotations_feature ON image_annotations(feature_name, feature_value);
  CREATE INDEX idx_annotations_image ON image_annotations(image_id);
  CREATE INDEX idx_registry_species ON image_registry(species);
  ```

#### 1b: Image fetching

- [ ] Implement `vision/data/fetch.py`:
  - [ ] iNaturalist API client:
    - [ ] `search_taxon(name) → taxon_id`
    - [ ] `fetch_observations(taxon_id, max_photos, quality_grade='research') → list[ObservationPhoto]`
    - [ ] Photo URL construction: replace "square" with "large" in URL
    - [ ] Rate limiting: 1s delay between requests
    - [ ] License filtering: only CC-licensed images
  - [ ] Content-hash naming (SHA256[:12]) for deduplication
  - [ ] Save to `data/images/raw/{hash}.jpg`
  - [ ] Insert row into `image_registry` with source metadata
- [ ] Implement `vision/scripts/fetch_images.py` CLI:
  - [ ] Input: species list (from DB query or explicit list)
  - [ ] Params: `--max-per-species 20 --source inaturalist`
  - [ ] Show progress with tqdm
- [ ] Fetch initial dataset — see `VISION_DATA_RETRIEVAL.md` for
      species selection strategy and target counts:
  - [ ] ~20 species covering all `hymenium_type` values initially
  - [ ] ~15-20 images per species
  - [ ] Target: ~300-400 images for initial pipeline validation
  - [ ] Then expand to ~100-150 species, ~1500-2000 images for
        full Tier 1 training

#### 1c: Auto-labeling from species DB

- [ ] Implement `vision/data/labeling.py`:
  - [ ] `auto_label_from_species(db_connection)`:
    - [ ] For each image in `image_registry` with a known species
    - [ ] Look up species features from `reconciled_species`
    - [ ] Normalize values to canonical vocabulary (using aliases
          from `morphological_vocabulary.yaml`)
    - [ ] Insert into `image_annotations` with
          `annotation_type='auto_from_species'`, `annotator='species_db'`
    - [ ] Skip null/missing features, don't invent labels
  - [ ] Idempotent: skip existing annotations
- [ ] Implement `vision/scripts/label_from_db.py` CLI
- [ ] Run on all fetched images

#### 1d: Preprocessing pipeline

- [ ] Implement `vision/data/preprocess.py`:
  - [ ] `ImagePreprocessor` class:
    - [ ] Constructor takes config YAML path, stores version
    - [ ] `process(raw_path, output_dir) → PreprocessResult`
    - [ ] `process_batch(raw_dir, output_dir) → list[PreprocessResult]`
    - [ ] Content-hash based naming (same IDs as raw)
    - [ ] Idempotent (skip already-processed)
  - [ ] `PreprocessResult` dataclass:
    `image_id, status, reject_reason, raw_size, processed_size, border_cropped`
  - [ ] Steps (all deterministic):
    1. Load image, convert to RGB
    2. Reject if corrupt or < 100px shortest side
    3. Detect/crop constant-color borders (std threshold on outer pixel rows)
    4. Resize shortest side → 256px, LANCZOS
    5. Save as JPEG q=95
  - [ ] Log rejections to `image_quality` table
- [ ] Implement `vision/scripts/run_preprocess.py` CLI:
  - [ ] `--config pipeline_configs/preprocess_v1.yaml`
  - [ ] `--input data/images/raw/ --output data/images/processed/v1/`
- [ ] Run on all fetched images

#### 1e: Embedding extraction

- [ ] Implement `vision/models/backbone.py`:
  - [ ] `FeatureExtractor` class:
    - [ ] Load `facebook/dinov2-base` from HuggingFace
    - [ ] Freeze all parameters
    - [ ] Property: `embed_dim → 768`
    - [ ] Forward: returns CLS token (batch of 768-d vectors)
- [ ] Implement `vision/models/embed.py`:
  - [ ] `EmbeddingExtractor` class:
    - [ ] Apply ImageNet normalization + CenterCrop(224) as torch transforms
    - [ ] `extract_single(image_path) → Tensor(768,)`
    - [ ] `extract_batch(processed_dir, output_dir, batch_size=32)`
    - [ ] MPS support, idempotent
    - [ ] `verify_cache(processed_dir, output_dir) → dict` with counts
          of total, cached, missing, orphaned
- [ ] Implement `vision/scripts/run_embed.py` CLI:
  - [ ] `--backbone facebook/dinov2-base`
  - [ ] `--input data/images/processed/v1/`
  - [ ] `--output data/embeddings/dinov2-base/processed-v1/`
- [ ] Run on all processed images

#### 1f: Pipeline status tool

- [ ] Implement `vision/scripts/pipeline_status.py`:
  - [ ] Count images at each stage: raw → processed → embedded
  - [ ] Count annotations per feature per annotation_type
  - [ ] Report class distribution per feature
  - [ ] Flag missing embeddings, orphaned files
  - [ ] Show rejection reasons summary from image_quality
  - [ ] Example output:
    ```
    === Pipeline Status ===
    Raw images:       1,847
    Processed (v1):   1,791 (56 rejected: 23 too_small, 33 corrupt)
    Embedded:         1,791 (dinov2-base/processed-v1)

    === Annotations ===
    hymenium_type:    1,791 images (auto_from_species)
      gills: 1,204 | pores: 287 | ridges: 98 | teeth: 64 | ...
    view_angle:       1,791 images (model_predicted, clip-zero-shot)
      side: 743 | top: 412 | underside: 298 | habitat: 201 | ...
    ```

---

### Phase 2: Train Tier 1 Morphological Heads (ungated)

**Goal:** Train the first four morphological feature classifiers on ALL
images (no context filtering). Evaluate whether weak supervision from
species DB is sufficient.

**Duration:** 3–5 days

#### 2a: Training infrastructure

- [ ] Implement `vision/data/manifest.py`:
  - [ ] `build_manifest(db_connection, feature_name, view_filter=None, ...) → DataFrame`
  - [ ] Join image_registry + image_annotations
  - [ ] Optional view/framing filters (unused initially, ready for later)
  - [ ] Export to parquet
- [ ] Implement `vision/data/dataset.py`:
  - [ ] `MushroomFeatureDataset(manifest_df, feature_name, embedding_cache_dir)`:
    - [ ] Filters manifest to rows with non-null labels for this feature
    - [ ] Loads cached embeddings from .pt files
    - [ ] Returns `(embedding, label_idx, image_id)`
    - [ ] Properties: `class_names`, `class_to_idx`, `num_classes`
- [ ] Implement `vision/data/splits.py`:
  - [ ] `stratified_species_split(manifest, feature_name, train=0.7, val=0.15, test=0.15)`:
    - [ ] **Split by species, not by image** — all images of one species
          go to same fold (prevents data leakage)
    - [ ] Stratify by feature value distribution
    - [ ] Return train/val/test DataFrames
- [ ] Implement `vision/models/heads.py`:
  - [ ] `FeatureHead(nn.Module)`:
    - [ ] `__init__(embed_dim, num_classes, dropout=0.1)`
    - [ ] Architecture: `Dropout → Linear(embed_dim, num_classes)`
    - [ ] Attributes: `feature_name`, `class_names`, `temperature` (for calibration)
    - [ ] `predict(embedding) → dict` using universal output contract
          (Section 3.2): class, confidence, status, runner_up
- [ ] Implement `vision/training/losses.py`:
  - [ ] `compute_class_weights(dataset) → Tensor` — inverse frequency
  - [ ] CrossEntropyLoss with configurable label_smoothing and class weights
- [ ] Implement `vision/training/metrics.py`:
  - [ ] `compute_metrics(preds, labels, class_names) → dict`:
        per-class accuracy, macro-F1, weighted-F1
  - [ ] `plot_confusion_matrix(preds, labels, class_names) → Figure`
  - [ ] `classification_report_str(preds, labels, class_names) → str`
- [ ] Implement `vision/tracking/mlflow_utils.py`:
  - [ ] `setup_experiment(feature_name, tier) → experiment_id`
  - [ ] `log_training_run(params, metrics, artifacts)`
  - [ ] Naming convention: `vision/morpho/{feature_name}`

#### 2b: Training loop

- [ ] Implement `vision/training/trainer.py`:
  - [ ] `train_head(feature_name, config) → TrainingResult`:
    1. Build manifest from DB (ungated)
    2. Stratified species split
    3. Load cached embeddings into MushroomFeatureDataset
    4. Compute class weights
    5. Initialize FeatureHead
    6. Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
    7. Loss: CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    8. Training loop (max 50 epochs):
       - Train one epoch on embeddings (fast — no image loading)
       - Validate: compute metrics
       - Log to MLflow (metrics per epoch)
       - Early stopping on val macro-F1 (patience=10)
    9. Save best model checkpoint
    10. Log confusion matrix + classification report as MLflow artifacts
    11. Log manifest parquet as artifact for reproducibility
- [ ] Implement `vision/scripts/train_head.py` CLI:
  - [ ] `--feature hymenium_type --preprocess-version v1`
  - [ ] `--device mps --lr 1e-3 --epochs 50`
- [ ] Implement `vision/scripts/train_all.py` CLI:
  - [ ] Train all features in a specified tier sequentially

#### 2c: Train and evaluate Tier 1 heads

- [ ] Train `hymenium_type` head. Target: >85% accuracy.
- [ ] Train `overall_body_form` head. Target: >80% accuracy.
- [ ] Train `cap_shape` head. Target: >75% accuracy.
- [ ] Train `cap_color` head. Target: >70% accuracy.
- [ ] For each: review confusion matrix, identify error patterns
- [ ] Compare all results against CLIP zero-shot baseline from Phase 0
- [ ] Implement `vision/scripts/evaluate_head.py` CLI
- [ ] Write `notebooks/03_training_analysis.ipynb`:
  - [ ] Learning curves, confusion matrices side by side
  - [ ] Per-class performance breakdown
  - [ ] Comparison table: CLIP zero-shot vs DINOv2 linear probe

#### 2d: Basic inference

- [ ] Implement `vision/inference/predict.py`:
  - [ ] `predict_image(image_path, heads, backbone) → dict[feature → prediction]`:
    1. Preprocess image (using same pipeline config)
    2. Extract DINOv2 embedding
    3. Run all active heads on the embedding
    4. Return dict of predictions with universal output contract
  - [ ] `predict_from_embedding(embedding, heads) → dict[feature → prediction]`
- [ ] Implement `vision/scripts/predict_image.py` CLI:
  - [ ] `--image path/to/photo.jpg --heads hymenium_type,cap_shape,cap_color`
  - [ ] Pretty-print predictions with confidence and status

---

### Phase 2b: Context Gating (only if Phase 2 error analysis shows need)

**Goal:** Add view-angle filtering to improve morphological head accuracy.
Only pursue if ungated heads show error patterns attributable to wrong
view angles.

**Trigger:** Phase 2 confusion matrices show systematic errors traceable
to view-angle issues.

- [ ] Train `view_angle` head using CLIP-bootstrapped labels from Phase 0c
      (stored in `image_annotations` with `annotator='clip-zero-shot'`)
- [ ] Evaluate view_angle head accuracy
- [ ] Implement `vision/inference/context.py`:
  - [ ] Run context heads on embedding → get view/framing predictions
  - [ ] Look up feature_applicability.yaml
  - [ ] Return which morpho heads to run and confidence threshold adjustments
  - [ ] Return `not_determinable` for inapplicable features
- [ ] Rebuild manifests with view-angle filters for each morpho feature
- [ ] Retrain morpho heads on filtered data
- [ ] Compare filtered vs ungated accuracy in MLflow
- [ ] Write `notebooks/04_error_analysis.ipynb`

---

### Phase 3: Confidence Calibration + Multi-View

**Goal:** Calibrate confidence scores. Handle multiple images of the
same specimen.

**Duration:** 3–4 days

- [ ] Implement `vision/inference/confidence.py`:
  - [ ] Temperature scaling per head (learned scalar on val set)
  - [ ] `calibrate_head(head, val_embeddings, val_labels) → temperature`
  - [ ] Reliability diagram: predicted confidence vs actual accuracy
  - [ ] Threshold tuning per head based on calibrated probabilities
  - [ ] Store calibration params with head checkpoint
- [ ] Implement `vision/inference/multi_view.py`:
  - [ ] `predict_specimen(images: list[Path], heads) → dict`:
    1. Preprocess + embed each image
    2. Optionally run context heads on each (if Phase 2b was done)
    3. For each morpho head: collect softmax from applicable images
    4. Average softmax distributions across views
    5. Apply confidence thresholds on aggregated predictions
    6. Return per-feature: prediction, confidence, status, contributing images

---

### Phase 4: Tier 2 Heads + Preprocessing Experiments

**Goal:** Expand feature coverage. Experiment with preprocessing
improvements where error analysis indicates need.

**Duration:** ongoing, incremental

- [ ] Fetch more images targeting Tier 2 feature diversity
- [ ] Train Tier 2 heads one at a time, evaluate
- [ ] If `cap_color` underperforms → create `preprocess_v2.yaml`
      with white balance correction, re-process, re-embed, compare
- [ ] If fine-detail heads underperform → try `preprocess_v3` with
      higher resolution (short_side=518)
- [ ] If any head consistently underperforms with linear probe →
      try 2-layer MLP head or LoRA on last 2 backbone blocks

---

### Phase 5: Integration with Lookalike Search

**Goal:** Connect visual feature extraction to the existing similarity
search pipeline.

- [ ] User uploads photo(s) of unknown mushroom
- [ ] Visual pipeline predicts features with confidence
- [ ] Feed predicted features (weighted by confidence) into existing
      pgvector similarity search
- [ ] Return ranked lookalikes with visual + textual feature comparison
- [ ] UI integration in Streamlit app

---

## 7. Training Recipe (per head)

Each head follows the same recipe:

```
1.  Build manifest:    query DB for images with this feature labeled
2.  Split:             stratified by species (not by image)
3.  Load:              cached 768-d embeddings for train/val sets
4.  Class weights:     inverse frequency from train set distribution
5.  Initialize:        FeatureHead(768, num_classes, dropout=0.1)
6.  Optimizer:         AdamW(lr=1e-3, weight_decay=1e-4)
7.  Loss:              CrossEntropyLoss(weight=weights, label_smoothing=0.1)
8.  Train:             max 50 epochs, early stopping on val macro-F1 (patience=10)
9.  Log:               MLflow — params, metrics/epoch, confusion matrix, model
10. Calibrate:         temperature scaling on val set (Phase 3)
11. Save:              head weights + class names + temperature + thresholds
```

Expected training time per head: seconds to low minutes on CPU/MPS.

---

## 8. MLflow Tracking Convention

```
Experiments:
  vision/morpho/hymenium_type
  vision/morpho/overall_body_form
  vision/morpho/cap_shape
  vision/morpho/cap_color
  vision/context/view_angle          (Phase 2b, if needed)
  ...

Per run, log:
  Params:
    feature_name, num_classes, tier
    preprocess_version, backbone, embedding_version
    n_train, n_val, n_test, class_distribution
    lr, weight_decay, dropout, label_smoothing
    manifest_hash, gated (true/false)

  Metrics (per epoch):
    train_loss, val_loss
    val_accuracy, val_macro_f1, val_weighted_f1

  Artifacts:
    confusion_matrix.png
    classification_report.txt
    manifest snapshot (.parquet)
    preprocess config (.yaml)
    best model checkpoint (.pt)
    calibration_plot.png (Phase 3)
```

---

## 9. Dependencies

```
# Core (most already in mushroom-ai requirements.txt)
torch >= 2.1              # MPS support
torchvision >= 0.16       # transforms v2
transformers >= 4.36      # DINOv2 + CLIP from HuggingFace
Pillow                    # image I/O

# Data
requests                  # image fetching (iNaturalist API)
tqdm                      # progress bars
pandas                    # manifest handling
pyarrow                   # parquet export

# Tracking
mlflow                    # already in stack

# Evaluation
scikit-learn              # classification_report, confusion_matrix
matplotlib                # plots
seaborn                   # confusion matrix heatmaps
```

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Weak labels noisy (image shows top but label is about underside) | Train ungated first; add context gating only if error analysis shows need |
| Class imbalance (gills dominate hymenium_type) | Class weights in loss; stratified splits; per-class metrics (not just accuracy) |
| DINOv2 features may miss fine mycological details | Fallback: LoRA on last 2 blocks; or higher-res preprocessing |
| Mixed image quality degrades performance | Training-time augmentation; preprocessing versioning for experiments |
| Color features unreliable across lighting | Preprocessing v2 with white balance; strong ColorJitter augmentation |
| Some features not visible in photo | `uncertain`/`not_determinable` are first-class outputs, not errors |
| Species-level label leakage between train/test | Split by species, not by image — all images of one species in same fold |
| iNaturalist rate limiting | 1s delay between requests; cache aggressively; fetch in batches |

---

## 11. Quick-Start Checklist

When starting implementation:

1. Create `vision/` directory in mushroom-ai repo
2. Drop this file as `vision/VISION_PROJECT_PLAN.md`
3. Drop data retrieval spec as `vision/VISION_DATA_RETRIEVAL.md`
4. Add pointer in `CLAUDE.md`:
   ```markdown
   ## Vision Module
   See vision/VISION_PROJECT_PLAN.md for the visual feature extraction
   pipeline plan. See vision/VISION_DATA_RETRIEVAL.md for the image
   fetching specification.
   ```
5. Start Phase 0a: create module skeleton + configs
6. Start Phase 0b: CLIP baseline notebook
7. Start Phase 1: data pipeline (can parallelize 1a-1d)
8. First end-to-end milestone: fetch 300 images → preprocess → embed →
   train `hymenium_type` head → evaluate → compare to CLIP baseline

---

## 12. Future Ideas

- **VLM to extract visible features.** Use more modern/large VLM
  setup with appropriate prompting and visual examples of what each feature
  looks like to determine if some mushroom has it or not clear from image.
  This goes more extending above plan against using more intelligent and
  general purpose tools for specialised applications. Aside feasibility,
  question is here: would it be more or less accurate wrt a DINOv2 setup?
