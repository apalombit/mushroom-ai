# Data Retrieval Specification

## Strategy

The existing `reconciled_species` table already maps each species to its
morphological features. Use this as the seed list: for each feature value,
look up which species have that value, then fetch images for those species.

### Step 1: Generate species shopping lists from DB

```sql
-- Example: which species have which hymenium_type?
SELECT
    (features->>'hymenium_type') AS hymenium_type,
    species_name,
    family
FROM reconciled_species
WHERE features->>'hymenium_type' IS NOT NULL
ORDER BY hymenium_type, family;
```

Run this for each Tier 1 feature. The output is a mapping:
`feature_value → [list of species]`

For each value, select 5-15 species (more for common values like "gills",
fewer for rare values like "alveolate"). Prefer diversity across genera
and families — don't fetch 15 Russula species for "gills" when you could
spread across Amanita, Russula, Lactarius, Tricholoma, etc.

### Step 2: Fetch images from iNaturalist

**Primary source: iNaturalist API v1**

Base URL: `https://api.inaturalist.org/v1/`

Key endpoints:
- `GET /observations` — search observations by taxon name
- `GET /taxa` — look up taxon_id from species name

Workflow per species:
```
1. Look up taxon_id:
   GET /taxa?q=Boletus+edulis&rank=species
   → extract results[0].id → e.g. 48701

2. Fetch research-grade observations with photos:
   GET /observations?taxon_id=48701
       &quality_grade=research
       &photos=true
       &per_page=30
       &order=desc
       &order_by=votes    (prefer well-voted observations)
       &place_id=97403    (optional: Europe, to match your species DB)

3. Extract photo URLs from response:
   Each observation has observation_photos[].photo.url
   URL pattern: replace "square" with "medium" or "large" in the URL
     square: https://inaturalist-open-data.s3...square.jpg (75×75)
     medium: .../medium.jpg (~500px)
     large:  .../large.jpg  (~1024px)
   → Use "large" for best quality

4. Download images, name by content hash, store to data/images/raw/

5. Record in image_registry:
   image_id, species, source='inaturalist',
   source_id=observation.id, source_url=observation.uri,
   license=photo.license_code
```

**Rate limiting:** iNaturalist allows 60 requests/minute for
unauthenticated, 120/minute with API token. Add 1s delay between
requests to be safe. For bulk fetching, consider using the
iNaturalist export/CSV tools instead.

**Licensing filter:** Only fetch images with open licenses:
`&license=cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc0`

### Step 3: Auto-label from species DB

Once images are fetched and registered, run `label_from_db.py`:

```python
# For each image in image_registry with a known species:
species_features = get_reconciled_features(species_name)

for feature_name, feature_value in species_features.items():
    if feature_value is not None:
        insert_annotation(
            image_id=image_id,
            annotation_type='auto_from_species',
            feature_name=feature_name,
            feature_value=normalize_to_canonical(feature_value),
            annotator='species_db',
        )
```

This gives every fetched image labels for ALL features that species has
in the database — not just the feature you fetched it for.


## Target Counts (Phase 1)

### Context heads (DEFERRED — only if Phase 2b is triggered)

View-angle labels are bootstrapped automatically via CLIP zero-shot
in Phase 0c — no manual annotation needed to get started. If Phase 2
error analysis shows context gating would help (Phase 2b), a DINOv2
view_angle head is trained on the CLIP-bootstrapped labels. Manual
annotation of ~50-100 images for spot-checking CLIP accuracy is
sufficient.

| Head | Target | Notes |
|------|--------|-------|
| `view_angle` | CLIP-bootstrapped, spot-check ~50 manually | Phase 0c handles this automatically |
| `framing` | CLIP-bootstrapped or deferred | Add if needed |
| `subject_count` | Deferred | Add if error analysis shows need |

### Tier 1 morphological heads (auto-labeled from species DB)

| Feature | Target species per value | Images per species | Total estimate |
|---------|------------------------|--------------------|----------------|
| `hymenium_type` (7 classes) | 5-15 | 15-20 | ~700 |
| `overall_body_form` (11 classes) | 5-10 | 15-20 | ~800 |
| `cap_shape` (11 classes) | 5-10 | 15-20 | ~800 |
| `cap_color` (14 classes) | 5-10 | 15-20 | ~1000 |

Total: ~3000-4000 unique images (with significant overlap — one image
of Boletus edulis serves hymenium_type, body_form, cap_shape, AND cap_color).

Realistic unique image count after dedup: ~1500-2000 images covering
~100-150 species.


## Species Selection Guidance

When selecting which species to fetch for each feature value, prefer:

1. **Species already in your 543-species DB** (labels are already there)
2. **Common, well-photographed species** (more images on iNaturalist)
3. **Diversity across genera** (prevents the model from learning genus
   appearance instead of the actual feature)
4. **Species present in northern Italy / Alpine region** (matches your
   project focus)

For rare feature values (e.g., `hymenium_type: alveolate` = basically
just Morchella and Gyromitra), fetch all available species with that
value. For dominant values (e.g., `hymenium_type: gills`), sample
broadly across families — Agaricaceae, Amanitaceae, Russulaceae,
Tricholomataceae, Cortinariaceae, etc.


## Fallback Sources

If iNaturalist doesn't have enough images for a species:

1. **Mushroom Observer** — `https://mushroomobserver.org/api2/`
   Similar API structure. Smaller but high quality.

2. **GBIF** — `https://api.gbif.org/v1/`
   Aggregates from many sources. Use `mediaType=StillImage` filter.

3. **Wikimedia Commons** — search via MediaWiki API
   `https://commons.wikimedia.org/w/api.php`
   Good for curated reference photos, fewer field observations.
