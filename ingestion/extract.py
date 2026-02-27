"""
LLM-based feature extraction from source text.

Takes raw text about a mushroom species and extracts structured features
using Instructor-validated LLM calls.

Pipeline:
    Source text → LLM (with ExtractedSpeciesFeatures schema) → validated features
    → store as Layer 1 (SourceObservation) in Postgres
"""

import hashlib
import logging

from config import settings
from db.connection import get_session
from db.models import SourceObservation
from llm.client import structured_completion
from llm.schemas import ExtractedSpeciesFeatures

logger = logging.getLogger(__name__)

# Truncate source text to keep prompts within a safe token budget.
# gemma3:27b has 8192 max output tokens; 8000 chars ≈ ~2000 tokens of input, leaving
# plenty of room for the structured output.
_MAX_TEXT_CHARS = 8_000

SYSTEM_PROMPT = """\
You are a mycologist extracting structured features from species descriptions.

GENERAL RULES
- Extract ONLY what is explicitly stated in the source text. Never guess or infer.
- Use null for any feature not mentioned — omission is better than a wrong value.
- Record ambiguity or conflicting information in extraction_notes.
- scientific_name must match the queried species name exactly.
- For list fields, include every value mentioned in the text.

LANGUAGE
- ALL values MUST be in English — translate any Italian, French, German, or other
  foreign-language terms before writing them.
- Color examples: giallo → yellow, rosso → red, bruno → brown, nero → black,
  bianco/biancastro/biancastra → white/whitish, verde → green, arancione → orange,
  vinoso → wine-colored, lilacino/lilla → lilac, ocra → ochre, grigio → grey,
  rosa → pink, viola → violet, beige → beige, crema → cream.
- Compound color forms must also be translated: "vinoso-bruno" → "wine-brown",
  "rosa-lilacino" → "rose-lilac", "giallo-brunastro" → "yellowish-brown",
  "rosso-arancio" → "red-orange", "giallo-ocra" → "yellow-ochre".
- Non-color Italian words to translate: nullo/assente → none/absent,
  dolciastro → sweetish, soda/sodo → firm, acre/acro → acrid,
  farinoso → mealy/farinaceous, fruttato → fruity, mite → mild.
- Tree names to translate: Abete rosso → Norway spruce, Abete bianco → silver fir,
  Faggio → beech, Castagno → chestnut, Quercia → oak, Pino → pine,
  Betulla → birch, Leccio → holm oak, Carpino → hornbeam.
- Do NOT copy foreign words verbatim, even partially (no "biancastra", "dolciastro",
  "Abete rosso", "Faggio").

FIELD-BY-FIELD GUIDANCE (with allowed values / examples):

CAP
- shape: convex, broadly convex, flat, depressed, umbonate, conical, bell-shaped, irregular
- colors: list of colors IN ENGLISH when fresh, e.g. ["scarlet red", "orange", "yellow-orange"]
  (translate from source language if needed)
- color_faded: color when dry/faded IN ENGLISH — e.g. buff, pale ochraceous, brownish
- hygrophanous: true if cap changes color markedly as it dries
- color_pattern: uniform | darker at center | two-toned | mottled | streaked
- surface_texture: smooth, slimy/viscid, dry, velvety, fibrous, hairy-scaly, powdery, matt
- margin_type: inrolled, wavy, even, striate/lined, fringed, lobed
- margin_lined_at_maturity: true if margin becomes striate with age
- central_depression: true if cap becomes funnel-shaped or depressed at center

HYMENIUM
- type: MUST be one of: gills | pores | teeth | ridges | smooth
  Use "ridges" for species with forking/blunt ridges or "false gills" that run down
  the stem and fork repeatedly (e.g. Cantharellus, Craterellus). These are NOT true
  gills — they are blunt, vein-like, and cannot be separated from the cap flesh.
  Use "gills" only for true blade-like gills that can be cleanly separated.
  Use "smooth" for species with no distinct hymenophore structure (e.g. Craterellus
  cornucopioides interior, puffballs).

GILLS (only when hymenium.type == "gills")
- attachment: free, adnate, decurrent, sinuate, adnexed
- spacing: crowded, close, subdistant, distant
- color_with_age: e.g. "white to pink then brown", "yellow becoming rusty"
- thickness: thin | thick (thick gills are diagnostic e.g. for Laccaria)
- texture: waxy (Hygrocybe), brittle (Russula/Lactarius), normal/soft

PORES (only when hymenium.type == "pores")
- color: pore surface color when fresh
- bruising_color: "blue" (bluing), "slowly orangish-brown", "none"
- density_per_mm: e.g. "1-2 per mm", "3-4 per mm"

STEM
- reticulation: none | partial (upper stem only) | full — key for boletes
- shape: equal, club-shaped (clavate), tapered base, bulbous, swollen base
- hollow_or_solid: hollow | stuffed | solid
- basal_mycelium_color: color of mycelium at base — e.g. white, lilac (diagnostic Laccaria), yellow
- finger_stain_color: color left on fingers when rubbed — e.g. yellow (Retiboletus ornatipes)

VEIL
- type: partial | universal | cortina (cobweb-like) | absent
- cortina_present: true if a cortina (cobweb partial veil) is present — key Cortinarius feature

FLESH
- taste: mild, bitter, acrid/peppery (Russula/Lactarius), farinaceous/mealy, not distinctive
- texture: firm, soft, brittle (snaps cleanly — Russula), insubstantial, watery
- quantity: insubstantial/thin | moderate | thick

SPORE PRINT
- spore_print_color: white, pink, brown, purple-brown, black, rusty-brown, olive, yellow

SPORES (microscopic — extract if mentioned)
- shape: globose, ellipsoid, subfusoid, amygdaliform, cylindrical
- ornamentation: smooth | echinulate (spiny) | warty | reticulate | striate
- amyloidity: amyloid | inamyloid | dextrinoid

MICROSCOPIC (extract if present in source)
- basidia_spore_count: 4-spored | 2-spored | mixed
- pileipellis_type: cutis | trichoderm | ixocutis | hymeniderm

CHEMICAL REACTIONS (extract if present)
- KOH_cap / KOH_flesh: yellow, orange, red, negative, blackening
- FeSO4_cap / FeSO4_flesh: blue-green, pink, grey-green, negative

ECOLOGY
- trophic_mode: mycorrhizal | saprotrophic | parasitic
- growth_pattern: solitary | scattered | gregarious | clustered | trooping
- growth_position: terrestrial | lignicolous (on wood) | coprophilous (on dung)

SAFETY
- edibility_status: MUST be one of: edible, choice, conditionally edible, inedible, toxic, deadly
- known_toxins: named toxins only e.g. ["amatoxins", "ibotenic acid", "muscimol", "gyromitrin"]
"""


def extract_features_from_text(
    scientific_name: str,
    text: str,
    source_name: str = "unknown",
) -> ExtractedSpeciesFeatures:
    """Extract structured features from source text using the LLM."""
    prompt = (
        f"Extract mushroom features for '{scientific_name}' from the following text.\n"
        f"Source: {source_name}\n\n"
        f"{text[:_MAX_TEXT_CHARS]}"
    )
    return structured_completion(
        prompt=prompt,
        response_model=ExtractedSpeciesFeatures,
        system=SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=4096,
    )


def save_extraction(
    features: ExtractedSpeciesFeatures,
    source_url: str,
    source_text: str,
    source_name: str,
) -> SourceObservation:
    """
    Upsert extraction result into source_observations (Layer 1).

    If a row for (scientific_name, source_name) already exists,
    its features_json and metadata are updated (upsert policy).
    """
    text_hash = hashlib.sha256(source_text.encode()).hexdigest()
    features_dict = features.model_dump()
    model_string = f"{settings.llm_provider}/{settings.llm_model}"

    session = get_session()
    try:
        existing = (
            session.query(SourceObservation)
            .filter_by(scientific_name=features.scientific_name, source_name=source_name)
            .first()
        )
        if existing:
            existing.features_json = features_dict
            existing.source_url = source_url
            existing.source_text_hash = text_hash
            existing.extraction_model = model_string
            existing.extraction_notes = features.extraction_notes
            obs = existing
        else:
            obs = SourceObservation(
                scientific_name=features.scientific_name,
                source_name=source_name,
                source_url=source_url,
                source_text_hash=text_hash,
                features_json=features_dict,
                extraction_model=model_string,
                extraction_notes=features.extraction_notes,
            )
            session.add(obs)
        session.commit()
        session.refresh(obs)
        return obs
    finally:
        session.close()
