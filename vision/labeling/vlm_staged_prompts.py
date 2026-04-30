"""Staged (two-stage) per-feature VLM prompts (Path C).

One stage-1 prompt that picks a coarse family, plus one stage-2 prompt per
family that does within-family disambiguation. Stage-2 prompts are
deliberately SHORT and focused on the discriminating cues for their family
members — they do NOT re-include the full single-shot class enumeration,
which is the whole point of staging.

Tone differs from the single-shot prompts: anti-bias is softened (only at
stage 1) because stage-2 has 2-3 classes and the model can afford to commit.
"""

# ---------------------------------------------------------------------------
# Shared confidence block — same as single-shot
# ---------------------------------------------------------------------------

_CONFIDENCE_BLOCK = """\
Confidence levels:
- high: the feature is clearly and unambiguously visible. You would bet on this answer.
- low: visible but partially obscured, ambiguous angle, or borderline between classes. \
Return your best guess but flag the uncertainty.
- cannot_tell: the image does not show this feature reliably enough to classify. \
Set the classification value to null. This is the correct and expected answer for many \
images — an honest null is always better than a confident wrong answer."""


# ===========================================================================
# Hymenium — stage 1: coarse family classification
# ===========================================================================

HYMENIUM_STAGE1_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "Identify the COARSE TEXTURE of the spore-bearing surface (hymenium).\n"
    "Report only what you can directly observe — do not infer from species knowledge.\n\n"
    + _CONFIDENCE_BLOCK
)


HYMENIUM_STAGE1_USER = """\
Examine this mushroom image and identify the COARSE TEXTURE FAMILY of the \
spore-bearing surface.

First describe what you see on the underside (if visible), then choose the family.

If the underside / fertile surface is NOT visible at all (e.g., only a top-down cap \
view, only the stem, or only the outer skin of a closed fruiting body), set \
confidence=cannot_tell and family=null.

Texture families:

- linear_radial: Long, narrow, radiating structures arranged like spokes — sharp \
blades or blunt folds running outward from the stem. Includes both gills (sharp \
blades) and ridges (blunt folds). Choose this whenever the dominant pattern is \
line-like structures running radially.

- punctate: Discrete units packed close together — many small openings, individual \
spines hanging down, or large honeycomb-like pits. Includes pores (tube openings), \
teeth (hanging spines), and alveolate (large cavities). Choose this whenever the \
dominant pattern is dot-like, spine-like, or pit-like discrete units rather than \
radiating lines.

- featureless_or_internal: No visible discrete structures at all (a flat, blank \
fertile surface with no holes, spines, pits, or radial lines), OR an internal \
spore mass visible because the body has been cut open or ruptured.

Important guidance:
- Choose linear_radial when you see long structures running outward from the stem, \
whether sharp-edged or blunt — stage 2 will decide gills vs ridges.
- Choose punctate when you see ANY discrete units (holes, spines, or pits), even \
if you cannot tell which type — stage 2 will decide pores vs teeth vs alveolate. \
A textured surface with discrete structures should always be punctate, not featureless.
- Choose featureless_or_internal ONLY when the visible surface is genuinely flat \
with NO discrete structures at all, OR when you can see the inside of a torn / \
sliced / ruptured body (gleba). Do NOT pick this family just because you are \
uncertain whether a textured surface is line-like or unit-like.
- Family classification is easier than identifying the exact class — commit at \
confidence=low rather than abstaining whenever any fertile surface is visible. \
Only set cannot_tell when the underside / fertile surface is not visible at all.

Your family value must be exactly one of: \
linear_radial | punctate | featureless_or_internal | null.
"""


# ===========================================================================
# Hymenium — stage 2 / linear_radial: gills vs ridges
# ===========================================================================

HYMENIUM_STAGE2_LINEAR_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "The spore-bearing surface shows long radiating structures (blade-like or fold-like).\n"
    "Decide whether they are gills or ridges based on three specific cues.\n\n"
    + _CONFIDENCE_BLOCK
)


HYMENIUM_STAGE2_LINEAR_USER = """\
The underside of this mushroom shows long radiating structures. Decide: gills or ridges?

Examine these THREE cues in order, then commit:

1. STEM ATTACHMENT — Do the structures run DOWN the stem surface (decurrent), or \
stop at / near the cap-stem junction?
   - Run DOWN the stem → suggests ridges
   - Stop at the stem → suggests gills

2. EDGE SHAPE — Are the edges sharp and knife-like (thin distinct blades), or \
blunt and rounded (shallow folds)?
   - Sharp knife-like → gills
   - Blunt rounded → ridges

3. FORKING PATTERN — Are the structures uniform, parallel, evenly spaced? Or do \
they fork irregularly with cross-veins between adjacent folds?
   - Uniform parallel → gills
   - Irregular forking with cross-veins → ridges

Commitment rule (asymmetric — gills is the much more common class):
- DEFAULT to gills unless you have STRONG evidence for ridges.
- Choose ridges with confidence=high only when ALL THREE cues clearly align: \
folds running DOWN the stem (decurrent), blunt rounded edges, AND irregular \
forking with cross-veins. This is the chanterelle pattern.
- Choose ridges with confidence=low when at least the stem-attachment cue is \
clearly decurrent AND edges look blunt rather than sharp (2/3 cues, with stem \
attachment as a mandatory cue).
- If you only see ONE ridges cue (e.g., some forking but with sharp edges and \
no decurrent attachment), choose gills — it is far more likely.
- Sharp-edged, parallel, blade-like structures should be gills even when there \
is some forking near the cap-stem junction (this is a common gill pattern, not ridges).

Only set hymenium_type=null and confidence=cannot_tell if the surface is too \
obscured to score the cues at all.

Your hymenium_type value must be exactly one of: gills | ridges | null.
"""


# ===========================================================================
# Hymenium — stage 2 / punctate: pores vs teeth vs alveolate
# ===========================================================================

HYMENIUM_STAGE2_PUNCTATE_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "The spore-bearing surface shows discrete punctate structures.\n"
    "Decide whether they are pores, teeth, or alveolate cavities.\n\n"
    + _CONFIDENCE_BLOCK
)


HYMENIUM_STAGE2_PUNCTATE_USER = """\
The underside or fertile surface shows discrete punctate structures. \
Decide: pores, teeth, or alveolate?

Three options, defined by physical relief:

- pores: 2-D openings in a relatively FLAT surface. The structures are HOLES — \
darker spots where vertical tubes open downward. The surface itself is flat; only \
the color differs between hole and not-hole. Typical of boletes, polypores, \
bracket fungi.

- teeth: 3-D PROJECTIONS hanging downward. The structures stick OUT from the \
surface — individual conical or cylindrical points. They CAST SHADOWS underneath \
and have visible depth/relief. Typical of Hydnum (hedgehog), Hericium (lion's mane).

- alveolate: LARGE, conspicuous, irregular pits or honeycomb-like depressions. \
The cavities are much bigger than pores (millimeters to centimeters across, not \
tiny dots). Typical of morels (Morchella) and false morels (Gyromitra).

Critical tiebreaker — pores vs teeth (the most common confusion):
- Holes are 2-D — flat surface with darker spots in it.
- Spines are 3-D — they cast shadows and have visible depth.
- IF YOU CAN SEE SHADOWS UNDER THE STRUCTURES, OR ANY 3-D RELIEF / DEPTH, choose teeth.
- Densely-packed teeth can resemble pores from above; the shadow / depth cue is the \
disambiguator. Look for highlights on the tips of structures — that means 3-D, so teeth.

Pores vs alveolate cue:
- Pores are TINY (under ~1 mm) and uniform.
- Alveolate cavities are LARGE and IRREGULAR — visibly honeycomb-like, much wider than \
pores. If a single unit is roughly 1/5 to 1/20 of the cap width, that is alveolate; \
pores would be 1/100 or smaller.

Only set hymenium_type=null and confidence=cannot_tell if the surface is too \
obscured to apply these cues. With ambiguous-but-visible cases, pick the more \
likely answer at confidence=low — staying committed is preferable to abstaining.

Your hymenium_type value must be exactly one of: pores | teeth | alveolate | null.
"""


# ===========================================================================
# Hymenium — stage 2 / featureless_or_internal: smooth vs gleba
# ===========================================================================

HYMENIUM_STAGE2_FEATURELESS_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "The fertile surface shows no discrete structures, OR an internal spore mass.\n"
    "Decide whether the visible surface is smooth (external featureless) or "
    "gleba (internal spore mass).\n\n"
    + _CONFIDENCE_BLOCK
)


HYMENIUM_STAGE2_FEATURELESS_USER = """\
This mushroom shows no discrete fertile-surface structures. Decide: smooth or gleba?

Two options:

- smooth: An external fertile surface that is completely flat and featureless — \
a crust, a club, a coral arm. The surface is on the OUTSIDE of an intact body. \
Typical of corticioid / resupinate fungi and some club or coral fungi.

- gleba: An internal spore mass exposed because the body has been cut open or \
has ruptured. The visible content is the INSIDE of a puffball, earthstar, \
stinkhorn, or similar. Look for: a powdery / spongy / olive-brown mass inside a \
torn or sliced wall.

Distinguishing cue: do you see the OUTSIDE of an intact mushroom (smooth) or the \
INSIDE of a cut / ruptured body (gleba)?
- Body looks closed / intact, surface is plain → smooth.
- Mass is visible behind a torn skin or inside an open cup → gleba.

If you genuinely cannot tell whether the body is intact or opened, set \
confidence=cannot_tell and hymenium_type=null.

Your hymenium_type value must be exactly one of: smooth | gleba | null.
"""
