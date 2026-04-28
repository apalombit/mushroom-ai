"""Per-feature system and user prompts for VLM extraction.

Prompts include rich textual class descriptions sourced from
morphological_vocabulary.yaml and explicit confidence-level definitions.
Each feature has a system prompt (shared structure) and a user prompt
(feature-specific class descriptions + visual guidance).
"""

# ---------------------------------------------------------------------------
# Shared confidence block (injected into every system prompt)
# ---------------------------------------------------------------------------

_CONFIDENCE_BLOCK = """\
Confidence levels:
- high: the feature is clearly and unambiguously visible. You would bet on this answer.
- low: visible but partially obscured, ambiguous angle, or borderline between classes. \
Return your best guess but flag the uncertainty.
- cannot_tell: the image does not show this feature reliably enough to classify. \
Set the feature value to null. This is the correct and expected answer for many \
images — an honest null is always better than a confident wrong answer."""

# ---------------------------------------------------------------------------
# hymenium_type
# ---------------------------------------------------------------------------

HYMENIUM_TYPE_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "Focus ONLY on the spore-bearing surface (hymenium) visible in this image.\n"
    "Report only what you can directly observe — do not infer from species knowledge.\n\n"
    + _CONFIDENCE_BLOCK
)

HYMENIUM_TYPE_USER = """\
Examine this mushroom image and identify the hymenium type (spore-bearing surface).

First describe the structure, spacing, and texture of whatever fertile surface is \
visible, then classify.

If the underside/fertile surface is NOT visible at all (e.g., only a top-down cap \
view, only the stem, or only the outer skin of a closed fruiting body), set \
confidence=cannot_tell and hymenium_type=null. \
If you can see the fertile surface but it is blurry, distant, or partly obscured, \
still attempt a classification with confidence=low.

Hymenium types (listed rarest first — do not let order bias your choice; pick the \
option whose description best matches what you actually see):

- gleba: An enclosed internal spore mass, not on an outer surface. Visible when \
a puffball, earthstar, or stinkhorn is cut open or has ruptured. If you see a \
round, closed body without visible gills/pores/teeth on the outside, it is likely \
gleba — set confidence=cannot_tell if the interior is not visible.

- alveolate: Deep pits or large honeycomb-like depressions forming the fertile \
surface. The cavities are large, conspicuous, and irregularly shaped — much bigger \
than pores. Typical of morels (Morchella) and false morels (Gyromitra). If the \
surface looks like a sponge with large open cavities or a honeycomb, choose alveolate.

- smooth: A flat, completely undifferentiated fertile surface with no gills, pores, \
teeth, or ridges. No structures of any kind — entirely featureless. Seen in \
corticioid/resupinate crusts and some club or coral fungi.

- teeth: Downward-hanging spines or tooth-like projections — individual conical or \
cylindrical structures pointing downward. Can be densely packed (Hydnum hedgehog \
mushrooms) or long and branching (Hericium/lion's mane). Even when densely packed, \
each tooth is a separate pointed projection hanging freely, unlike pores which are \
openings in a flat surface.

- ridges: Blunt, shallow, vein-like folds that are part of the flesh itself — they \
merge seamlessly into the cap AND continue down the stem (decurrent attachment). \
Ridges cannot be peeled off. Key features that distinguish ridges from gills: \
(1) rounded/blunt edges, not sharp knife-like edges; (2) irregular forking and \
rejoining pattern with cross-veins connecting adjacent folds; (3) they run DOWN \
the stem surface, not stopping at the cap-stem junction. Typical of chanterelles \
(Cantharellus), trumpet chanterelles (Craterellus), and pig's ear (Gomphus).

- pores: A sponge-like surface with many small round openings — the mouths of \
vertical tubes packed together. The surface appears relatively flat with a pattern \
of tiny holes. Seen on the underside of boletes, polypores, and bracket fungi. \
Pores can be very fine (almost invisible) or coarse.

- gills: Thin, blade-like lamellae radiating outward from the stem. Each gill is a \
separate structure with a sharp, knife-like edge that can be peeled away from the \
cap flesh. Gills are typically uniform in thickness, evenly spaced, and run parallel \
to each other. Crucially, gills usually STOP at or near the stem — they are free, \
adnate, or adnexed but do NOT run down the stem.

Key distinctions:

- GILLS vs RIDGES (most important): Check THREE features:
  1. STEM ATTACHMENT: Do the folds run DOWN the stem (decurrent)? → ridges. \
Do they stop at or near the cap-stem junction? → gills.
  2. EDGE SHAPE: Are the edges sharp and knife-like with distinct thin blades? \
→ gills. Are they blunt, rounded, and shallow folds? → ridges.
  3. FORKING PATTERN: Are the folds uniform, parallel, and evenly spaced? → gills. \
Do they fork and rejoin irregularly with cross-veins between them? → ridges.
  If TWO of these three point to ridges, choose ridges. Chanterelle-like mushrooms \
with folds running down the stem are almost always ridges, not gills.

- PORES vs TEETH: If the structures are individual pointed spines hanging \
downward, choose teeth — even if tightly packed. If the surface is flat with \
holes (tube openings), choose pores. Fine teeth can resemble pores at low \
resolution — look for individual pointed structures rather than a flat surface.

- PORES vs ALVEOLATE: Pores are tiny tube openings in a relatively flat surface. \
Alveolate has large, deep, conspicuous pits (like a sponge or honeycomb). Morels \
are always alveolate, never pores.

- PORES vs SMOOTH: If the surface has ANY pattern of holes, however tiny, it is \
pores. Smooth means completely featureless — no structures at all.

- TEETH vs GILLS: Teeth hang downward as individual spines; gills are flat blades \
arranged radially. If the structures are pointed and separate, choose teeth.

Final answer rule (multiple-choice mode):
Choose the option whose description most accurately matches what you observe. \
Your hymenium_type value must be exactly one of: \
gleba | alveolate | smooth | teeth | ridges | pores | gills | null. \
Do not default to the most common class (gills) when the visual evidence does not \
clearly support it — when in doubt and the structure is not gill-like, prefer \
confidence=low or cannot_tell over a gills guess."""

# ---------------------------------------------------------------------------
# cap_color
# ---------------------------------------------------------------------------

CAP_COLOR_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "Focus ONLY on the cap (pileus) surface color visible in this image.\n"
    "Report only what you can directly observe — do not infer from species knowledge.\n\n"
    + _CONFIDENCE_BLOCK
)

CAP_COLOR_USER = """\
Examine this mushroom image and identify the BASE cap (pileus) color.

First describe the colors you observe on the cap surface, noting any gradients, \
zones, or lighting conditions. Then classify to the closest canonical color.

Important guidance:
- Report the BASE cap color visible across the majority of the surface.
- If the cap has darker scales, warts, spots, or fibrils over a lighter ground \
color, report the GROUND color (e.g., a white cap with brown scales → white).
- If the cap shows a gradient (darker center, paler margin), classify by the \
color covering the LARGEST area.

If the cap is not visible, or is in deep shadow/overexposed/wet-reflective making \
color unreliable, set confidence=cannot_tell and cap_color=null.

Color palette (listed rarest first — do not let order bias your choice; pick the \
single closest match to what you actually see):
- blue: Blue, indigo
- green: True green, verdigris
- purple: Violet, lilac, lavender
- olive: Olive green, greenish-brown
- pink: Light red, salmon, rose, flesh-colored
- black: Black, very dark brown-black
- red: Bright red, scarlet, vermillion, crimson
- orange: True orange, amber, apricot, tawny
- grey: Neutral grey, ash, silvery
- yellow: Clear yellow, lemon, golden, straw
- white: White, off-white, ivory, cream, pale buff — any very pale color without a strong warm tint
- brown: All shades of brown — light sandy/tan through dark chestnut, ochre, warm earthy tones

Final answer rule (multiple-choice mode):
Choose the color whose description most accurately matches the dominant cap surface. \
Your cap_color value must be exactly one of: \
blue | green | purple | olive | pink | black | red | orange | grey | yellow | white | brown | null. \
Brown is the most common class — only choose brown when the cap is genuinely brown; \
if it is more accurately described as grey, yellow, or another color, pick that one."""

# ---------------------------------------------------------------------------
# Prompt registry — maps feature name → (system_prompt, user_prompt)
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: dict[str, tuple[str, str]] = {
    "hymenium_type": (HYMENIUM_TYPE_SYSTEM, HYMENIUM_TYPE_USER),
    "cap_color": (CAP_COLOR_SYSTEM, CAP_COLOR_USER),
}
