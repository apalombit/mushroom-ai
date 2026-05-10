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
# ring_presence
# ---------------------------------------------------------------------------

RING_PRESENCE_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "Focus ONLY on the upper portion of the stem (stipe), where a ring (annulus) "
    "would attach.\n"
    "Report only what you can directly observe — do not infer from species knowledge.\n\n"
    + _CONFIDENCE_BLOCK
)

RING_PRESENCE_USER = """\
Examine this mushroom image and decide whether a RING (annulus) is present on the \
upper stem.

First describe the upper stem region — anything wrapping the stipe just below the \
cap, any darker zone or hanging tissue. Then classify.

If the upper portion of the stem is NOT visible at all (e.g., only a top-down cap \
view, only the underside, or the stem is fully obscured), set \
confidence=cannot_tell and ring_presence=null.

What counts as a RING:
- A membranous skirt, collar, or hanging tissue attached to the upper stem — the \
classic annulus. Clear ring → present at high confidence.
- A double-layered or cog-wheel ring → present at high confidence.
- A faint band of darker, fibrillose, or differently-colored tissue around the \
stem where a ring once was (a "ring zone") → present at low confidence. Even a \
faded ring counts.
- Cobweb-like silky fibers stretching from the cap margin to the stem, OR a rusty \
fibrous band on the upper stem (cortina or its remnant) → present at low \
confidence.

What does NOT count as a ring (anti-traps):
- Scales, fibrils, or color zones spread along the WHOLE length of the stem are \
stem-surface texture, not a ring.
- A swollen base or bulb at the bottom of the stem is not a ring (that is a base \
feature, judged separately).
- Universal-veil patches stuck to the cap are not a ring.
- Soil, leaves, or debris stuck to the stem are not a ring.

Options (listed rarest first — do not let order bias your choice; pick the option \
whose description best matches what you actually see):

- present: A ring, ring-zone, or cortina/cortina-zone is visible on the upper stem.
- absent: The upper stem is clearly visible and bare — no ring, no zone, no \
fibrous band, no cortina remnant.

Final answer rule (multiple-choice mode):
Choose the option that best matches what you see. Your ring_presence value must be \
exactly one of: present | absent | null. \
Do not default to "absent" when the upper stem is poorly visible — if you can't \
judge it confidently, set confidence=cannot_tell and ring_presence=null instead."""


# ---------------------------------------------------------------------------
# volva_presence
# ---------------------------------------------------------------------------

VOLVA_PRESENCE_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "Focus ONLY on the BASE of the stem (stipe), where a volva — the remnant of the "
    "universal veil — would be located.\n"
    "Report only what you can directly observe — do not infer from species knowledge.\n\n"
    + _CONFIDENCE_BLOCK
)

VOLVA_PRESENCE_USER = """\
Examine this mushroom image and decide whether a VOLVA is present at the stem base.

First describe the base of the stem — any cup, sac, ring of scales, patches, or \
distinctive structure surrounding the very bottom of the stipe. Then classify.

If the BASE of the stem is NOT visible at all (e.g., only a top-down cap view, the \
base is buried in soil/leaves with the structure obscured, only the cap and upper \
stem are shown), set confidence=cannot_tell and volva_presence=null.

What counts as a VOLVA:
- A sac-like or cup-shaped membrane at the base with a free margin separated from \
the stem — the classic saccate volva of Amanita. Clear sac → present at high \
confidence.
- A distinct rim, collar, or sock-like wrapping at the base attached to the stem \
along its length → present at high confidence.
- Concentric rings, bands, scales, or warty patches around a bulbous base — \
universal-veil remnants such as friable, zoned, or napiform forms → present at \
low confidence.
- Patches of universal veil tissue stuck to the cap are corroborating evidence — \
if you also see warts/patches on the cap and a sac/rim/scaly base, lean toward present.

What does NOT count as a volva (anti-traps):
- A simple swollen bulb at the base with NO sac, NO scales, NO rim, NO adhering \
veil tissue is just a bulbous stem (bulbous stem shape ≠ volva).
- A plain tapered or equal stem base with no extra structure → absent.
- Soil, moss, or leaf litter clinging to the base is not a volva.
- A ring (annulus) on the upper stem is a separate feature, not a volva.
- An EARTHSTAR's outer rays are NOT a volva. Earthstars (Geastrum) have a \
star-shaped split outer skin that opens out around a central spore sac — the \
"rays" point outward from the body, not upward like a cup. There is no stem and \
no basal sac structure; this is an entire fruiting-body morphology, not a \
volva. If you see a star-shaped pattern of pointed rays radiating outward at \
ground level around a central spherical or domed structure → absent.
- A cup-fungus's whole body (Peziza, Discina, Helvella cup forms) is NOT a \
volva. Cup fungi have no stipe; the cup IS the fruiting body. Volva is only a \
basal cup AT THE BOTTOM of a stem — there must be a stem rising out of it.
- Gelatinous or jelly fungi (Tremella, etc.) and bracket/crust fungi growing \
from wood have no stem-and-base structure → cannot_tell or absent.

Options (listed rarest first — do not let order bias your choice; pick the option \
whose description best matches what you actually see):

- present: A sac, cup, rim, or distinct universal-veil remnants (concentric \
bands, scales, patches) are visible at the stem base.
- absent: The base is clearly visible and is either plain or merely bulbous, with \
NO sac, rim, scales, bands, or adhering veil tissue.

Final answer rule (multiple-choice mode):
Choose the option that best matches what you see. Your volva_presence value must \
be exactly one of: present | absent | null. \
Do not default to "absent" when the stem base is poorly visible — if you can't \
judge it confidently, set confidence=cannot_tell and volva_presence=null instead. \
And do not call a plain bulbous base a volva — true volvas have visible \
membrane, cup, rim, or distinct veil-remnant texture."""


# ---------------------------------------------------------------------------
# substrate
# ---------------------------------------------------------------------------

SUBSTRATE_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "Focus ONLY on what the mushroom is growing FROM — the surface, material, or "
    "object directly underneath or attached to its base.\n"
    "Report only what you can directly observe — do not infer from species knowledge.\n\n"
    + _CONFIDENCE_BLOCK
)

SUBSTRATE_USER = """\
Examine this mushroom image and identify the SUBSTRATE — what the mushroom is \
growing from.

First describe the immediate surroundings and the attachment point: what does the \
ground or surface look like at the base? Is there visible wood, bark, leaves, dung, \
or bare earth? Then classify.

If the base of the mushroom and its surroundings are NOT visible (e.g., a tight \
top-down cap-only crop, a detail shot of the cap surface, or the entire base is \
out of frame), set confidence=cannot_tell and substrate=null.

Substrate classes (listed rarest first — do not let order bias your choice; pick \
the option whose description best matches what you actually see):

- dung: Growing directly on animal dung, manure, or droppings. The substrate is a \
recognizable fecal mass or pile, not just brown soil. Coprophilous species.

- woody debris: Growing on small woody material — twigs, bark chips, wood shavings, \
mulch, or fragments smaller than a fist. Distinguish from dead wood by SIZE: \
large fallen branches, logs, or stumps are dead wood, not woody debris.

- leaf litter: Growing among loose decomposing leaves and forest-floor debris, \
with leaves clearly forming the upper layer of the substrate around the base. \
The mushroom appears to emerge from leaves rather than from bare soil.

- living tree: Growing from the bark, trunk, or exposed roots of a LIVING tree — \
green foliage above, intact bark, often vertical attachment to a standing trunk. \
Common for parasitic or wound-pathogen species.

- dead wood: Growing on or from a fallen log, stump, large branch, or rotting \
wood. The wood structure (bark, grain, cambium layer, decay zone) is visible. \
Moss-covered logs are still dead wood — examine through the moss for wood. If \
the mushroom appears to come from soil but you can see wood texture peeking out \
under leaves or moss, choose dead wood.

- soil: Growing from bare ground, humus, or sparse moss/grass directly on earth, \
with NO visible wood structure underneath. Mycorrhizal species typically grow on \
soil even when near trees — proximity to a trunk does not make it living tree if \
the attachment is at ground level.

Key distinctions:

- DEAD WOOD vs SOIL: If you can see grain, bark, or rotten wood texture anywhere \
near the base, it is dead wood — even if leaves partially cover it. Mushrooms \
growing from BURIED wood often look soil-attached at first glance; look for any \
visible wood under the surrounding leaves.

- DEAD WOOD vs WOODY DEBRIS: Size matters. A fallen branch or chunk you could \
not lift one-handed is dead wood. Twigs, bark chips, mulch, or scattered \
fragments are woody debris.

- DEAD WOOD vs LIVING TREE: Living tree has intact bark, often green foliage \
above, and the mushroom is attached to a vertical living trunk or large root. \
Dead wood has visible decay, missing bark, soft/punky texture, and is often \
horizontal (fallen).

- LEAF LITTER vs SOIL: Leaf litter has loose leaves clearly visible AS the top \
layer the mushroom is emerging from. Soil with a few stray leaves nearby is \
still soil — the leaves must be the dominant material at the base.

- DUNG vs SOIL: Dung is a recognizable fecal mass, not just dark soil. If you \
cannot see distinct droppings or a manure pile, choose soil even on rich dark \
ground.

Final answer rule (multiple-choice mode):
Choose the option whose description most accurately matches what you observe. \
Your substrate value must be exactly one of: \
dung | woody debris | leaf litter | living tree | dead wood | soil | null. \
Soil is the most common class — only choose soil when you can rule out wood, \
leaf litter, and dung. When in doubt and the base is poorly visible, prefer \
confidence=cannot_tell over a soil guess."""


# ---------------------------------------------------------------------------
# surface_texture (cap)
# ---------------------------------------------------------------------------

SURFACE_TEXTURE_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "Focus ONLY on the cap (pileus) surface texture visible in this image.\n"
    "Report only what you can directly observe — do not infer from species knowledge.\n\n"
    + _CONFIDENCE_BLOCK
)

SURFACE_TEXTURE_USER = """\
Examine this mushroom image and identify the cap (pileus) SURFACE TEXTURE.

First describe what the cap surface looks like in detail — any fibers, scales, \
hairs, dust, pits, wrinkles, or whether it is featureless and glossy. Note whether \
the cap appears wet or dry, since wetness can hide fine textures. Then classify.

If the cap is NOT visible at sufficient detail (e.g., distant shot, the cap is \
fully obscured, only the stem or underside is shown, or the cap is so wet that \
the surface looks like reflective glaze), set confidence=cannot_tell and \
surface_texture=null.

Texture classes (listed rarest first — do not let order bias your choice; pick \
the option whose description best matches what you actually see):

- areolate: Surface cracked into irregular block-like patches, resembling dried \
mud. The cracks form polygonal islands separated by visible gaps.

- pitted: Small rounded depressions or pits across the surface — like the cap of \
a morel-shaped fruiting body. Distinct holes, not raised structures.

- reticulate: A raised net-like pattern of ridges forming a mesh or honeycomb \
across the surface. Most often seen on bolete stems but applicable to caps too.

- pruinose: Covered with a very fine powder, looking frosted or dusted with flour. \
The surface appears matte and chalky, not fibrous. No discrete fibers or scales.

- warty: Discrete rounded wart-like projections — small bumps or pyramidal lumps \
sitting on the surface. Typically universal-veil remnants (e.g. Amanita pieces).

- squarrose: Erect, spreading, or recurved scales/fibril tips that point outward \
or curl back from the surface — especially prominent at the disc.

- floccose: Loose, cottony tufts or patches — more open and fluffy than tomentose, \
giving a wispy, broken-up appearance rather than a continuous mat.

- wrinkled: Broad longitudinal folds or ridges across the surface (rugose). The \
folds are part of the flesh itself, not separate structures.

- tomentose: A densely matted, woolly, or felt-like layer of interwoven fibrils. \
Continuous fuzzy coating, denser and more matted than velvety, more compact and \
felt-like than floccose.

- silky: Fine, closely appressed fibrils giving a satin-like sheen. The fibrils \
lie flat against the cap and reflect light to look like silk or satin. Subtler \
than fibrillose — no individual fibers stand out.

- fibrillose: Covered with fine thread-like fibers (fibrils) arranged radially \
or irregularly. Distinct individual fibers are visible against the ground color.

- velvety: A compact short layer of fine, soft hairs giving a velvet-like \
appearance. The surface looks softly fuzzy and matte, like the skin of a peach or \
like velvet fabric. No discrete scales or radiating fibers.

- scaly: Bearing distinct scales — flat appressed scales or raised tile-like \
plates of differing color/material from the underlying flesh. Each scale is a \
separate piece of tissue, not a fiber.

- smooth: No surface fibrils, scales, hairs, pits, warts, or ornamentation \
(glabrous). The cap looks glossy or matte but completely featureless.

Key distinctions:

- SMOOTH vs SILKY: Smooth has zero structure. Silky has appressed fibrils that \
give a satin sheen — you can detect a faint directional fiber pattern under \
oblique light. If in doubt and the surface looks featureless, choose smooth.

- SILKY vs FIBRILLOSE: Silky fibrils are SO appressed they look like a sheen, \
not individual fibers. Fibrillose has fibers you can pick out individually \
against the cap color.

- FIBRILLOSE vs VELVETY: Fibrillose fibers radiate or run irregularly and are \
visible as separate threads. Velvety is a uniform short fuzzy mat with no \
directional fiber pattern — looks like fabric.

- VELVETY vs TOMENTOSE: Velvety is short, even, finely fuzzy. Tomentose is \
denser, longer, more woolly/felted — visibly matted.

- TOMENTOSE vs FLOCCOSE: Tomentose is a continuous matted layer. Floccose is \
broken into loose cottony tufts or patches — more open and fluffy.

- SCALY vs FIBRILLOSE: Scales are distinct flat or raised pieces of tissue. \
Fibrils are thread-like fibers. If you see discrete chunks/plates/tiles → scaly. \
If you see threads/fibers → fibrillose.

- SCALY vs SQUARROSE: Squarrose is scaly with the scale tips pointing OUT or \
curling back — erect/recurved rather than flat. Appressed flat scales → scaly.

- WARTY vs SCALY: Warts are rounded bumps (like Amanita veil remnants — \
pyramidal or hemispherical). Scales are flat or tile-like pieces.

- WRINKLED vs RETICULATE: Wrinkled has broad folds going one way (often radial). \
Reticulate has ridges forming a NET pattern — interconnected lines.

- PRUINOSE vs SMOOTH: Pruinose has a chalky/dusty matte coating, like flour. \
Smooth has no coating at all. If the surface looks frosted or powdered, it is \
pruinose, not smooth.

Wet-cap caveat: A heavily wet, glossy, or reflective cap can mask fibers, \
scales, and pruinose dust, making any textured surface look smooth. If the cap \
is visibly soaked or dripping, prefer confidence=low or cannot_tell over a \
confident "smooth".

Final answer rule (multiple-choice mode):
Choose the option whose description most accurately matches what you observe. \
Your surface_texture value must be exactly one of: \
areolate | pitted | reticulate | pruinose | warty | squarrose | floccose | \
wrinkled | tomentose | silky | fibrillose | velvety | scaly | smooth | null. \
Smooth is the most common class — only choose smooth when the cap is genuinely \
featureless and dry. When in doubt and the surface has any visible structure, \
prefer the matching textured class over a smooth guess."""


# ---------------------------------------------------------------------------
# stem_shape
# ---------------------------------------------------------------------------

STEM_SHAPE_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "Focus ONLY on the overall shape (profile) of the stem (stipe), looking at how "
    "its diameter changes from the apex (just below the cap) to the base.\n"
    "Report only what you can directly observe — do not infer from species knowledge.\n\n"
    + _CONFIDENCE_BLOCK
)

STEM_SHAPE_USER = """\
Examine this mushroom image and identify the STEM (stipe) SHAPE in profile.

First describe the stem from top to bottom: does the diameter stay uniform? Is \
the base swollen, tapered, or extended into a root? Is there a swelling in the \
middle? Is the cross-section round or laterally flattened? Then classify.

If the stem is NOT visible at sufficient detail (cap-only view, the BASE is \
buried in soil/leaves and the lower stem is hidden, only the upper stem and \
cap visible), set confidence=cannot_tell and stem_shape=null. The base \
specifically is critical — without seeing the base you cannot reliably \
distinguish equal/clavate/bulbous/attenuated/rooting.

Stem shape classes (listed rarest first — do not let order bias your choice; \
pick the option whose description best matches what you actually see):

- rooting: The base extends into a long, root-like pseudorrhiza that penetrates \
deep into the substrate. The stem narrows into a tail-like extension that goes \
INTO the soil/wood. This is a strong, distinctive feature when present.

- compressed: The stem is laterally flattened so its cross-section is oval or \
elliptic rather than round. From profile views this can appear normal — only \
classify compressed if you can clearly see the flattening from an angle.

- ventricose: Swollen in the MIDDLE, narrowing toward both the apex and the \
base. The widest point is somewhere along the stem's length, not at the base. \
Like a small barrel or spindle.

- obclavate: Inverted club shape — widest at the apex (just below the cap), \
gradually narrowing toward the base. Top-heavy, tapering downward.

- attenuated: The stem tapers, becoming thinner toward the base (or sometimes \
toward the apex). The narrowing is gradual along the lower half. Distinct from \
rooting because there is NO root-like extension into the substrate — the \
narrowing simply ends at ground level.

- clavate: Club-shaped — the stem gradually thickens toward the base. The base \
is wider than the apex but not abruptly bulbous; the widening is a smooth \
gradient along the lower half of the stem.

- bulbous: A distinct, ABRUPT swelling at the very base of the stem, forming a \
bulb-like or onion-like enlargement. The bulb is clearly different from the \
rest of the stem and the transition is sharp, not gradual.

- equal: Uniform diameter from apex to base — a roughly cylindrical stem with no \
swelling, taper, or distinctive base shape. The width at the top matches the \
width at the bottom.

Key distinctions:

- EQUAL vs CLAVATE: Equal stems stay the same width top-to-bottom. Clavate \
stems gradually thicken downward — the base is noticeably wider than the apex, \
with a smooth gradient over the lower half. If the change is barely visible or \
within ±20% width variation, choose equal.

- CLAVATE vs BULBOUS: Clavate has a SMOOTH GRADIENT thickening toward the base. \
Bulbous has an ABRUPT swelling — a clear discontinuity where the bulb begins. \
A bulb looks like an onion stuck on the bottom; a clavate stem looks like a club.

- ATTENUATED vs ROOTING: Both narrow toward the base, but rooting extends into \
the substrate as a visible root-like tail penetrating the soil/wood. \
Attenuated simply tapers and stops at ground level with no underground extension.

- ATTENUATED vs EQUAL: Choose attenuated only if there is a clear taper visible \
along ≥30% of the stem's length. Slight irregularities or perspective foreshortening \
can make any stem look slightly tapered — when uncertain, prefer equal.

- VENTRICOSE vs CLAVATE / EQUAL: Ventricose has the MAXIMUM WIDTH in the middle \
of the stem with narrowing at BOTH ends. If the stem is widest at the base, \
choose clavate. If it is uniformly wide, choose equal.

- BULBOUS without volva: A bulbous base is a stem-shape feature; whether the \
bulb has a volva (sac, cup, scales, bands) is a separate feature. A plain bulb \
with no veil remnants is still bulbous for shape purposes.

Buried-base caveat: If the lower portion of the stem disappears into soil, \
leaves, moss, or wood and you cannot see whether the buried part swells, tapers, \
or has a root, set confidence=cannot_tell. Do not guess equal just because the \
visible upper portion is uniform.

Final answer rule (multiple-choice mode):
Choose the option whose description most accurately matches what you observe. \
Your stem_shape value must be exactly one of: \
rooting | compressed | ventricose | obclavate | attenuated | clavate | bulbous | \
equal | null. \
Equal is the most common class — only choose equal when the stem is visibly \
uniform from top to bottom. When in doubt and the base shape is poorly visible, \
prefer confidence=cannot_tell over an equal guess."""


# ---------------------------------------------------------------------------
# cap_shape
# ---------------------------------------------------------------------------

CAP_SHAPE_SYSTEM = (
    "You are a mycology assistant analyzing a single mushroom photograph.\n"
    "Focus ONLY on the overall PROFILE shape of the cap (pileus) visible in this "
    "image — its silhouette from the side.\n"
    "Report only what you can directly observe — do not infer from species knowledge.\n\n"
    + _CONFIDENCE_BLOCK
)

CAP_SHAPE_USER = """\
Examine this mushroom image and identify the CAP (pileus) SHAPE in profile.

First describe the cap silhouette: is the top a rounded dome, a flat plate, a \
cone, a bell, an egg, a ball? Is there a distinct central bump? A central \
depression? Does it form a funnel? Then classify.

Side-view requirement (critical): cap shape requires a side or 3/4 angle so \
that the profile is visible. A pure top-down photo cannot distinguish convex \
from flat from depressed — set visible=false and confidence=cannot_tell. Do \
NOT guess convex from a top-down view just because most caps are convex.

Maturity caveat: the SAME species can be convex (button) → flat (mature) → \
depressed (overmature). Classify the visible specimen AS PHOTOGRAPHED, not the \
species' typical mature form. If multiple caps of different ages are in the \
frame, classify the dominant/most-visible specimen.

Cap shape classes (listed rarest first — do not let order bias your choice; \
pick the option whose description best matches what you actually see):

- ovoid: Distinctly egg-shaped — taller than wide, with a rounded apex. Often \
seen in very young specimens before the cap expands. The whole cap is an oval \
with the long axis vertical.

- globose: Spherical or near-spherical — a ball-shape attached to (or sitting \
on) the substrate. Puffballs and very young button stages can look globose. \
The cap is roughly as tall as it is wide.

- irregular: No coherent geometric profile — lobed, contorted, asymmetric, or \
fused with neighboring caps so that no single shape is recognizable. Use only \
when the cap genuinely has no regular shape; do NOT use as a fallback for \
"hard to classify".

- infundibuliform: Funnel-shaped — the cap edges rise CLEARLY ABOVE the center, \
forming a deep cup or funnel that channels water toward the stem. The center \
is well below the rim. Seen in mature Craterellus, mature Clitocybe, etc.

- depressed: A central dimple or shallow concavity, but the cap edges are \
roughly LEVEL with or only slightly above the center. The depression is a \
saucer-like sunken middle, not a deep funnel.

- umbonate: An otherwise flat or convex cap with a DISTINCT raised central \
bump (umbo) at the apex. The umbo is a well-defined bump, often \
nipple-shaped. If the bump is barely there or you have to squint to see it, \
prefer convex.

- parabolic: Tall and arched — the profile is HIGHER THAN WIDE, like a tall \
narrow dome or a parabola. The cap is clearly taller than its diameter. \
Distinct from convex (which is wider than tall).

- campanulate: Bell-shaped — sides nearly straight or slightly outward-curved, \
narrowing toward the stem, like an upside-down bell. The cap silhouette \
matches a bell or a flared skirt. Common in Coprinus / Mycena / Panaeolus.

- conical: A clean cone — straight sides converging to a point at the apex. \
Distinct from campanulate (which has curved bell-like flaring sides) and from \
umbonate (which has a flat/convex base with a small bump on top).

- flat: A plane disc with a roughly flat top — neither dome nor depression, no \
central bump or dimple. Common in mature Russula, mature Lactarius, mature \
Pleurotus.

- convex: A rounded dome — wider than tall, smoothly curved on top with no \
central bump and no central depression. The most common cap shape. The cap \
silhouette resembles a hemisphere or a half-egg lying on its side.

Key distinctions:

- CONVEX vs FLAT: Convex has a clearly rounded top; flat is a plane disc. If \
the cap is even slightly domed, choose convex.

- CONVEX vs UMBONATE: Umbonate requires a DISTINCT central bump on an \
otherwise flat or convex base. A barely-noticeable rise → convex.

- CONVEX vs PARABOLIC: Parabolic is taller than wide (vertical bell/dome). \
Convex is wider than tall (low rounded dome).

- DEPRESSED vs INFUNDIBULIFORM: Both have a sunken center. Depressed has cap \
edges ROUGHLY LEVEL with or only slightly above the center — a saucer dimple. \
Infundibuliform has cap edges CLEARLY RISING above the center — a deep \
funnel that could hold water. If the cap clearly forms a funnel, choose \
infundibuliform; if it's just a shallow dimple, choose depressed.

- CONICAL vs CAMPANULATE: Conical has straight sides meeting at an apex (a \
geometric cone). Campanulate has curved sides like a bell — sides are not \
straight. Bell-shaped curve → campanulate; straight cone → conical.

- CONICAL vs PARABOLIC: Conical is straight-sided; parabolic is curved. If \
the silhouette is a smoothly arched dome that is taller than wide, choose \
parabolic. If the sides are straight lines converging to a point, choose \
conical.

- GLOBOSE vs OVOID: Globose is spherical (height ≈ width). Ovoid is taller \
than wide and egg-shaped. If the shape is recognizably an egg with a more \
pointed top, choose ovoid; if it is a ball, choose globose.

- IRREGULAR — use sparingly: only when the cap genuinely has no coherent \
geometric profile (lobed, contorted, fused). Do not use as a fallback for \
"too hard to classify"; prefer confidence=cannot_tell when the shape is \
unclear.

Top-down caveat: From a strict top-down photo you cannot distinguish convex / \
flat / depressed / umbonate / infundibuliform — they all look like a circle. \
If you only see a top-down view, set visible=false and confidence=cannot_tell. \
Do NOT default to convex from a top-down view.

Final answer rule (multiple-choice mode):
Choose the option whose description most accurately matches what you observe. \
Your cap_shape value must be exactly one of: \
ovoid | globose | irregular | infundibuliform | depressed | umbonate | \
parabolic | campanulate | conical | flat | convex | null. \
Convex is the most common class — only choose convex when the cap is \
genuinely a low rounded dome. When in doubt and the profile is poorly \
visible (top-down crop, distant photo), prefer confidence=cannot_tell over a \
convex guess."""


# ---------------------------------------------------------------------------
# Prompt registry — maps feature name → (system_prompt, user_prompt)
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: dict[str, tuple[str, str]] = {
    "hymenium_type": (HYMENIUM_TYPE_SYSTEM, HYMENIUM_TYPE_USER),
    "cap_color": (CAP_COLOR_SYSTEM, CAP_COLOR_USER),
    "ring_presence": (RING_PRESENCE_SYSTEM, RING_PRESENCE_USER),
    "volva_presence": (VOLVA_PRESENCE_SYSTEM, VOLVA_PRESENCE_USER),
    "substrate": (SUBSTRATE_SYSTEM, SUBSTRATE_USER),
    "surface_texture": (SURFACE_TEXTURE_SYSTEM, SURFACE_TEXTURE_USER),
    "stem_shape": (STEM_SHAPE_SYSTEM, STEM_SHAPE_USER),
    "cap_shape": (CAP_SHAPE_SYSTEM, CAP_SHAPE_USER),
}
