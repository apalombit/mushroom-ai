"""
Streamlit dashboard for Mushroom Lookalikes Finder.

Run:
    streamlit run ui/app.py --server.port 8501
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests  # noqa: E402
import streamlit as st  # noqa: E402
import yaml  # noqa: E402

from config import settings  # noqa: E402
from ingestion.rubric import EMBEDDING_GROUPS  # noqa: E402

_PROFILE_FEATURES: dict[str, list[str]] = yaml.safe_load(
    (_ROOT / "ingestion" / "profiles" / f"{settings.grouping_profile}.yaml").read_text()
)

API_BASE = "http://localhost:8001"

st.set_page_config(page_title="🍄 Mushroom Lookalikes Finder", layout="wide")

st.title("🍄 Mushroom Lookalikes Finder")
st.markdown("Find species that look similar — and learn how to tell them apart.")

# --- Session state init ---
if "species_input" not in st.session_state:
    st.session_state.species_input = ""

# --- Input ---
col_input, col_context = st.columns([2, 1])
with col_input:
    species_name = st.text_input(
        "Mushroom name (scientific or common)",
        placeholder="e.g. Amanita caesarea",
        key="species_input",
    )
with col_context:
    region = st.text_input("Region (optional)", placeholder="e.g. Northern Italy")
    season = st.selectbox("Season (optional)", ["", "spring", "summer", "autumn", "winter"])

# --- Example species (quick explore) ---
st.markdown("**Try an example:**")
examples = ["Amanita caesarea", "Agaricus campestris", "Boletus edulis", "Cantharellus cibarius"]
example_cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    with example_cols[i]:
        st.button(
            ex,
            key=f"ex_{i}",
            on_click=lambda name=ex: st.session_state.update({"species_input": name}),
        )

# --- Weight sliders ---
with st.expander("⚙️ Similarity weights (advanced)"):
    st.caption("Weights are sent as-is; the API normalizes them to sum to 1.")
    cols = st.columns(3)
    group_weights: dict[str, float] = {}
    for i, group in enumerate(EMBEDDING_GROUPS):
        default_w = getattr(settings, f"weight_{group}", None) or round(
            (1.0 - settings.weight_numeric) / len(EMBEDDING_GROUPS), 4
        )
        features = _PROFILE_FEATURES.get(group, [])
        tooltip = ", ".join(features) if features else group
        with cols[i % 3]:
            group_weights[group] = st.slider(
                group.replace("_", " ").title(),
                0.0, 0.5, float(round(default_w, 4)), 0.001,
                help=tooltip,
                key=f"w_{group}",
            )
    st.divider()
    w_numeric = st.slider(
        "Numeric (size/measurements)", 0.0, 0.5, float(settings.weight_numeric), 0.001,
        key="w_numeric",
    )
    body_form_filter = st.checkbox(
        "Body-form filter", value=False, help="Exclude species with incompatible body form"
    )

_EDIBILITY_ICON = {
    "edible": "🟢",
    "choice": "⭐",
    "conditionally edible": "🟡",
    "inedible": "🟠",
    "toxic": "🔴",
    "deadly": "⛔",
}


def edibility_badge(edibility: str | None) -> str:
    if not edibility:
        return "❓ unknown"
    icon = _EDIBILITY_ICON.get(edibility, "❓")
    return f"{icon} {edibility}"


def _show_section(title: str, data: dict, fields: dict[str, str]) -> None:
    """Render a named sub-section (cap, gills, stem …) if it has any non-empty values."""
    if not data:
        return
    items = []
    for key, label in fields.items():
        val = data.get(key)
        if val is None or val == [] or val == "":
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val if v)
        if isinstance(val, bool):
            val = "yes" if val else "no"
        items.append(f"**{label}:** {val}")
    if items:
        st.markdown(f"*{title}*")
        for item in items:
            st.markdown(f"&nbsp;&nbsp;&nbsp;— {item}")


def _show_list(label: str, values: list | None) -> None:
    if values:
        st.markdown(f"**{label}:** {', '.join(str(v) for v in values if v)}")


def _render_species_features(s: dict) -> None:
    """Render structured features for one species in two columns."""
    features = s.get("features") or {}
    if not features:
        st.info("No structured features available for this species.")
        return

    # Metadata strip
    meta = []
    for label, key in [("Order", "order"), ("Family", "family"), ("Genus", "genus")]:
        val = s.get(key) or features.get(key)
        if val:
            meta.append(f"{label}: **{val}**")
    if s.get("source_count"):
        meta.append(f"Sources: {s['source_count']}")
    conf = s.get("reconciliation_confidence")
    if conf is not None:
        meta.append(f"Confidence: {conf:.0%}")
    synonyms = features.get("synonyms") or []
    if synonyms:
        meta.append(f"Synonyms: {', '.join(synonyms)}")
    if meta:
        st.caption(" · ".join(meta))

    # Source links
    source_links = []
    for src in s.get("sources") or []:
        name = src.get("source_name", "")
        url = src.get("source_url")
        if url:
            source_links.append(f"[{name}]({url})")
        elif name:
            source_links.append(name)
    if source_links:
        st.caption("🔗 " + " · ".join(source_links))

    col_morph, col_eco = st.columns(2)

    with col_morph:
        st.markdown("**🔬 Morphological**")

        _show_section(
            "Cap",
            features.get("cap") or {},
            {
                "shape": "Shape",
                "colors": "Colors",
                "color_faded": "Color (faded)",
                "surface_moisture": "Surface moisture",
                "color_pattern": "Pattern",
                "surface_texture": "Surface",
                "scales_or_warts": "Scales / warts",
                "margin_type": "Margin",
                "margin_lined_at_maturity": "Margin lined at maturity",
                "central_depression": "Central depression",
                "diameter_min_cm": "Diam. min (cm)",
                "diameter_max_cm": "Diam. max (cm)",
            },
        )

        hymenium_type = (features.get("hymenium") or {}).get("type")
        if hymenium_type:
            st.markdown(f"*Hymenium type:* **{hymenium_type}**")

        if hymenium_type == "gills" or (features.get("gills") or {}).get("attachment"):
            _show_section(
                "Gills",
                features.get("gills") or {},
                {
                    "attachment": "Attachment",
                    "spacing": "Spacing",
                    "color": "Color",
                    "color_with_age": "Color with age",
                    "thickness": "Thickness",
                    "texture": "Texture",
                },
            )
        if hymenium_type == "pores" or (features.get("pores") or {}).get("color"):
            _show_section(
                "Pores",
                features.get("pores") or {},
                {
                    "color": "Color",
                    "color_with_age": "Color with age",
                    "bruising_color": "Bruising",
                    "density_per_mm": "Density (per mm)",
                },
            )
            _show_section(
                "Tubes",
                features.get("tubes") or {},
                {
                    "depth_mm": "Depth (mm)",
                },
            )

        _show_section(
            "Stem",
            features.get("stem") or {},
            {
                "color": "Color",
                "color_with_age": "Color with age",
                "surface_texture": "Surface",
                "reticulation": "Reticulation",
                "shape": "Shape",
                "consistency": "Consistency",
                "hollow_or_solid": "Hollow / solid",
                "base_color": "Base color",
                "basal_mycelium_color": "Basal mycelium",
                "finger_stain_color": "Finger stain",
                "height_min_cm": "Height min (cm)",
                "height_max_cm": "Height max (cm)",
                "diameter_min_cm": "Diam. min (cm)",
                "diameter_max_cm": "Diam. max (cm)",
            },
        )
        _show_section(
            "Veil / ring",
            features.get("veil") or {},
            {
                "present": "Present",
                "type": "Type",
                "cortina_present": "Cortina",
                "shape": "Shape",
                "color": "Color",
            },
        )
        _show_section(
            "Volva",
            features.get("volva") or {},
            {
                "present": "Present",
                "type": "Type",
                "shape": "Shape",
                "color": "Color",
            },
        )
        _show_section(
            "Flesh",
            features.get("flesh") or {},
            {
                "color": "Color",
                "bruising_color": "Bruising",
                "odor": "Odor",
                "taste": "Taste",
                "texture": "Texture",
                "quantity": "Quantity",
            },
        )

        sp = features.get("spore_print_color")
        if sp:
            st.markdown(f"*Spore print color:* {sp}")
        sz = features.get("overall_size_class")
        if sz:
            st.markdown(f"*Overall size:* {sz}")

        _show_section(
            "Spores (microscopic)",
            features.get("spore") or {},
            {
                "shape": "Shape",
                "length_min_um": "Length min (µm)",
                "length_max_um": "Length max (µm)",
                "width_min_um": "Width min (µm)",
                "width_max_um": "Width max (µm)",
                "ornamentation": "Ornamentation",
                "spine_length_um": "Spine length (µm)",
                "spine_base_width_um": "Spine base width (µm)",
                "amyloidity": "Amyloidity",
                "color_in_KOH": "Color in KOH",
            },
        )
        _show_section(
            "Microscopic anatomy",
            features.get("microscopic") or {},
            {
                "basidia_spore_count": "Basidia",
                "cheilocystidia_shape": "Cheilocystidia shape",
                "cheilocystidia_dims_um": "Cheilocystidia dims (µm)",
                "pleurocystidia_shape": "Pleurocystidia shape",
                "pleurocystidia_dims_um": "Pleurocystidia dims (µm)",
                "cystidia_color_in_KOH": "Cystidia in KOH",
                "pileipellis_type": "Pileipellis type",
                "pileipellis_element_width_um": "Pileipellis width (µm)",
                "pileipellis_terminal_cell_shape": "Terminal cell shape",
            },
        )
        _show_section(
            "Chemical reactions",
            features.get("chemical") or {},
            {
                "KOH_cap": "KOH (cap)",
                "KOH_flesh": "KOH (flesh)",
                "NH4OH_cap": "NH₄OH (cap)",
                "NH4OH_flesh": "NH₄OH (flesh)",
                "FeSO4_cap": "FeSO₄ (cap)",
                "FeSO4_flesh": "FeSO₄ (flesh)",
            },
        )

    with col_eco:
        st.markdown("**🌿 Ecological**")
        eco = features.get("ecology") or {}
        if eco.get("trophic_mode"):
            st.markdown(f"**Trophic mode:** {eco['trophic_mode']}")
        _show_list("Habitat", eco.get("habitat_types"))
        if eco.get("substrate"):
            st.markdown(f"**Substrate:** {eco['substrate']}")
        _show_list("Associated trees", eco.get("associated_trees"))
        _show_list("Fruiting seasons", eco.get("fruiting_seasons"))
        if eco.get("fruiting_months"):
            st.markdown(f"**Fruiting months:** {eco['fruiting_months']}")
        _show_list("Regions", eco.get("geographic_regions"))
        if eco.get("growth_habit"):
            st.markdown(f"**Growth habit:** {eco['growth_habit']}")
        if eco.get("growth_position"):
            st.markdown(f"**Growth position:** {eco['growth_position']}")
        if eco.get("altitude_notes"):
            st.markdown(f"**Altitude:** {eco['altitude_notes']}")
        if eco.get("microhabitat_notes"):
            st.markdown(f"**Microhabitat:** {eco['microhabitat_notes']}")

        st.markdown("---")
        st.markdown("**⚠️ Safety**")
        edib = s.get("edibility") or features.get("edibility_status") or features.get("edibility")
        if edib:
            st.markdown(f"**Edibility:** {edibility_badge(edib)}")
        toxins = features.get("known_toxins") or []
        if toxins:
            st.markdown(f"**Toxins:** {', '.join(toxins)}")
        lookalikes = features.get("known_lookalikes") or []
        if lookalikes:
            st.markdown(f"**Known lookalikes:** {', '.join(lookalikes)}")

    notes = features.get("extraction_notes")
    if notes:
        st.caption(f"📝 Extraction notes: {notes}")


# --- Search ---
if st.button("Find Lookalikes", type="primary", disabled=not species_name):
    with st.spinner("Searching..."):
        payload = {
            "species_name": species_name,
            "region": region or None,
            "season": season or None,
            "weights": group_weights,
            "weight_numeric": w_numeric,
            "body_form_filter": body_form_filter,
            "top_k": 10,
        }
        try:
            resp = requests.post(f"{API_BASE}/api/v1/lookalikes", json=payload, timeout=60)
            if resp.status_code == 404:
                detail = resp.json().get("detail", "Species not found")
                st.error(f"❌ {detail}")
                st.stop()
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            st.error("Cannot connect to backend. Run: `make serve`")
            st.stop()
        except requests.HTTPError as e:
            st.error(f"API error: {e}")
            st.stop()

    # --- Safety warning ---
    safety = data.get("explanation_safety_warning")
    if safety:
        st.error(f"⚠️ **Safety Warning:** {safety}")

    # --- At-a-glance summary ---
    summary = data.get("explanation_summary")
    if summary:
        st.info(summary)

    # --- Notable confusions ---
    notable = data.get("explanation_notable_pairs", [])
    if notable:
        st.markdown("**Key confusions:**")
        for pair in notable:
            st.markdown(f"- {pair}")

    st.divider()
    st.markdown(
        f"**{len(data['candidates'])} lookalikes found** "
        f"(compared {data['species_count_in_db']} species in database)"
    )

    # --- Ranked candidates ---
    for i, cand in enumerate(data["candidates"], 1):
        name = cand["scientific_name"]
        edib = edibility_badge(cand.get("edibility"))
        overall = cand["similarity_overall"]
        common = ", ".join(cand.get("common_names") or [])
        header = f"{i}. **{name}** {edib} — similarity: {overall:.1%}"
        if common:
            header += f" ({common})"

        with st.expander(header):
            group_rows = [
                {"Group": g, "Score": f"{v:.1%}"}
                for g, v in sorted(
                    cand["group_similarities"].items(), key=lambda x: -x[1]
                )
            ]
            group_rows.append({"Group": "numeric", "Score": f"{cand['similarity_numeric']:.1%}"})
            st.dataframe(group_rows, use_container_width=True, hide_index=True)

            comparisons = cand.get("feature_comparisons", [])
            if comparisons:
                st.markdown("**Feature comparison:**")
                rows = []
                for fc in comparisons:
                    rows.append(
                        {
                            "Feature": fc["feature_name"],
                            "Group": fc["feature_group"],
                            species_name: fc["query_value"] or "—",
                            name: fc["candidate_value"] or "—",
                            "Similar": "✓" if fc["is_similar"] else "✗",
                        }
                    )
                st.dataframe(rows, use_container_width=True)

# --- Species index expander ---
with st.expander("📋 Species currently indexed in the database"):
    try:
        r = requests.get(f"{API_BASE}/api/v1/species?limit=1000", timeout=10)
        if r.ok:
            species_list = r.json()
            st.caption(f"{len(species_list)} species in database")
            for s in species_list:
                common = ", ".join(s["common_names"] or [])
                header = f"*{s['scientific_name']}*"
                if common:
                    header += f" — {common}"
                header += f"  {edibility_badge(s.get('edibility'))}"
                with st.expander(header):
                    _render_species_features(s)
    except Exception:
        st.warning("Could not load species list — is the API running?")

# --- Known associations expander ---
with st.expander("🔗 Known dangerous confusions (ground truth pairs)"):
    try:
        r = requests.get(f"{API_BASE}/api/v1/associations", timeout=10)
        if r.ok:
            rows = [
                {
                    "Species A": p["species_a"],
                    "Species B": p["species_b"],
                    "Why it matters": p.get("danger_note") or "—",
                }
                for p in r.json()
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
    except Exception:
        st.warning("Could not load associations — is the API running?")

# --- Footer ---
st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only. "
    "Never eat wild mushrooms based solely on automated identification."
)
