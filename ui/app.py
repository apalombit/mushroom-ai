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

st.set_page_config(page_title="Mushroom Lookalikes Finder", layout="wide")

st.title("Mushroom Lookalikes Finder")
st.markdown("Find species that look similar — and learn how to tell them apart.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _render_source_links(sources: list[dict] | None) -> None:
    """Render compact source reference links."""
    if not sources:
        return
    links = []
    for src in sources:
        name = src.get("source_name", "")
        url = src.get("source_url")
        if url:
            links.append(f"[{name}]({url})")
        elif name:
            links.append(name)
    if links:
        st.caption("🔗 " + " · ".join(links))


def _render_image_thumbnails(image_refs: list, height_px: int = 200) -> None:
    """Render uniform-sized image thumbnails with source attribution."""
    if not image_refs:
        return
    cols = st.columns(len(image_refs))
    for col, ref in zip(cols, image_refs):
        with col:
            if isinstance(ref, str):
                url, source_name, source_url = ref, "", None
            elif isinstance(ref, dict):
                url = ref.get("image_url", "")
                source_name = ref.get("source_name", "")
                source_url = ref.get("source_url")
            else:
                url = getattr(ref, "image_url", "")
                source_name = getattr(ref, "source_name", "")
                source_url = getattr(ref, "source_url", None)
            html = (
                f'<a href="{url}" target="_blank" title="Click to view full size">'
                f'<img src="{url}" '
                f'style="max-height:{height_px}px;max-width:100%;'
                f'object-fit:contain;border-radius:6px;cursor:zoom-in;" />'
                f'</a>'
            )
            st.markdown(html, unsafe_allow_html=True)
            if source_url and source_name:
                st.caption(f"[{source_name}]({source_url})")
            elif source_name:
                st.caption(source_name)


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

    _render_source_links(s.get("sources"))

    # Species images
    _render_image_thumbnails(s.get("image_urls", []))

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


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_search, tab_index = st.tabs(["🔍 Find Lookalikes", "📋 Species Index"])

# ===========================================================================
# Tab 1: Find Lookalikes
# ===========================================================================
with tab_search:
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
    examples = [
        "Amanita caesarea",
        "Agaricus campestris",
        "Boletus edulis",
        "Cantharellus cibarius",
    ]
    example_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with example_cols[i]:
            st.button(
                ex,
                key=f"ex_{i}",
                on_click=lambda name=ex: st.session_state.update({"species_input": name}),
            )

    # --- Search mode toggle ---
    dangerous_mode = st.toggle(
        "Dangerous lookalikes only",
        value=False,
        help="Show only lookalikes with opposite edibility (safe vs dangerous).",
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
                    0.0,
                    0.5,
                    float(round(default_w, 4)),
                    0.001,
                    help=tooltip,
                    key=f"w_{group}",
                )
        st.divider()
        w_numeric = st.slider(
            "Numeric (size/measurements)",
            0.0,
            0.5,
            float(settings.weight_numeric),
            0.001,
            key="w_numeric",
        )

        st.divider()
        st.markdown("**Embedding / Jaccard blend**")
        alpha = st.slider(
            "Alpha",
            0.0,
            1.0,
            0.5,
            0.05,
            help="1.0 = pure embedding, 0.0 = pure Jaccard, 0.5 = equal blend",
            key="alpha",
        )

        st.divider()
        st.markdown("**Filters & candidate pool**")
        filter_cols = st.columns(3)
        with filter_cols[0]:
            body_form_filter = st.checkbox(
                "Body-form filter",
                value=settings.weight_body_form_filter,
                help="Exclude species with incompatible body form",
            )
        with filter_cols[1]:
            hymenium_filter = st.checkbox(
                "Hymenium filter",
                value=settings.weight_hymenium_filter,
                help="Exclude species with incompatible hymenium type",
            )
        with filter_cols[2]:
            size_class_filter = st.checkbox(
                "Size class filter",
                value=settings.weight_size_class_filter,
                help="Exclude species with incompatible size class",
            )
        pool_cols = st.columns(2)
        with pool_cols[0]:
            morpho_pool_required = st.checkbox(
                "Morpho pool required",
                value=settings.weight_morpho_pool_required,
                help="Candidates must appear in at least one morphological embedding group",
            )
        with pool_cols[1]:
            morphotype_prefilter = st.checkbox(
                "Morphotype prefilter",
                value=settings.weight_morphotype_prefilter,
                help="Pre-filter candidates by morphotype signature match",
            )

        st.divider()
        aggregation_strategy = st.radio(
            "Aggregation strategy",
            options=["learned_ranker", "weighted_avg", "rrf", "contrastive_gate"],
            format_func={
                "weighted_avg": "Weighted Average",
                "rrf": "RRF",
                "contrastive_gate": "Contrastive + Gate",
                "learned_ranker": "GBDT Ranker",
            }.get,
            horizontal=True,
        )
        z_threshold = 0.0
        if aggregation_strategy == "contrastive_gate":
            z_threshold = st.slider(
                "Z-score gate threshold",
                -1.0,
                2.0,
                0.0,
                0.1,
                help=(
                    "Groups below this z-score are excluded from scoring. "
                    "0.0 = above-average only."
                ),
            )

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
                "hymenium_filter": hymenium_filter,
                "size_class_filter": size_class_filter,
                "alpha": alpha,
                "morpho_pool_required": morpho_pool_required,
                "morphotype_prefilter": morphotype_prefilter,
                "dangerous_filter": dangerous_mode,
                "top_k": 10,
                "aggregation_strategy": aggregation_strategy,
                "contrastive_z_threshold": z_threshold,
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

        # --- No-edibility warning for dangerous mode ---
        if dangerous_mode and not data.get("query_species_edibility"):
            st.warning("No edibility data for this species — filter had no effect.")

        # --- Query species images ---
        _render_image_thumbnails(data.get("query_species_image_urls", []))

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
        mode_label = " dangerous" if dangerous_mode else ""
        st.markdown(
            f"**{len(data['candidates'])}{mode_label} lookalikes found** "
            f"(compared {data['species_count_in_db']} species in database)"
        )

        # --- Ranked candidates (card layout) ---
        for i, cand in enumerate(data["candidates"], 1):
            name = cand["scientific_name"]
            edib_text = cand.get("edibility") or "unknown"
            overall = cand["similarity_overall"]
            common = ", ".join(cand.get("common_names") or [])

            with st.container(border=True):
                # Header row: name | edibility | similarity
                col_name, col_edib, col_sim = st.columns([3, 1, 1])
                with col_name:
                    st.markdown(f"#### {i}. {name}")
                    if common:
                        st.caption(common)
                with col_edib:
                    if edib_text in ("deadly", "toxic"):
                        st.error(edibility_badge(edib_text))
                    elif edib_text in ("inedible", "conditionally edible"):
                        st.warning(edibility_badge(edib_text))
                    elif edib_text in ("edible", "choice"):
                        st.success(edibility_badge(edib_text))
                    else:
                        st.info(edibility_badge(edib_text))
                with col_sim:
                    st.metric("Similarity", f"{overall:.0%}")
                    st.progress(min(overall, 1.0))

                # Candidate images
                _render_image_thumbnails(cand.get("image_urls", []))

                # Source reference links (always visible)
                _render_source_links(cand.get("sources"))

                # Top distinguishing features (upfront)
                comparisons = cand.get("feature_comparisons", [])
                diffs = [
                    fc
                    for fc in comparisons
                    if not fc["is_similar"] and fc["query_value"] and fc["candidate_value"]
                ]
                if diffs:
                    st.markdown("**Key differences:**")
                    for fc in diffs[:3]:
                        feat = fc["feature_name"].replace(".", " > ")
                        st.markdown(
                            f"- **{feat}**: {species_name} = *{fc['query_value']}* "
                            f"vs {name} = *{fc['candidate_value']}*"
                        )

                # Detailed breakdown (collapsed)
                with st.expander("Show detailed breakdown"):
                    group_rows = [
                        {"Group": g, "Score": f"{v:.1%}"}
                        for g, v in sorted(cand["group_similarities"].items(), key=lambda x: -x[1])
                    ]
                    group_rows.append(
                        {"Group": "numeric", "Score": f"{cand['similarity_numeric']:.1%}"}
                    )
                    group_rows.append(
                        {
                            "Group": "jaccard",
                            "Score": f"{cand.get('similarity_jaccard', 0):.1%}",
                        }
                    )
                    st.dataframe(group_rows, use_container_width=True, hide_index=True)

                    if comparisons:
                        st.markdown("**Full feature comparison:**")
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


# ===========================================================================
# Tab 2: Species Index
# ===========================================================================
with tab_index:
    st.subheader("Species Index")
    idx_search = st.text_input(
        "Search by name...", placeholder="e.g. Amanita or Caesar", key="idx_search"
    )

    try:
        r = requests.get(f"{API_BASE}/api/v1/species/summary", timeout=10)
        if r.ok:
            all_species = r.json()

            # Client-side filter
            if idx_search:
                term = idx_search.lower()
                all_species = [
                    s
                    for s in all_species
                    if term in s["scientific_name"].lower()
                    or any(term in cn.lower() for cn in (s.get("common_names") or []))
                ]

            st.caption(f"Showing {len(all_species)} species")

            # Compact table view
            table_rows = []
            for s in all_species:
                common = ", ".join(s.get("common_names") or [])
                table_rows.append(
                    {
                        "Species": s["scientific_name"],
                        "Common names": common,
                        "Edibility": edibility_badge(s.get("edibility")),
                    }
                )
            if table_rows:
                st.dataframe(table_rows, use_container_width=True, hide_index=True)

            # Detail view: select a species to see full profile
            species_names = [s["scientific_name"] for s in all_species]
            if species_names:
                selected = st.selectbox(
                    "Select a species for full profile",
                    options=[""] + species_names,
                    key="idx_select",
                )
                if selected:
                    with st.spinner(f"Loading profile for {selected}..."):
                        profile_resp = requests.get(
                            f"{API_BASE}/api/v1/species/{selected}", timeout=10
                        )
                    if profile_resp.ok:
                        _render_species_features(profile_resp.json())
                    else:
                        st.warning("Could not load full profile.")
        else:
            st.warning("Could not load species list.")
    except requests.ConnectionError:
        st.warning("Could not load species list — is the API running?")
    except Exception:
        st.warning("Could not load species list — unexpected error.")

# --- Footer ---
st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only. "
    "Never eat wild mushrooms based solely on automated identification."
)
