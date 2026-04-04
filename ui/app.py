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
from ui.i18n import LANGUAGES, common_names_it, t, tv  # noqa: E402

_PROFILE_FEATURES: dict[str, list[str]] = yaml.safe_load(
    (_ROOT / "ingestion" / "profiles" / f"{settings.grouping_profile}.yaml").read_text()
)

API_BASE = "http://localhost:8001"

st.set_page_config(page_title="Mushroom Lookalikes Finder", layout="wide")

# --- Language toggle (sidebar, before any translated content) ---
if "lang" not in st.session_state:
    st.session_state.lang = "en"
st.sidebar.selectbox(
    "🌐 Language",
    options=list(LANGUAGES.keys()),
    format_func=lambda k: LANGUAGES[k],
    key="lang",
)

st.title(t("page_title"))
st.markdown(t("page_subtitle"))

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
        return "❓ " + t("edibility_unknown")
    icon = _EDIBILITY_ICON.get(edibility, "❓")
    return f"{icon} {t('edibility_' + edibility.replace(' ', '_'))}"


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
                f"</a>"
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
            val = ", ".join(tv(str(v)) for v in val if v)
        elif isinstance(val, bool):
            val = t("bool_yes") if val else t("bool_no")
        elif isinstance(val, str):
            val = tv(val)
        items.append(f"**{label}:** {val}")
    if items:
        st.markdown(f"*{title}*")
        for item in items:
            st.markdown(f"&nbsp;&nbsp;&nbsp;— {item}")


def _show_list(label: str, values: list | None) -> None:
    if values:
        st.markdown(f"**{label}:** {', '.join(tv(str(v)) for v in values if v)}")


def _common_names_display(s: dict) -> str:
    """Return common names string, preferring Italian when lang=it."""
    lang = st.session_state.get("lang", "en")
    if lang == "it":
        it_names = common_names_it(s.get("scientific_name", ""))
        if it_names:
            return ", ".join(it_names)
    return ", ".join(s.get("common_names") or [])


def _render_species_features(s: dict) -> None:
    """Render structured features for one species in two columns."""
    features = s.get("features") or {}
    if not features:
        st.info(t("no_features"))
        return

    # Metadata strip
    meta = []
    for tkey, key in [
        ("meta_order", "order"),
        ("meta_family", "family"),
        ("meta_genus", "genus"),
    ]:
        val = s.get(key) or features.get(key)
        if val:
            meta.append(f"{t(tkey)}: **{val}**")
    if s.get("source_count"):
        meta.append(f"{t('meta_sources')}: {s['source_count']}")
    conf = s.get("reconciliation_confidence")
    if conf is not None:
        meta.append(f"{t('meta_confidence')}: {conf:.0%}")
    synonyms = features.get("synonyms") or []
    if synonyms:
        meta.append(f"{t('meta_synonyms')}: {', '.join(synonyms)}")
    if meta:
        st.caption(" · ".join(meta))

    _render_source_links(s.get("sources"))

    # Species images
    _render_image_thumbnails(s.get("image_urls", []))

    col_morph, col_eco = st.columns(2)

    with col_morph:
        st.markdown(f"**🔬 {t('section_morphological')}**")

        _show_section(
            t("section_cap"),
            features.get("cap") or {},
            {
                "shape": t("field_shape"),
                "colors": t("field_colors"),
                "color_faded": t("field_color_faded"),
                "surface_moisture": t("field_surface_moisture"),
                "color_pattern": t("field_pattern"),
                "surface_texture": t("field_surface"),
                "scales_or_warts": t("field_scales_warts"),
                "margin_type": t("field_margin"),
                "margin_lined_at_maturity": t("field_margin_lined"),
                "central_depression": t("field_central_depression"),
                "diameter_min_cm": t("field_diam_min"),
                "diameter_max_cm": t("field_diam_max"),
            },
        )

        hymenium_type = (features.get("hymenium") or {}).get("type")
        if hymenium_type:
            st.markdown(f"*{t('section_hymenium_type')}:* **{tv(hymenium_type)}**")

        if hymenium_type == "gills" or (features.get("gills") or {}).get("attachment"):
            _show_section(
                t("section_gills"),
                features.get("gills") or {},
                {
                    "attachment": t("field_attachment"),
                    "spacing": t("field_spacing"),
                    "color": t("field_color"),
                    "color_with_age": t("field_color_age"),
                    "thickness": t("field_thickness"),
                    "texture": t("field_texture"),
                },
            )
        if hymenium_type == "pores" or (features.get("pores") or {}).get("color"):
            _show_section(
                t("section_pores"),
                features.get("pores") or {},
                {
                    "color": t("field_color"),
                    "color_with_age": t("field_color_age"),
                    "bruising_color": t("field_bruising"),
                    "density_per_mm": t("field_density"),
                },
            )
            _show_section(
                t("section_tubes"),
                features.get("tubes") or {},
                {
                    "depth_mm": t("field_depth"),
                },
            )

        _show_section(
            t("section_stem"),
            features.get("stem") or {},
            {
                "color": t("field_color"),
                "color_with_age": t("field_color_age"),
                "surface_texture": t("field_surface"),
                "reticulation": t("field_reticulation"),
                "shape": t("field_shape"),
                "consistency": t("field_consistency"),
                "hollow_or_solid": t("field_hollow_solid"),
                "base_color": t("field_base_color"),
                "basal_mycelium_color": t("field_basal_mycelium"),
                "finger_stain_color": t("field_finger_stain"),
                "height_min_cm": t("field_height_min"),
                "height_max_cm": t("field_height_max"),
                "diameter_min_cm": t("field_diam_min"),
                "diameter_max_cm": t("field_diam_max"),
            },
        )
        _show_section(
            t("section_veil"),
            features.get("veil") or {},
            {
                "present": t("field_present"),
                "type": t("field_type"),
                "cortina_present": t("field_cortina"),
                "shape": t("field_shape"),
                "color": t("field_color"),
            },
        )
        _show_section(
            t("section_volva"),
            features.get("volva") or {},
            {
                "present": t("field_present"),
                "type": t("field_type"),
                "shape": t("field_shape"),
                "color": t("field_color"),
            },
        )
        _show_section(
            t("section_flesh"),
            features.get("flesh") or {},
            {
                "color": t("field_color"),
                "bruising_color": t("field_bruising"),
                "odor": t("field_odor"),
                "taste": t("field_taste"),
                "texture": t("field_texture"),
                "quantity": t("field_quantity"),
            },
        )

        sp = features.get("spore_print_color")
        if sp:
            st.markdown(f"*{t('section_spore_print')}:* {tv(sp)}")
        sz = features.get("overall_size_class")
        if sz:
            st.markdown(f"*{t('section_overall_size')}:* {tv(sz)}")

        _show_section(
            t("section_spores"),
            features.get("spore") or {},
            {
                "shape": t("field_shape"),
                "length_min_um": t("field_length_min"),
                "length_max_um": t("field_length_max"),
                "width_min_um": t("field_width_min"),
                "width_max_um": t("field_width_max"),
                "ornamentation": t("field_ornamentation"),
                "spine_length_um": t("field_spine_length"),
                "spine_base_width_um": t("field_spine_base_width"),
                "amyloidity": t("field_amyloidity"),
                "color_in_KOH": t("field_color_koh"),
            },
        )
        _show_section(
            t("section_microscopic"),
            features.get("microscopic") or {},
            {
                "basidia_spore_count": t("field_basidia"),
                "cheilocystidia_shape": t("field_cheilocystidia_shape"),
                "cheilocystidia_dims_um": t("field_cheilocystidia_dims"),
                "pleurocystidia_shape": t("field_pleurocystidia_shape"),
                "pleurocystidia_dims_um": t("field_pleurocystidia_dims"),
                "cystidia_color_in_KOH": t("field_cystidia_koh"),
                "pileipellis_type": t("field_pileipellis_type"),
                "pileipellis_element_width_um": t("field_pileipellis_width"),
                "pileipellis_terminal_cell_shape": t("field_terminal_cell"),
            },
        )
        _show_section(
            t("section_chemical"),
            features.get("chemical") or {},
            {
                "KOH_cap": t("field_koh_cap"),
                "KOH_flesh": t("field_koh_flesh"),
                "NH4OH_cap": t("field_nh4oh_cap"),
                "NH4OH_flesh": t("field_nh4oh_flesh"),
                "FeSO4_cap": t("field_feso4_cap"),
                "FeSO4_flesh": t("field_feso4_flesh"),
            },
        )

    with col_eco:
        st.markdown(f"**🌿 {t('section_ecological')}**")
        eco = features.get("ecology") or {}
        if eco.get("trophic_mode"):
            st.markdown(f"**{t('field_trophic_mode')}:** {tv(eco['trophic_mode'])}")
        _show_list(t("field_habitat"), eco.get("habitat_types"))
        if eco.get("substrate"):
            st.markdown(f"**{t('field_substrate')}:** {tv(eco['substrate'])}")
        _show_list(t("field_associated_trees"), eco.get("associated_trees"))
        _show_list(t("field_fruiting_seasons"), eco.get("fruiting_seasons"))
        if eco.get("fruiting_months"):
            st.markdown(f"**{t('field_fruiting_months')}:** {eco['fruiting_months']}")
        _show_list(t("field_regions"), eco.get("geographic_regions"))
        if eco.get("growth_habit"):
            st.markdown(f"**{t('field_growth_habit')}:** {tv(eco['growth_habit'])}")
        if eco.get("growth_position"):
            st.markdown(f"**{t('field_growth_position')}:** {tv(eco['growth_position'])}")
        if eco.get("altitude_notes"):
            st.markdown(f"**{t('field_altitude')}:** {eco['altitude_notes']}")
        if eco.get("microhabitat_notes"):
            st.markdown(f"**{t('field_microhabitat')}:** {eco['microhabitat_notes']}")

        st.markdown("---")
        st.markdown(f"**⚠️ {t('section_safety')}**")
        edib = s.get("edibility") or features.get("edibility_status") or features.get("edibility")
        if edib:
            st.markdown(f"**{t('field_edibility')}:** {edibility_badge(edib)}")
        toxins = features.get("known_toxins") or []
        if toxins:
            st.markdown(f"**{t('field_toxins')}:** {', '.join(toxins)}")
        lookalikes = features.get("known_lookalikes") or []
        if lookalikes:
            st.markdown(f"**{t('field_known_lookalikes')}:** {', '.join(lookalikes)}")

    notes = features.get("extraction_notes")
    if notes:
        st.caption(f"📝 {t('field_extraction_notes')}: {notes}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_search, tab_index = st.tabs([f"🔍 {t('tab_search')}", f"📋 {t('tab_index')}"])

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
            t("input_species_label"),
            placeholder=t("input_species_placeholder"),
            key="species_input",
        )
    with col_context:
        region = st.text_input(t("input_region_label"), placeholder=t("input_region_placeholder"))
        _SEASONS = ["", "spring", "summer", "autumn", "winter"]
        season = st.selectbox(
            t("input_season_label"),
            _SEASONS,
            format_func=lambda s: t(f"season_{s}") if s else "",
        )

    # --- Example species (quick explore) ---
    st.markdown(f"**{t('try_example')}**")
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
        t("toggle_dangerous"),
        value=False,
        help=t("toggle_dangerous_help"),
    )

    # --- Weight sliders ---
    with st.expander(f"⚙️ {t('weights_title')}"):
        st.caption(t("weights_caption"))
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
            t("weight_numeric_label"),
            0.0,
            0.5,
            float(settings.weight_numeric),
            0.001,
            key="w_numeric",
        )

        st.divider()
        st.markdown(f"**{t('weights_blend_title')}**")
        alpha = st.slider(
            "Alpha",
            0.0,
            1.0,
            0.5,
            0.05,
            help=t("weights_alpha_help"),
            key="alpha",
        )

        st.divider()
        st.markdown(f"**{t('weights_filters_title')}**")
        filter_cols = st.columns(3)
        with filter_cols[0]:
            body_form_filter = st.checkbox(
                t("filter_body_form"),
                value=settings.weight_body_form_filter,
                help=t("filter_body_form_help"),
            )
        with filter_cols[1]:
            hymenium_filter = st.checkbox(
                t("filter_hymenium"),
                value=settings.weight_hymenium_filter,
                help=t("filter_hymenium_help"),
            )
        with filter_cols[2]:
            size_class_filter = st.checkbox(
                t("filter_size_class"),
                value=settings.weight_size_class_filter,
                help=t("filter_size_class_help"),
            )
        pool_cols = st.columns(2)
        with pool_cols[0]:
            morpho_pool_required = st.checkbox(
                t("filter_morpho_pool"),
                value=settings.weight_morpho_pool_required,
                help=t("filter_morpho_pool_help"),
            )
        with pool_cols[1]:
            morphotype_prefilter = st.checkbox(
                t("filter_morphotype"),
                value=settings.weight_morphotype_prefilter,
                help=t("filter_morphotype_help"),
            )

        st.divider()
        _AGG_LABELS = {
            "weighted_avg": t("agg_weighted_avg"),
            "rrf": t("agg_rrf"),
            "contrastive_gate": t("agg_contrastive_gate"),
            "learned_ranker": t("agg_learned_ranker"),
        }
        aggregation_strategy = st.radio(
            t("aggregation_strategy"),
            options=["learned_ranker", "weighted_avg", "rrf", "contrastive_gate"],
            format_func=_AGG_LABELS.get,
            horizontal=True,
        )
        z_threshold = 0.0
        if aggregation_strategy == "contrastive_gate":
            z_threshold = st.slider(
                t("z_threshold_label"),
                -1.0,
                2.0,
                0.0,
                0.1,
                help=t("z_threshold_help"),
            )

    # --- Search ---
    if st.button(t("btn_find"), type="primary", disabled=not species_name):
        with st.spinner(t("searching")):
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
                    detail = resp.json().get("detail", t("error_not_found"))
                    st.error(f"❌ {detail}")
                    st.stop()
                resp.raise_for_status()
                data = resp.json()
            except requests.ConnectionError:
                st.error(t("error_no_connection"))
                st.stop()
            except requests.HTTPError as e:
                st.error(t("error_api").format(error=e))
                st.stop()

        # --- No-edibility warning for dangerous mode ---
        if dangerous_mode and not data.get("query_species_edibility"):
            st.warning(t("no_edibility_warning"))

        # --- Query species images ---
        _render_image_thumbnails(data.get("query_species_image_urls", []))

        # --- Safety warning ---
        safety = data.get("explanation_safety_warning")
        if safety:
            st.error(f"⚠️ **{t('safety_warning_prefix')}** {safety}")

        # --- At-a-glance summary ---
        summary = data.get("explanation_summary")
        if summary:
            st.info(summary)

        # --- Notable confusions ---
        notable = data.get("explanation_notable_pairs", [])
        if notable:
            st.markdown(f"**{t('key_confusions')}**")
            for pair in notable:
                st.markdown(f"- {pair}")

        st.divider()
        mode_label = " dangerous" if dangerous_mode else ""
        st.markdown(
            f"**{t('results_found').format(count=len(data['candidates']), mode=mode_label)}** "
            f"({t('results_compared').format(count=data['species_count_in_db'])})"
        )

        # --- Ranked candidates (card layout) ---
        for i, cand in enumerate(data["candidates"], 1):
            name = cand["scientific_name"]
            edib_text = cand.get("edibility") or "unknown"
            overall = cand["similarity_overall"]

            # Common names: prefer Italian when lang=it
            common = _common_names_display(cand)

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
                    st.metric(t("similarity_label"), f"{overall:.0%}")
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
                    st.markdown(f"**{t('key_differences')}**")
                    for fc in diffs[:3]:
                        feat = fc["feature_name"].replace(".", " > ")
                        st.markdown(
                            f"- **{feat}**: {species_name} = *{fc['query_value']}* "
                            f"vs {name} = *{fc['candidate_value']}*"
                        )

                # Detailed breakdown (collapsed)
                with st.expander(t("show_breakdown")):
                    group_rows = [
                        {t("col_group"): g, t("col_score"): f"{v:.1%}"}
                        for g, v in sorted(cand["group_similarities"].items(), key=lambda x: -x[1])
                    ]
                    group_rows.append(
                        {
                            t("col_group"): "numeric",
                            t("col_score"): f"{cand['similarity_numeric']:.1%}",
                        }
                    )
                    group_rows.append(
                        {
                            t("col_group"): "jaccard",
                            t("col_score"): f"{cand.get('similarity_jaccard', 0):.1%}",
                        }
                    )
                    st.dataframe(group_rows, use_container_width=True, hide_index=True)

                    if comparisons:
                        st.markdown(f"**{t('full_comparison')}**")
                        rows = []
                        for fc in comparisons:
                            rows.append(
                                {
                                    t("col_feature"): fc["feature_name"],
                                    t("col_group"): fc["feature_group"],
                                    species_name: fc["query_value"] or "—",
                                    name: fc["candidate_value"] or "—",
                                    t("col_similar"): "✓" if fc["is_similar"] else "✗",
                                }
                            )
                        st.dataframe(rows, use_container_width=True)


# ===========================================================================
# Tab 2: Species Index
# ===========================================================================
with tab_index:
    st.subheader(t("index_title"))
    idx_search = st.text_input(
        "Search by name...", placeholder=t("index_search_placeholder"), key="idx_search"
    )

    try:
        r = requests.get(f"{API_BASE}/api/v1/species/summary", timeout=10)
        if r.ok:
            all_species = r.json()

            # Client-side filter (search both English and Italian common names)
            if idx_search:
                term = idx_search.lower()
                filtered = []
                for s in all_species:
                    if term in s["scientific_name"].lower():
                        filtered.append(s)
                        continue
                    if any(term in cn.lower() for cn in (s.get("common_names") or [])):
                        filtered.append(s)
                        continue
                    it_names = common_names_it(s["scientific_name"])
                    if any(term in cn.lower() for cn in it_names):
                        filtered.append(s)
                all_species = filtered

            st.caption(t("index_showing").format(count=len(all_species)))

            # Compact table view
            table_rows = []
            for s in all_species:
                common = _common_names_display(s)
                table_rows.append(
                    {
                        t("col_species"): s["scientific_name"],
                        t("col_common_names"): common,
                        t("col_edibility"): edibility_badge(s.get("edibility")),
                    }
                )
            if table_rows:
                st.dataframe(table_rows, use_container_width=True, hide_index=True)

            # Detail view: select a species to see full profile
            species_names = [s["scientific_name"] for s in all_species]
            if species_names:
                selected = st.selectbox(
                    t("index_select"),
                    options=[""] + species_names,
                    key="idx_select",
                )
                if selected:
                    with st.spinner(t("index_loading").format(name=selected)):
                        profile_resp = requests.get(
                            f"{API_BASE}/api/v1/species/{selected}", timeout=10
                        )
                    if profile_resp.ok:
                        _render_species_features(profile_resp.json())
                    else:
                        st.warning(t("error_load_profile"))
        else:
            st.warning(t("error_load_species"))
    except requests.ConnectionError:
        st.warning(t("error_no_connection_index"))
    except Exception:
        st.warning(t("error_unexpected"))

# --- Footer ---
st.divider()
st.caption(f"⚠️ {t('footer_warning')}")
