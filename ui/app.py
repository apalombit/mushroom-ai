"""
Streamlit dashboard for Mushroom Lookalikes Finder.

Run:
    streamlit run ui/app.py --server.port 8501

Layout:
    - Species name input + optional context (region, season)
    - Example species buttons for quick exploration
    - Weight sliders for tuning morphological/ecological/taxonomic balance
    - Results: safety warning (red), LLM summary, ranked lookalike list
    - Expandable feature comparison table per lookalike
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8001"

st.set_page_config(page_title="🍄 Mushroom Lookalikes Finder", layout="wide")

st.title("🍄 Mushroom Lookalikes Finder")
st.markdown("Find species that look similar — and learn how to tell them apart.")

# --- Input ---
col_input, col_context = st.columns([2, 1])
with col_input:
    species_name = st.text_input(
        "Mushroom name (scientific or common)",
        placeholder="e.g. Amanita caesarea",
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
        if st.button(ex, key=f"ex_{i}"):
            species_name = ex

# --- Weight sliders ---
with st.expander("⚙️ Similarity weights (advanced)"):
    w_morph = st.slider("Morphological", 0.0, 1.0, 0.60, 0.05)
    w_eco = st.slider("Ecological", 0.0, 1.0, 0.25, 0.05)
    w_taxon = st.slider("Taxonomic", 0.0, 1.0, 0.15, 0.05)

_EDIBILITY_ICON = {
    "edible": "🟢",
    "conditionally_edible": "🟡",
    "inedible": "🟠",
    "toxic": "🔴",
    "deadly": "⛔",
}


def edibility_badge(edibility: str | None) -> str:
    if not edibility:
        return "❓ unknown"
    icon = _EDIBILITY_ICON.get(edibility, "❓")
    return f"{icon} {edibility}"


# --- Search ---
if st.button("Find Lookalikes", type="primary", disabled=not species_name):
    with st.spinner("Searching..."):
        payload = {
            "species_name": species_name,
            "region": region or None,
            "season": season or None,
            "weight_morphological": w_morph,
            "weight_ecological": w_eco,
            "weight_taxonomic": w_taxon,
            "top_k": 10,
        }
        try:
            resp = requests.post(f"{API_BASE}/api/v1/lookalikes", json=payload, timeout=60)
            if resp.status_code == 404:
                st.error(f"Species not found: **{species_name}**. Is it in the database?")
                st.stop()
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            st.error("Cannot connect to backend. Run: `make serve`")
            st.stop()
        except requests.HTTPError as e:
            st.error(f"API error: {e}")
            st.stop()

    # --- Safety warning (prominent red box) ---
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
            col1, col2, col3 = st.columns(3)
            col1.metric("Morphological", f"{cand['similarity_morphological']:.1%}")
            col2.metric("Ecological", f"{cand['similarity_ecological']:.1%}")
            col3.metric("Taxonomic", f"{cand['similarity_taxonomic']:.1%}")

            comparisons = cand.get("feature_comparisons", [])
            if comparisons:
                st.markdown("**Feature comparison:**")
                rows = []
                for fc in comparisons:
                    rows.append({
                        "Feature": fc["feature_name"],
                        "Group": fc["feature_group"],
                        species_name: fc["query_value"] or "—",
                        name: fc["candidate_value"] or "—",
                        "Similar": "✓" if fc["is_similar"] else "✗",
                    })
                st.dataframe(rows, use_container_width=True)

# --- Footer ---
st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only. "
    "Never eat wild mushrooms based solely on automated identification."
)
