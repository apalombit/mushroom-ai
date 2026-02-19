"""
Streamlit dashboard for Mushroom Lookalikes Finder.

Run:
    streamlit run ui/app.py --server.port 8501

Layout:
    - Species name input + optional context (region, season)
    - Weight sliders for tuning morphological/ecological/taxonomic balance
    - Example species buttons for quick exploration
    - Results: "At a glance" LLM summary at top
    - Ranked lookalike list with color-coded danger levels
    - Expandable feature comparison sections per lookalike

TODO:
    - [ ] Implement API calls to FastAPI backend
    - [ ] Build results display with color-coded danger levels
    - [ ] Add expandable feature comparison tables
    - [ ] Add example species buttons from seed list
    - [ ] Handle errors (backend down, species not found, no results)
"""

import streamlit as st

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

# --- Search ---
if st.button("Find Lookalikes", type="primary", disabled=not species_name):
    st.info("⏳ Searching... (not yet connected to backend)")

    # TODO:
    # 1. POST to http://localhost:8001/api/v1/lookalikes
    # 2. Display LLM explanation summary at top
    # 3. Show ranked list with danger-level color coding
    # 4. Expandable per-candidate feature comparison table

    st.warning("Backend not yet implemented.")

# --- Footer ---
st.divider()
st.caption(
    "⚠️ This tool is for educational purposes only. "
    "Never eat wild mushrooms based solely on automated identification."
)
