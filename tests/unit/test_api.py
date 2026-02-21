"""Unit tests for FastAPI endpoints. Mocks similarity search and LLM calls."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """TestClient with DB connectivity check bypassed."""
    with patch("api.main.check_connection", return_value=True):
        with TestClient(app) as c:
            yield c


def _mock_session(count=10):
    session = MagicMock()
    session.query.return_value.count.return_value = count
    session.query.return_value.offset.return_value.limit.return_value.all.return_value = []
    session.query.return_value.filter.return_value.first.return_value = None
    return session


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "mushroom-lookalikes"}


# ---------------------------------------------------------------------------
# Species list
# ---------------------------------------------------------------------------

def test_species_list_empty(client):
    with patch("api.routes.get_session", return_value=_mock_session()):
        resp = client.get("/api/v1/species")
    assert resp.status_code == 200
    assert resp.json() == []


def test_species_list_returns_profiles(client):
    mock_row = MagicMock()
    mock_row.scientific_name = "Amanita muscaria"
    mock_row.common_names = ["Fly agaric"]
    mock_row.family = "Amanitaceae"
    mock_row.genus = "Amanita"
    mock_row.edibility = "toxic"
    mock_row.features_json = {}
    mock_row.source_count = 1
    mock_row.needs_review = False
    mock_row.reconciliation_confidence = 1.0

    session = _mock_session()
    session.query.return_value.offset.return_value.limit.return_value.all.return_value = [mock_row]

    with patch("api.routes.get_session", return_value=session):
        resp = client.get("/api/v1/species")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["scientific_name"] == "Amanita muscaria"
    assert data[0]["edibility"] == "toxic"


# ---------------------------------------------------------------------------
# Single species lookup
# ---------------------------------------------------------------------------

def test_get_species_not_found(client):
    with patch("api.routes.get_session", return_value=_mock_session()):
        resp = client.get("/api/v1/species/Nonexistentus fakicus")
    assert resp.status_code == 404


def test_get_species_found(client):
    mock_row = MagicMock()
    mock_row.scientific_name = "Amanita caesarea"
    mock_row.common_names = ["Caesar's mushroom"]
    mock_row.family = "Amanitaceae"
    mock_row.genus = "Amanita"
    mock_row.edibility = "edible"
    mock_row.features_json = {"cap": {"colors": ["orange"]}}
    mock_row.source_count = 1
    mock_row.needs_review = False
    mock_row.reconciliation_confidence = 1.0

    session = _mock_session()
    session.query.return_value.filter.return_value.first.return_value = mock_row

    with patch("api.routes.get_session", return_value=session):
        resp = client.get("/api/v1/species/Amanita caesarea")

    assert resp.status_code == 200
    data = resp.json()
    assert data["scientific_name"] == "Amanita caesarea"
    assert data["edibility"] == "edible"


# ---------------------------------------------------------------------------
# Lookalikes endpoint
# ---------------------------------------------------------------------------

def test_find_lookalikes_species_not_found(client):
    with patch("api.routes.search_lookalikes", side_effect=ValueError("Species not found")):
        with patch("api.routes.get_session", return_value=_mock_session()):
            resp = client.post(
                "/api/v1/lookalikes",
                json={"species_name": "Nonexistentus fakicus"},
            )
    assert resp.status_code == 404


def test_find_lookalikes_no_embeddings(client):
    with patch(
        "api.routes.search_lookalikes",
        side_effect=ValueError("has no embeddings"),
    ):
        with patch("api.routes.get_session", return_value=_mock_session()):
            resp = client.post(
                "/api/v1/lookalikes",
                json={"species_name": "Amanita caesarea"},
            )
    assert resp.status_code == 404


def test_find_lookalikes_success(client):
    mock_query_species = MagicMock()
    mock_query_species.edibility = "edible"

    mock_candidate = {
        "scientific_name": "Amanita muscaria",
        "common_names": ["Fly agaric"],
        "edibility": "toxic",
        "similarity_morphological": 0.80,
        "similarity_ecological": 0.70,
        "similarity_taxonomic": 0.90,
        "similarity_overall": 0.82,
        "features_json": {},
        "feature_comparisons": [],
    }

    mock_explanation = MagicMock()
    mock_explanation.summary = "Test summary."
    mock_explanation.notable_pairs = ["Amanita caesarea vs Amanita muscaria."]
    mock_explanation.safety_warning = "Amanita muscaria is toxic!"

    sl = patch("api.routes.search_lookalikes", return_value=(mock_query_species, [mock_candidate]))
    ct = patch("api.routes.build_comparison_table", return_value=[mock_candidate])
    ge = patch("api.routes.generate_explanation", return_value=mock_explanation)
    gs = patch("api.routes.get_session", return_value=_mock_session(count=10))
    with sl, ct, ge, gs:
        resp = client.post("/api/v1/lookalikes", json={"species_name": "Amanita caesarea"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["query_species"] == "Amanita caesarea"
    assert data["query_species_edibility"] == "edible"
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["scientific_name"] == "Amanita muscaria"
    assert data["candidates"][0]["edibility"] == "toxic"
    assert data["explanation_summary"] == "Test summary."
    assert data["explanation_safety_warning"] == "Amanita muscaria is toxic!"
    assert data["species_count_in_db"] == 10


# ---------------------------------------------------------------------------
# Associations endpoint
# ---------------------------------------------------------------------------

def test_list_associations_empty(client):
    session = _mock_session()
    session.query.return_value.all.return_value = []
    with patch("api.routes.get_session", return_value=session):
        resp = client.get("/api/v1/associations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_associations_returns_pairs(client):
    from db.models import GroundTruthPair
    mock_pair = MagicMock(spec=GroundTruthPair)
    mock_pair.species_a = "Amanita caesarea"
    mock_pair.species_b = "Amanita muscaria"
    mock_pair.danger_note = "A is edible, B is toxic"
    mock_pair.source = "literature"

    session = _mock_session()
    session.query.return_value.all.return_value = [mock_pair]
    with patch("api.routes.get_session", return_value=session):
        resp = client.get("/api/v1/associations")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["species_a"] == "Amanita caesarea"
    assert data[0]["species_b"] == "Amanita muscaria"
    assert data[0]["danger_note"] == "A is edible, B is toxic"


def test_find_lookalikes_explanation_failure_is_non_fatal(client):
    """A failed LLM explanation should not break the lookalike response."""
    mock_query_species = MagicMock()
    mock_query_species.edibility = "edible"
    mock_candidate = {
        "scientific_name": "Amanita muscaria",
        "common_names": [],
        "edibility": "toxic",
        "similarity_morphological": 0.80,
        "similarity_ecological": 0.70,
        "similarity_taxonomic": 0.90,
        "similarity_overall": 0.82,
        "features_json": {},
        "feature_comparisons": [],
    }

    sl = patch("api.routes.search_lookalikes", return_value=(mock_query_species, [mock_candidate]))
    ct = patch("api.routes.build_comparison_table", return_value=[mock_candidate])
    ge = patch("api.routes.generate_explanation", side_effect=RuntimeError("LLM down"))
    gs = patch("api.routes.get_session", return_value=_mock_session())
    with sl, ct, ge, gs:
        resp = client.post("/api/v1/lookalikes", json={"species_name": "Amanita caesarea"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["explanation_summary"] is None
    assert data["candidates"][0]["scientific_name"] == "Amanita muscaria"
