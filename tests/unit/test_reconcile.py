"""Unit tests for reconciliation. No external services required."""

from unittest.mock import MagicMock

from db.models import ReconciledSpecies, SourceObservation
from ingestion.reconcile import reconcile_species


def _make_session(observations, existing=None):
    """Build a mock SQLAlchemy session for reconciliation tests."""
    obs_mock = MagicMock()
    obs_mock.filter_by.return_value.all.return_value = observations

    rs_mock = MagicMock()
    rs_mock.filter_by.return_value.first.return_value = existing

    session = MagicMock()

    def side_effect(model):
        if model is SourceObservation:
            return obs_mock
        return rs_mock

    session.query.side_effect = side_effect
    session.add = MagicMock()
    session.commit = MagicMock()
    session.refresh = MagicMock()
    return session


def test_single_source_copies_features_directly(sample_features_json):
    """Single-source reconciliation copies features_json with confidence=1.0."""
    obs = MagicMock(spec=SourceObservation)
    obs.features_json = sample_features_json
    obs.extraction_timestamp = None

    session = _make_session([obs])
    added = []
    session.add.side_effect = added.append

    reconcile_species(session, "Amanita muscaria")

    assert len(added) == 1
    row = added[0]
    assert isinstance(row, ReconciledSpecies)
    assert row.scientific_name == "Amanita muscaria"
    assert row.reconciliation_confidence == 1.0
    assert row.features_json == sample_features_json
    session.commit.assert_called()


def test_single_source_sets_taxonomy_fields(sample_features_json):
    """Taxonomy fields are extracted from features_json."""
    obs = MagicMock(spec=SourceObservation)
    obs.features_json = sample_features_json
    obs.extraction_timestamp = None

    session = _make_session([obs])
    added = []
    session.add.side_effect = added.append

    reconcile_species(session, "Amanita muscaria")

    row = added[0]
    assert row.family == "Amanitaceae"
    assert row.genus == "Amanita"
    assert row.edibility == "toxic"


def test_no_observations_returns_none():
    """Returns None when there are no source observations."""
    session = _make_session([])
    result = reconcile_species(session, "Nonexistentus fakicus")
    assert result is None
    session.add.assert_not_called()


def test_incremental_skip_when_up_to_date(sample_features_json):
    """Skips reconciliation when reconciled_at >= newest extraction_timestamp."""
    from datetime import datetime

    old_ts = datetime(2024, 1, 1)
    new_ts = datetime(2024, 6, 1)

    obs = MagicMock(spec=SourceObservation)
    obs.features_json = sample_features_json
    obs.extraction_timestamp = old_ts

    existing = MagicMock(spec=ReconciledSpecies)
    existing.reconciled_at = new_ts  # newer than extraction → skip

    session = _make_session([obs], existing=existing)

    result = reconcile_species(session, "Amanita muscaria")

    assert result is existing
    session.add.assert_not_called()
    session.commit.assert_not_called()
