"""Integration tests for database connectivity and schema. Requires live Postgres."""

from sqlalchemy import text

from db.connection import check_connection, engine, get_session
from db.models import GroundTruthPair


def test_check_connection():
    assert check_connection() is True


def test_all_tables_exist():
    expected = {"source_observations", "reconciled_species", "ground_truth_pairs"}
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
        )
        tables = {row[0] for row in result}
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"


def test_ground_truth_pairs_loaded():
    session = get_session()
    try:
        count = session.query(GroundTruthPair).count()
        assert count > 0, "No ground truth pairs found — run python -m scripts.init_db first"
    finally:
        session.close()
