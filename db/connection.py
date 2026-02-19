"""Database connection and session management."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from config import settings

engine = create_engine(settings.database_url, echo=(settings.environment == "development"))
SessionLocal = sessionmaker(bind=engine)


def get_session() -> Session:
    """Get a new database session. Caller must close it."""
    return SessionLocal()


def check_connection() -> bool:
    """Verify database is reachable and pgvector extension is available."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
