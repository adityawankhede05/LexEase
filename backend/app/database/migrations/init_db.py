"""
Database initialization for LexEase.

Creates all tables defined in the ORM models if they do not already exist.
Safe to call on every application startup — SQLAlchemy's ``create_all``
is idempotent (it skips tables that are already present).

Usage:
    from app.database.migrations.init_db import init_db
    init_db()
"""

import logging

from app.database.models import Base
from app.database.session import engine

logger = logging.getLogger(__name__)


def init_db() -> None:
    """
    Creates all database tables using the configured engine.

    This function is called once at application startup via the FastAPI
    lifespan event.  It is intentionally simple — no Alembic migrations
    are required for the SQLite phase of development.

    When switching to PostgreSQL, replace this with Alembic's
    ``upgrade("head")`` call.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as exc:
        logger.error(f"Database initialization failed: {exc}")
        raise
