import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app.database.document_context_store as store_module
import app.database.session as session_module
from app.database.models import Base


@pytest.fixture(autouse=True)
def db_session_override(monkeypatch):
    """
    Creates an isolated in-memory SQLite database for each test run.
    Uses StaticPool so connections share the exact same in-memory database.
    """
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    test_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )

    monkeypatch.setattr(session_module, "engine", test_engine)
    monkeypatch.setattr(session_module, "SessionLocal", test_session_local)

    # Patch existing singleton store to use the in-memory test session
    monkeypatch.setattr(
        store_module.document_context_store, "_session_factory", test_session_local
    )
    store_module.document_context_store._fallback_store.clear()

    yield test_session_local

    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()
