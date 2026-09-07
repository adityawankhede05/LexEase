import io
import fitz
import pytest
from fastapi.testclient import TestClient

from app.database.document_context_store import DocumentContextStore, document_context_store
from app.database.models import Clause, ClauseRiskResult, Document, QAInteraction, Summary
import app.database.session as session_module
from app.main import app

def get_test_db():
    return session_module.SessionLocal()
from app.schemas.clause_analysis import ClauseRiskResult as SchemaRiskResult, RiskLevel
from app.schemas.document import ClauseSegment
from app.schemas.qa import DocumentQAResponse
from app.schemas.summary import DocumentSummaryResponse
from app.services.clause_analysis import ClauseAnalysisService
from app.services.qa import DocumentQAService
from app.services.summary import DocumentSummaryService

client = TestClient(app)


def _make_clause(idx: int = 1, text: str = "") -> ClauseSegment:
    return ClauseSegment(
        clause_id=f"clause_{idx}",
        clause_number=str(idx),
        text=text or f"Sample text for clause {idx}.",
    )


def _create_mock_pdf(text: str = "Preamble\n\n1. Payment terms apply.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


# ---------------------------------------------------------------------------
# 1. DocumentContextStore Unit Tests
# ---------------------------------------------------------------------------

def test_store_and_retrieve():
    """Verify that storing clauses returns a valid document_id and retrieves the exact clauses."""
    clauses = [_make_clause(1, "Clause 1 text"), _make_clause(2, "Clause 2 text")]
    doc_id = document_context_store.store(clauses, filename="contract.pdf", page_count=2, character_count=50)

    retrieved = document_context_store.get(doc_id)
    assert retrieved is not None
    assert len(retrieved) == 2
    assert retrieved[0].clause_id == "clause_1"
    assert retrieved[0].text == "Clause 1 text"
    assert retrieved[1].clause_id == "clause_2"
    assert retrieved[1].text == "Clause 2 text"


def test_get_missing_document():
    """Verify that querying a nonexistent document_id returns None."""
    assert document_context_store.get("non-existent-uuid") is None


def test_delete_document():
    """Verify that deleting a document removes it and subsequent get returns None."""
    clauses = [_make_clause(1, "Confidentiality clause")]
    doc_id = document_context_store.store(clauses)
    assert document_context_store.get(doc_id) is not None

    document_context_store.delete(doc_id)
    assert document_context_store.get(doc_id) is None


def test_persistence_across_separate_store_instances(db_session_override):
    """Verify that a newly instantiated DocumentContextStore retrieves data persisted by another instance."""
    store_a = DocumentContextStore(session_factory=db_session_override)
    store_b = DocumentContextStore(session_factory=db_session_override)

    clauses = [_make_clause(1, "Indemnification clause")]
    doc_id = store_a.store(clauses)

    retrieved = store_b.get(doc_id)
    assert retrieved is not None
    assert len(retrieved) == 1
    assert retrieved[0].text == "Indemnification clause"


def test_document_isolation():
    """Verify that storing multiple documents maintains full isolation without clause bleeding."""
    doc_id_1 = document_context_store.store([_make_clause(1, "Doc 1 Clause")])
    doc_id_2 = document_context_store.store([_make_clause(2, "Doc 2 Clause")])

    clauses_1 = document_context_store.get(doc_id_1)
    clauses_2 = document_context_store.get(doc_id_2)

    assert len(clauses_1) == 1
    assert clauses_1[0].text == "Doc 1 Clause"

    assert len(clauses_2) == 1
    assert clauses_2[0].text == "Doc 2 Clause"


# ---------------------------------------------------------------------------
# 2. Upload Endpoint Persistence Integration
# ---------------------------------------------------------------------------

def test_upload_creates_document_and_clauses_in_db():
    """Verify that POST /documents/upload creates records in documents and clauses tables."""
    pdf_bytes = _create_mock_pdf("Sample agreement\n\n1. Rent is INR 20000.\n\n2. Security deposit.")
    response = client.post(
        "/documents/upload",
        files={"file": ("lease.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    doc_id = data["document_id"]
    assert doc_id is not None

    with get_test_db() as db:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        assert doc is not None
        assert doc.filename == "lease.pdf"
        assert doc.page_count == 1

        db_clauses = db.query(Clause).filter(Clause.document_id == doc_id).order_by(Clause.position.asc()).all()
        assert len(db_clauses) == len(data["clauses"])
        assert db_clauses[0].clause_id == data["clauses"][0]["clause_id"]


# ---------------------------------------------------------------------------
# 3. Summary Persistence
# ---------------------------------------------------------------------------

def test_summary_persistence_direct():
    """Verify that summary persistence saves and upserts summary records in the database."""
    summary_service = DocumentSummaryService()
    doc_id = "test-summary-doc-id"
    summary_data = DocumentSummaryResponse(
        summary="This is a test summary.",
        key_points=["Point 1", "Point 2"],
        document_type="Employment Agreement",
    )

    summary_service._persist_summary(doc_id, summary_data)

    with get_test_db() as db:
        s = db.query(Summary).filter(Summary.document_id == doc_id).first()
        assert s is not None
        assert s.summary_text == "This is a test summary."
        assert s.key_points == ["Point 1", "Point 2"]
        assert s.document_type == "Employment Agreement"

    # Test upsert (update)
    updated_summary = DocumentSummaryResponse(
        summary="Updated summary text.",
        key_points=["Point A"],
        document_type="Commercial Lease",
    )
    summary_service._persist_summary(doc_id, updated_summary)

    with get_test_db() as db:
        count = db.query(Summary).filter(Summary.document_id == doc_id).count()
        assert count == 1
        s = db.query(Summary).filter(Summary.document_id == doc_id).first()
        assert s.summary_text == "Updated summary text."
        assert s.key_points == ["Point A"]
        assert s.document_type == "Commercial Lease"


# ---------------------------------------------------------------------------
# 4. Risk Analysis Persistence
# ---------------------------------------------------------------------------

def test_risk_analysis_persistence_direct():
    """Verify that clause risk analysis results are persisted in clause_risk_results table."""
    service = ClauseAnalysisService()
    doc_id = "test-risk-doc-id"
    results = [
        SchemaRiskResult(
            clause_id="clause_1",
            clause_number="1.",
            risk_level=RiskLevel.HIGH,
            explanation="Uncapped indemnification liability.",
            recommendation="Cap liability to total fees paid.",
            confidence=0.95,
            risk_score=85.0,
            risk_label="HIGH",
            reasons=["Uncapped damages"],
            issues=["High legal exposure"],
            recommended_action="Negotiate cap",
        )
    ]

    service._persist_results(doc_id, results)

    with get_test_db() as db:
        stored = db.query(ClauseRiskResult).filter(ClauseRiskResult.document_id == doc_id).all()
        assert len(stored) == 1
        assert stored[0].clause_id == "clause_1"
        assert stored[0].risk_level == "high"
        assert stored[0].confidence == 0.95
        assert stored[0].risk_score == 85.0
        assert stored[0].reasons == ["Uncapped damages"]
        assert stored[0].issues == ["High legal exposure"]
        assert stored[0].recommended_action == "Negotiate cap"


# ---------------------------------------------------------------------------
# 5. Q&A Persistence & 404 Behavior
# ---------------------------------------------------------------------------

def test_qa_persistence_direct():
    """Verify that completed Q&A interactions are persisted in qa_interactions table."""
    qa_service = DocumentQAService()
    doc_id = "test-qa-doc-id"
    qa_resp = DocumentQAResponse(
        answer="The monthly rent is INR 25,000.",
        source_clauses=["clause_1"],
        confidence=0.98,
        cannot_answer=False,
    )

    qa_service._persist_interaction(doc_id, "What is the monthly rent?", qa_resp)

    with get_test_db() as db:
        stored = db.query(QAInteraction).filter(QAInteraction.document_id == doc_id).all()
        assert len(stored) == 1
        assert stored[0].question == "What is the monthly rent?"
        assert stored[0].answer == "The monthly rent is INR 25,000."
        assert stored[0].source_clauses == ["clause_1"]
        assert stored[0].confidence == 0.98
        assert stored[0].cannot_answer is False


def test_qa_unknown_document_id_returns_404():
    """Verify that POST /documents/ask returns 404 when document_id is unknown."""
    response = client.post(
        "/documents/ask",
        json={"document_id": "nonexistent-doc-id", "question": "What is the termination period?"},
    )
    assert response.status_code == 404
    assert "No document context found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 6. Database Failure Handling Resilience
# ---------------------------------------------------------------------------

def test_store_fallback_on_db_failure(monkeypatch):
    """Verify that if database write fails, store uses in-memory fallback and does not raise."""
    store = DocumentContextStore()

    def mock_get_session_fail():
        raise RuntimeError("Simulated DB connection error")

    monkeypatch.setattr(store, "_get_session", mock_get_session_fail)

    clauses = [_make_clause(1, "Fallback clause")]
    doc_id = store.store(clauses)
    assert doc_id is not None

    # Retrieval falls back to memory safely
    retrieved = store.get(doc_id)
    assert retrieved is not None
    assert len(retrieved) == 1
    assert retrieved[0].text == "Fallback clause"
