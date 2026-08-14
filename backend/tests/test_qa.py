import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError
from app.database.document_context_store import DocumentContextStore, document_context_store
from app.main import app
from app.schemas.document import ClauseSegment
from app.schemas.qa import DocumentQARequest, DocumentQAResponse
from app.services.ai import AIService
from app.services.exceptions import DocumentContextNotFoundError, QAGenerationError
from app.services.qa import DocumentQAService
from app.services.retrieval import ClauseRetrievalService

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

class MockAIProvider(BaseAIProvider):
    """In-memory AI provider returning canned responses without hitting Gemini."""

    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.last_prompt: str | None = None
        self.call_count = 0

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self.response_text


def _make_clause(idx: int = 1, text: str = "") -> ClauseSegment:
    default_text = f"Clause {idx} details: The tenant shall pay monthly rent of INR 25,000."
    return ClauseSegment(
        clause_id=f"clause_{idx}",
        clause_number=str(idx),
        text=text or default_text,
    )


# ---------------------------------------------------------------------------
# 1. Valid Question Schema
# ---------------------------------------------------------------------------

def test_qa_request_valid_schema():
    """DocumentQARequest accepts valid document_id and question."""
    req = DocumentQARequest(document_id="doc_123", question="What is the monthly rent?")
    assert req.document_id == "doc_123"
    assert req.question == "What is the monthly rent?"


# ---------------------------------------------------------------------------
# 2. Empty Question Schema Validation
# ---------------------------------------------------------------------------

def test_qa_request_empty_question():
    """DocumentQARequest raises ValidationError when question is empty."""
    with pytest.raises(ValidationError):
        DocumentQARequest(document_id="doc_123", question="")


# ---------------------------------------------------------------------------
# 3. Excessively Long Question Schema Validation
# ---------------------------------------------------------------------------

def test_qa_request_excessively_long_question():
    """DocumentQARequest raises ValidationError when question exceeds 2000 characters."""
    long_q = "a" * 2001
    with pytest.raises(ValidationError):
        DocumentQARequest(document_id="doc_123", question=long_q)


# ---------------------------------------------------------------------------
# 4. Missing Document Context Error
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_qa_service_missing_document():
    """DocumentQAService raises DocumentContextNotFoundError if document_id not in store."""
    store = DocumentContextStore()
    service = DocumentQAService(
        ai_service=AIService(provider=MockAIProvider()),
        store=store,
    )
    with pytest.raises(DocumentContextNotFoundError) as exc_info:
        await service.answer_question(
            document_id="non_existent_doc", question="What is rent?"
        )
    assert "No document context found" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Relevant Clause Retrieval
# ---------------------------------------------------------------------------

def test_retrieval_relevant_clauses():
    """ClauseRetrievalService returns clauses containing question search terms."""
    service = ClauseRetrievalService()
    clauses = [
        _make_clause(1, "The tenant agrees to pay monthly rent by the 5th."),
        _make_clause(2, "The landlord must maintain structural safety."),
        _make_clause(3, "Subletting the property is strictly prohibited."),
    ]
    # Question terms after stop-word removal: {"monthly", "rent"} (2 terms)
    # Clause 1 contains both → score 2/2 = 1.0 >= 0.6
    results = service.retrieve("What is the monthly rent?", clauses)
    assert len(results) >= 1
    assert results[0].clause_id == "clause_1"


# ---------------------------------------------------------------------------
# 6. Irrelevant Clause Filtering
# ---------------------------------------------------------------------------

def test_retrieval_irrelevant_filtering():
    """ClauseRetrievalService filters out clauses with zero term overlap."""
    service = ClauseRetrievalService()
    clauses = [
        _make_clause(1, "The landlord provides maintenance service."),
        _make_clause(2, "Arbitration shall be held in New Delhi."),
    ]
    # Question terms: rent, payment
    results = service.retrieve("What are the rent payment rules?", clauses)
    assert results == []


# ---------------------------------------------------------------------------
# 7. Ranking of Relevant Clauses
# ---------------------------------------------------------------------------

def test_retrieval_ranking_order():
    """ClauseRetrievalService ranks clauses with higher keyword overlap first."""
    service = ClauseRetrievalService()
    clauses = [
        _make_clause(1, "Rent is payable monthly."),
        _make_clause(2, "Monthly rent payment must be made to the landlord's account."),
        _make_clause(3, "General notice terms."),
    ]
    # Question terms: {"monthly", "rent"} (2 terms after stop-word removal)
    # Clause 1 contains both → score 2/2 = 1.0 >= 0.6
    # Clause 2 contains both → score 2/2 = 1.0 >= 0.6 (tied, ordered by document position)
    results = service.retrieve("monthly rent", clauses)
    assert len(results) == 2
    # Both score 1.0; clause_1 appears first in document (index 0), so clause_1 has
    # tie-breaker -0 > clause_2 tie-breaker -1 → clause_1 ranks first
    assert results[0].clause_id == "clause_1"
    assert results[1].clause_id == "clause_2"


# ---------------------------------------------------------------------------
# 8. No Relevant Clauses Found
# ---------------------------------------------------------------------------

def test_retrieval_no_relevant_clauses():
    """ClauseRetrievalService returns empty list when question has no matching clause terms."""
    service = ClauseRetrievalService()
    clauses = [_make_clause(1, "Confidentiality clause details.")]
    results = service.retrieve("What is the security deposit amount?", clauses)
    assert results == []


# ---------------------------------------------------------------------------
# 9. Successful AI Response Using MockAIProvider
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_qa_service_success_mock():
    """DocumentQAService returns DocumentQAResponse using MockAIProvider."""
    store = DocumentContextStore()
    # Clause contains "rent" → question term "rent" (1/1 = 1.0 >= 0.6)
    doc_id = store.store([_make_clause(1, "Tenant shall pay rent of INR 20,000.")])

    mock_json = json.dumps({
        "answer": "The tenant must pay rent of INR 20,000.",
        "source_clauses": ["clause_1"],
        "confidence": 0.95,
        "cannot_answer": False,
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    service = DocumentQAService(
        ai_service=AIService(provider=mock_provider),
        store=store,
    )

    # Question terms after stop-word removal: {"rent"} (1 term) → score 1/1 = 1.0 >= 0.6
    response = await service.answer_question(
        document_id=doc_id, question="What is the rent?"
    )

    assert isinstance(response, DocumentQAResponse)
    assert response.answer == "The tenant must pay rent of INR 20,000."
    assert response.source_clauses == ["clause_1"]
    assert response.confidence == 0.95
    assert response.cannot_answer is False
    assert mock_provider.call_count == 1


# ---------------------------------------------------------------------------
# 10. Grounded Source Clause IDs
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_qa_grounded_source_clause_ids():
    """Verify that source_clauses in AI response matches the clause_ids sent to it."""
    store = DocumentContextStore()
    # Question term "rent" (1/1 = 1.0 >= 0.6) → clause_1 is retrieved
    doc_id = store.store([
        _make_clause(1, "Monthly rent payment is INR 15,000 due on the 5th."),
    ])

    mock_json = json.dumps({
        "answer": "Rent is INR 15,000 due on the 5th.",
        "source_clauses": ["clause_1"],
        "confidence": 0.9,
        "cannot_answer": False,
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    service = DocumentQAService(
        ai_service=AIService(provider=mock_provider),
        store=store,
    )

    # Question term: {"rent"} → clause_1 has "rent" → 1/1 = 1.0 >= 0.6
    response = await service.answer_question(
        document_id=doc_id, question="What is the rent?"
    )
    assert response.source_clauses == ["clause_1"]


# ---------------------------------------------------------------------------
# 11. Cannot Answer Response (No AI Call Made)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_qa_cannot_answer_no_relevant_clauses():
    """DocumentQAService returns cannot_answer=True without calling Gemini when retrieval yields no matches."""
    store = DocumentContextStore()
    doc_id = store.store([_make_clause(1, "Lease term is 12 months.")])

    mock_provider = MockAIProvider(response_text="{}")
    service = DocumentQAService(
        ai_service=AIService(provider=mock_provider),
        store=store,
    )

    response = await service.answer_question(
        document_id=doc_id, question="What is the pet policy?"
    )

    assert response.cannot_answer is True
    assert response.confidence == 0.0
    assert response.source_clauses == []
    assert "does not contain" in response.answer
    assert mock_provider.call_count == 0  # No AI call made


# ---------------------------------------------------------------------------
# 12. Invalid AI JSON Error Handling
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_qa_invalid_ai_json():
    """DocumentQAService raises QAGenerationError when AI returns malformed JSON."""
    store = DocumentContextStore()
    # Clause contains "rent" → question term "rent" (1/1 = 1.0 >= 0.6)
    doc_id = store.store([_make_clause(1, "Tenant shall pay rent.")])

    mock_provider = MockAIProvider(response_text="invalid-json")
    service = DocumentQAService(
        ai_service=AIService(provider=mock_provider),
        store=store,
    )

    # Question term after stop-word removal: {"rent"} → clause has "rent" → 1/1 = 1.0
    with pytest.raises(QAGenerationError) as exc_info:
        await service.answer_question(
            document_id=doc_id, question="What is the rent?"
        )
    assert "Document Q&A failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 13. AI Provider Failure Handling
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_qa_ai_provider_failure():
    """DocumentQAService raises QAGenerationError when AI provider throws an error."""
    class FailingProvider(BaseAIProvider):
        async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
            raise AIProviderError("Gemini API connection error.")

    store = DocumentContextStore()
    # Clause contains "rent" → question term "rent" (1/1 = 1.0 >= 0.6)
    doc_id = store.store([_make_clause(1, "Tenant shall pay rent.")])

    service = DocumentQAService(
        ai_service=AIService(provider=FailingProvider()),
        store=store,
    )

    # Question term: {"rent"} → clause has "rent" → 1/1 = 1.0 — AI is called, then fails
    with pytest.raises(QAGenerationError) as exc_info:
        await service.answer_question(
            document_id=doc_id, question="What is the rent?"
        )
    assert "Document Q&A failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 14. Confidence Validation Range
# ---------------------------------------------------------------------------

def test_qa_response_confidence_range_validation():
    """DocumentQAResponse validates confidence within [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        DocumentQAResponse(
            answer="Test answer.",
            source_clauses=["clause_1"],
            confidence=1.5,
            cannot_answer=False,
        )

    with pytest.raises(ValidationError):
        DocumentQAResponse(
            answer="Test answer.",
            source_clauses=["clause_1"],
            confidence=-0.1,
            cannot_answer=False,
        )


# ---------------------------------------------------------------------------
# 15. Large Document Retrieval
# ---------------------------------------------------------------------------

def test_retrieval_large_document_top_n():
    """ClauseRetrievalService retrieves top 5 relevant clauses from a 50+ clause document."""
    service = ClauseRetrievalService(top_n=5)
    clauses = [
        _make_clause(i, f"Clause {i} describes general policy number {i}.")
        for i in range(1, 55)
    ]
    # Add target keywords to specific clauses
    clauses[10] = _make_clause(11, "Clause 11 defines notice termination period of 30 days.")
    clauses[20] = _make_clause(21, "Clause 21 specifies termination notice requirements.")

    results = service.retrieve("What is the termination notice period?", clauses)
    assert len(results) <= 5
    assert len(results) >= 2
    clause_ids = [c.clause_id for c in results]
    assert "clause_11" in clause_ids
    assert "clause_21" in clause_ids


# ---------------------------------------------------------------------------
# 16. Endpoint Success (200 OK)
# ---------------------------------------------------------------------------

def test_ask_endpoint_success():
    """POST /documents/ask returns 200 OK with valid response JSON."""
    doc_id = document_context_store.store([
        _make_clause(1, "The rent is INR 25,000 payable by the 5th.")
    ])

    mock_qa_response = DocumentQAResponse(
        answer="The rent is INR 25,000.",
        source_clauses=["clause_1"],
        confidence=0.92,
        cannot_answer=False,
    )

    with patch.object(
        DocumentQAService, "answer_question", new_callable=AsyncMock
    ) as mock_answer:
        mock_answer.return_value = mock_qa_response

        payload = {"document_id": doc_id, "question": "What is the rent?"}
        response = client.post("/documents/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The rent is INR 25,000."
    assert data["source_clauses"] == ["clause_1"]
    assert data["confidence"] == 0.92
    assert data["cannot_answer"] is False


# ---------------------------------------------------------------------------
# 17. Endpoint Validation Errors (422 Unprocessable Entity)
# ---------------------------------------------------------------------------

def test_ask_endpoint_validation_errors():
    """POST /documents/ask returns 422 when request body fails validation."""
    # Empty question
    res1 = client.post("/documents/ask", json={"document_id": "doc1", "question": ""})
    assert res1.status_code == 422

    # Empty document_id
    res2 = client.post("/documents/ask", json={"document_id": "", "question": "What is rent?"})
    assert res2.status_code == 422

    # Missing question
    res3 = client.post("/documents/ask", json={"document_id": "doc1"})
    assert res3.status_code == 422


# ---------------------------------------------------------------------------
# 18. Endpoint Document-Not-Found Error (404 Not Found)
# ---------------------------------------------------------------------------

def test_ask_endpoint_document_not_found():
    """POST /documents/ask returns 404 when document_id is not in store."""
    payload = {"document_id": "unknown_doc_id", "question": "What is the rent?"}
    response = client.post("/documents/ask", json=payload)

    assert response.status_code == 404
    assert "No document context found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 19. Unrelated Question - Capital of France
# ---------------------------------------------------------------------------

def test_retrieval_unrelated_france_capital():
    """ClauseRetrievalService returns [] for 'What is the capital of France?' against legal contract."""
    service = ClauseRetrievalService()
    clauses = [
        _make_clause(1, "Working capital of the company shall be maintained at INR 10,00,000."),
        _make_clause(2, "The tenant shall pay monthly rent on or before the 5th."),
        _make_clause(3, "Notice of termination must be provided 30 days in advance."),
    ]
    results = service.retrieve("What is the capital of France?", clauses)
    assert results == []


# ---------------------------------------------------------------------------
# 20. Unrelated Question - Recipe for Pizza
# ---------------------------------------------------------------------------

def test_retrieval_unrelated_pizza_recipe():
    """ClauseRetrievalService returns [] for 'What is the recipe for making pizza?' against legal contract."""
    service = ClauseRetrievalService()
    clauses = [
        _make_clause(1, "The decision making process involves all directors."),
        _make_clause(2, "Payments are subject to standard deductions."),
    ]
    results = service.retrieve("What is the recipe for making pizza?", clauses)
    assert results == []


# ---------------------------------------------------------------------------
# 21. Unrelated Question - World Cup Winner
# ---------------------------------------------------------------------------

def test_retrieval_unrelated_world_cup():
    """ClauseRetrievalService returns [] for 'Who won the World Cup?' against legal contract."""
    service = ClauseRetrievalService()
    clauses = [
        _make_clause(1, "The party who won the arbitration award shall recover costs."),
        _make_clause(2, "Confidentiality terms apply worldwide."),
    ]
    results = service.retrieve("Who won the World Cup?", clauses)
    assert results == []


# ---------------------------------------------------------------------------
# 22. End-to-End Service Unrelated Question - No AI Call Made
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_qa_service_unrelated_question_no_ai_call():
    """DocumentQAService returns cannot_answer=True with 0 AI calls for unrelated questions."""
    store = DocumentContextStore()
    doc_id = store.store([
        _make_clause(1, "Working capital requirements are specified in Schedule A."),
        _make_clause(2, "Rent is INR 25,000 per month."),
    ])

    mock_provider = MockAIProvider(response_text="{}")
    service = DocumentQAService(
        ai_service=AIService(provider=mock_provider),
        store=store,
    )

    response = await service.answer_question(
        document_id=doc_id, question="What is the capital of France?"
    )

    assert response.cannot_answer is True
    assert response.confidence == 0.0
    assert response.source_clauses == []
    assert "does not contain" in response.answer
    assert mock_provider.call_count == 0  # Crucial: Gemini Provider was NOT invoked


# ---------------------------------------------------------------------------
# 23. Genuine Multi-Term Match (Sufficient Score)
# ---------------------------------------------------------------------------

def test_retrieval_genuine_multi_term_match():
    """ClauseRetrievalService returns relevant clauses when query terms meet relevance threshold."""
    service = ClauseRetrievalService()
    clauses = [
        _make_clause(1, "The security deposit of INR 50,000 shall be refunded upon termination."),
        _make_clause(2, "General notice terms for disputes."),
    ]
    results = service.retrieve("What is the security deposit amount?", clauses)
    assert len(results) == 1
    assert results[0].clause_id == "clause_1"


# ---------------------------------------------------------------------------
# 24. Monthly Rent Due Date Regression Tests
# ---------------------------------------------------------------------------

def test_retrieval_monthly_rent_due_regression():
    """ClauseRetrievalService accurately retrieves rent clauses for 'What is the monthly rent and when is it due?'."""
    service = ClauseRetrievalService()
    
    # Variation 1: Clause with 'monthly rent'
    clauses_1 = [
        _make_clause(1, "The tenant shall pay monthly rent of INR 25,000 by the 5th of each month."),
        _make_clause(2, "The landlord may inspect with 24 hours notice."),
    ]
    results_1 = service.retrieve("What is the monthly rent and when is it due?", clauses_1)
    assert len(results_1) >= 1
    assert results_1[0].clause_id == "clause_1"

    # Variation 2: Clause with 'rent' and 'month' (inflection variation)
    clauses_2 = [
        _make_clause(1, "The tenant shall pay rent of INR 25,000 by the 5th of each month."),
        _make_clause(2, "Notice of 30 days is required for termination."),
    ]
    results_2 = service.retrieve("What is the monthly rent and when is it due?", clauses_2)
    assert len(results_2) >= 1
    assert results_2[0].clause_id == "clause_1"


@pytest.mark.anyio
async def test_qa_service_monthly_rent_due_end_to_end():
    """DocumentQAService returns answer and cannot_answer=False for 'What is the monthly rent and when is it due?'."""
    store = DocumentContextStore()
    doc_id = store.store([
        _make_clause(1, "The tenant shall pay monthly rent of INR 25,000 by the 5th of each month."),
        _make_clause(2, "The premises shall be used for residential purposes only.")
    ])

    mock_json = json.dumps({
        "answer": "The monthly rent is INR 25,000 payable on or before the 5th of each month.",
        "source_clauses": ["clause_1"],
        "confidence": 0.95,
        "cannot_answer": False,
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    service = DocumentQAService(
        ai_service=AIService(provider=mock_provider),
        store=store,
    )

    response = await service.answer_question(
        document_id=doc_id,
        question="What is the monthly rent and when is it due?"
    )

    assert isinstance(response, DocumentQAResponse)
    assert response.cannot_answer is False
    assert "25,000" in response.answer
    assert response.source_clauses == ["clause_1"]
    assert mock_provider.call_count == 1


