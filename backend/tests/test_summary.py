import json
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient

from app.ai.base_provider import BaseAIProvider
from app.ai.prompt_builder import PromptBuilder
from app.schemas.ai import AITask
from app.schemas.document import ClauseSegment
from app.schemas.summary import DocumentSummaryRequest, DocumentSummaryResponse
from app.services.ai import AIService
from app.services.exceptions import EmptyClauseListError, SummaryGenerationError
from app.services.summary import DocumentSummaryService
from app.main import app

client = TestClient(app)

class MockAIProvider(BaseAIProvider):
    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.last_prompt: str | None = None

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.last_prompt = prompt
        return self.response_text


def test_prompt_builder_document_summary():
    """Verify PromptBuilder constructs document summary prompt using template."""
    payload = {"document_text": "Clause 1: Tenant shall pay rent."}
    prompt = PromptBuilder.build_prompt(AITask.DOCUMENT_SUMMARY, payload)
    assert "Summarize the following legal document into clear, accessible plain English" in prompt
    assert "Clause 1: Tenant shall pay rent." in prompt
    assert "Return ONLY raw valid JSON" in prompt


def test_prompt_builder_document_summary_missing_payload():
    """Verify PromptBuilder raises ValueError if document_text is missing."""
    with pytest.raises(ValueError) as exc_info:
        PromptBuilder.build_prompt(AITask.DOCUMENT_SUMMARY, {})
    assert "Missing required key 'document_text'" in str(exc_info.value)


@pytest.mark.anyio
async def test_summary_service_success():
    """Verify DocumentSummaryService concatenates clauses and returns DocumentSummaryResponse."""
    mock_json = json.dumps({
        "summary": "The agreement specifies rent payment terms.",
        "key_points": ["Rent is due monthly", "Security deposit is required"],
        "document_type": "Rental Agreement"
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    ai_service = AIService(provider=mock_provider)
    summary_service = DocumentSummaryService(ai_service=ai_service)

    clauses = [
        ClauseSegment(clause_id="clause_1", clause_number="1.1", text="Tenant agrees to pay monthly rent."),
        ClauseSegment(clause_id="clause_2", clause_number="2.1", text="Deposit is equal to 2 months rent.")
    ]

    result = await summary_service.summarize_document(clauses)

    assert isinstance(result, DocumentSummaryResponse)
    assert result.summary == "The agreement specifies rent payment terms."
    assert len(result.key_points) == 2
    assert result.document_type == "Rental Agreement"
    assert mock_provider.last_prompt is not None
    assert "Clause 1.1: Tenant agrees to pay monthly rent." in mock_provider.last_prompt
    assert "Clause 2.1: Deposit is equal to 2 months rent." in mock_provider.last_prompt


@pytest.mark.anyio
async def test_summary_service_empty_clauses():
    """Verify DocumentSummaryService raises EmptyClauseListError when clause list is empty."""
    summary_service = DocumentSummaryService(ai_service=AIService(provider=MockAIProvider()))
    with pytest.raises(EmptyClauseListError) as exc_info:
        await summary_service.summarize_document([])
    assert "Clause list cannot be empty" in str(exc_info.value)


@pytest.mark.anyio
async def test_summary_service_ai_failure():
    """Verify DocumentSummaryService raises SummaryGenerationError on AI/validation failure."""
    mock_provider = MockAIProvider(response_text="invalid-json-response")
    ai_service = AIService(provider=mock_provider)
    summary_service = DocumentSummaryService(ai_service=ai_service)

    clauses = [ClauseSegment(clause_id="c1", clause_number="1", text="Sample legal text.")]

    with pytest.raises(SummaryGenerationError) as exc_info:
        await summary_service.summarize_document(clauses)
    assert "Document summarization failed" in str(exc_info.value)


@pytest.mark.anyio
async def test_summary_service_large_document():
    """Verify DocumentSummaryService handles large documents with 50+ clause segments cleanly."""
    mock_json = json.dumps({
        "summary": "Large master agreement covering operational terms.",
        "key_points": ["50 operational clauses covered", "High compliance level"],
        "document_type": "Master Service Agreement"
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    ai_service = AIService(provider=mock_provider)
    summary_service = DocumentSummaryService(ai_service=ai_service)

    clauses = [
        ClauseSegment(
            clause_id=f"clause_{i}",
            clause_number=f"{i}.0",
            text=f"This is section {i} specifying operational standard operating procedure requirement number {i}."
        )
        for i in range(1, 55)
    ]

    result = await summary_service.summarize_document(clauses)

    assert isinstance(result, DocumentSummaryResponse)
    assert result.summary == "Large master agreement covering operational terms."
    assert len(clauses) == 54
    assert mock_provider.last_prompt is not None
    assert "Clause 1.0:" in mock_provider.last_prompt
    assert "Clause 54.0:" in mock_provider.last_prompt


def test_summarize_endpoint_success():
    """Verify POST /documents/summarize returns 200 OK and valid JSON summary response."""
    mock_response = DocumentSummaryResponse(
        summary="This document outlines lease obligations.",
        key_points=["Pay rent on time", "No sub-letting"],
        document_type="Lease Agreement"
    )

    with patch.object(DocumentSummaryService, "summarize_document", new_callable=AsyncMock) as mock_summarize:
        mock_summarize.return_value = mock_response

        payload = {
            "clauses": [
                {"clause_id": "c1", "clause_number": "1", "text": "Tenant shall pay rent."},
                {"clause_id": "c2", "clause_number": "2", "text": "No sub-letting allowed."}
            ]
        }

        response = client.post("/documents/summarize", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "This document outlines lease obligations."
        assert len(data["key_points"]) == 2
        assert data["document_type"] == "Lease Agreement"


def test_summarize_endpoint_empty_clauses():
    """Verify POST /documents/summarize returns 400 Bad Request when clause list is empty."""
    with patch.object(DocumentSummaryService, "summarize_document", side_effect=EmptyClauseListError("Clause list cannot be empty.")):
        payload = {"clauses": []}
        response = client.post("/documents/summarize", json=payload)
        assert response.status_code == 400
        assert "Clause list cannot be empty" in response.json()["detail"]


def test_summarize_endpoint_ai_error():
    """Verify POST /documents/summarize returns 500 Internal Server Error when summarization fails."""
    with patch.object(DocumentSummaryService, "summarize_document", side_effect=SummaryGenerationError("AI Service unavailable")):
        payload = {
            "clauses": [{"clause_id": "c1", "clause_number": "1", "text": "Sample clause"}]
        }
        response = client.post("/documents/summarize", json=payload)
        assert response.status_code == 500
        assert "AI Service unavailable" in response.json()["detail"]
