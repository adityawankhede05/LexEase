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
from app.services.summary import DocumentSummaryService, estimate_tokens
from app.main import app

client = TestClient(app)


class MockAIProvider(BaseAIProvider):
    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.last_prompt: str | None = None
        self.prompts: list[str] = []
        self.call_count: int = 0

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.last_prompt = prompt
        self.prompts.append(prompt)
        self.call_count += 1
        return self.response_text


class SequentialMockAIProvider(BaseAIProvider):
    """Returns responses from a list sequentially on each call."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.call_count: int = 0

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.prompts.append(prompt)
        self.call_count += 1
        if self.responses:
            return self.responses.pop(0)
        return json.dumps({
            "summary": "Default summary",
            "key_points": ["Default key point"],
            "document_type": "Legal Document"
        })


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
    assert mock_provider.call_count == 1


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
async def test_summary_service_small_document_single_call():
    """Verify small document fitting into one chunk results in exactly 1 AI call."""
    mock_json = json.dumps({
        "summary": "Short agreement summary.",
        "key_points": ["Point 1", "Point 2"],
        "document_type": "NDA"
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    ai_service = AIService(provider=mock_provider)
    summary_service = DocumentSummaryService(ai_service=ai_service, max_chunk_tokens=3500)

    clauses = [
        ClauseSegment(clause_id=f"c_{i}", clause_number=f"{i}", text=f"Short clause text {i}.")
        for i in range(1, 6)
    ]

    result = await summary_service.summarize_document(clauses)

    assert isinstance(result, DocumentSummaryResponse)
    assert mock_provider.call_count == 1
    assert result.summary == "Short agreement summary."


@pytest.mark.anyio
async def test_summary_service_large_document_multiple_chunks():
    """Verify large document exceeding chunk token budget creates multiple chunk calls + 1 reduce call."""
    mock_json = json.dumps({
        "summary": "Chunk summary.",
        "key_points": ["Chunk key point"],
        "document_type": "Service Agreement"
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    ai_service = AIService(provider=mock_provider)
    # Set a small token budget of 50 tokens per chunk to force multi-chunk map-reduce
    summary_service = DocumentSummaryService(ai_service=ai_service, max_chunk_tokens=50, chunk_delay=0.0)

    clauses = [
        ClauseSegment(
            clause_id=f"clause_{i}",
            clause_number=f"{i}.0",
            text=f"This is section {i} specifying operational compliance obligation requirement {i}."
        )
        for i in range(1, 10)
    ]

    result = await summary_service.summarize_document(clauses)

    assert isinstance(result, DocumentSummaryResponse)
    # With 9 clauses and max_chunk_tokens=50, there should be multiple chunk calls + 1 reduce call
    assert mock_provider.call_count > 1
    # Check that the final reduce prompt contained the Section summary headers
    final_prompt = mock_provider.prompts[-1]
    assert "Summary of document sections:" in final_prompt
    assert "Section 1 Summary" in final_prompt


@pytest.mark.anyio
async def test_summary_service_chunked_summaries_combined_correctly():
    """Verify that intermediate summaries from each chunk are combined properly during reduction."""
    chunk1_response = json.dumps({
        "summary": "Section 1 covers confidentiality obligations.",
        "key_points": ["Keep secrets for 5 years", "No disclosure to competitors"],
        "document_type": "Non-Disclosure Agreement"
    })
    chunk2_response = json.dumps({
        "summary": "Section 2 covers dispute resolution and governing law.",
        "key_points": ["Delaware jurisdiction applies", "Binding arbitration required"],
        "document_type": "Non-Disclosure Agreement"
    })
    final_response = json.dumps({
        "summary": "The document is a Non-Disclosure Agreement covering 5-year confidentiality and Delaware arbitration.",
        "key_points": [
            "Keep secrets for 5 years",
            "No disclosure to competitors",
            "Delaware jurisdiction and arbitration"
        ],
        "document_type": "Non-Disclosure Agreement"
    })

    mock_provider = SequentialMockAIProvider(responses=[chunk1_response, chunk2_response, final_response])
    ai_service = AIService(provider=mock_provider)
    summary_service = DocumentSummaryService(ai_service=ai_service, max_chunk_tokens=40, chunk_delay=0.0)

    clauses = [
        ClauseSegment(clause_id="c1", clause_number="1", text="Receiving party shall maintain confidentiality for 5 years."),
        ClauseSegment(clause_id="c2", clause_number="2", text="Disputes shall be settled exclusively via Delaware arbitration.")
    ]

    result = await summary_service.summarize_document(clauses)

    assert isinstance(result, DocumentSummaryResponse)
    assert result.summary == "The document is a Non-Disclosure Agreement covering 5-year confidentiality and Delaware arbitration."
    assert len(result.key_points) == 3
    assert result.document_type == "Non-Disclosure Agreement"
    assert mock_provider.call_count == 3  # 2 map chunks + 1 reduce synthesis

    # Verify the reduce prompt had both chunk summaries and key points
    reduce_prompt = mock_provider.prompts[2]
    assert "Section 1 covers confidentiality obligations." in reduce_prompt
    assert "Keep secrets for 5 years" in reduce_prompt
    assert "Section 2 covers dispute resolution" in reduce_prompt
    assert "Delaware jurisdiction applies" in reduce_prompt


@pytest.mark.anyio
async def test_summary_service_no_request_exceeds_token_budget():
    """Verify that even for a massive document (e.g. 100+ clauses, 20k+ chars), no single request exceeds the target token budget."""
    mock_json = json.dumps({
        "summary": "Standard clause section summary.",
        "key_points": ["Operational compliance requirement"],
        "document_type": "Master Services Agreement"
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    ai_service = AIService(provider=mock_provider)
    # Default chunk token budget is 3500 tokens (well below 5000 max target)
    summary_service = DocumentSummaryService(ai_service=ai_service, max_chunk_tokens=3500, chunk_delay=0.0)

    # 100 clauses averaging ~250 characters each = ~25,000 characters (~8,000+ tokens)
    clauses = [
        ClauseSegment(
            clause_id=f"clause_{i}",
            clause_number=f"SEC-{i}",
            text=f"The Contractor agrees to provide comprehensive support services pursuant to Schedule {i}. "
                 f"All service level agreements regarding uptime, bug remediation, and quality standards "
                 f"specified in Exhibit {i} shall strictly apply with penalties for non-performance."
        )
        for i in range(1, 101)
    ]

    result = await summary_service.summarize_document(clauses)

    assert isinstance(result, DocumentSummaryResponse)
    assert mock_provider.call_count > 1

    # Verify EVERY prompt sent to the AI provider is within safe token budget (<= 5000 tokens)
    for idx, prompt in enumerate(mock_provider.prompts, 1):
        estimated = estimate_tokens(prompt)
        assert estimated <= 5000, f"Prompt {idx} exceeded token limit with {estimated} estimated tokens"


@pytest.mark.anyio
async def test_summary_service_chunk_pacing():
    """Verify DocumentSummaryService pauses with chunk_delay between chunks and before reduce phase."""
    mock_json = json.dumps({
        "summary": "Chunk summary.",
        "key_points": ["Key point"],
        "document_type": "Agreement"
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    ai_service = AIService(provider=mock_provider)
    summary_service = DocumentSummaryService(
        ai_service=ai_service,
        max_chunk_tokens=50,
        chunk_delay=2.5,
    )

    clauses = [
        ClauseSegment(
            clause_id=f"c_{i}",
            clause_number=f"{i}",
            text=f"Clause text {i} with sufficient words to exceed token chunk boundary."
        )
        for i in range(1, 4)
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await summary_service.summarize_document(clauses)

        assert isinstance(result, DocumentSummaryResponse)
        # Should sleep between map chunks and before the reduce step
        assert mock_sleep.call_count >= 2
        for call in mock_sleep.call_args_list:
            assert call.args[0] == 2.5


@pytest.mark.anyio
async def test_summary_service_single_chunk_no_pacing_sleep():
    """Verify single-chunk documents do not call asyncio.sleep for pacing."""
    mock_json = json.dumps({
        "summary": "Small document summary.",
        "key_points": ["Small point"],
        "document_type": "NDA"
    })
    mock_provider = MockAIProvider(response_text=mock_json)
    ai_service = AIService(provider=mock_provider)
    summary_service = DocumentSummaryService(
        ai_service=ai_service,
        max_chunk_tokens=3500,
        chunk_delay=5.0,
    )

    clauses = [
        ClauseSegment(clause_id="c1", clause_number="1", text="Short clause text.")
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await summary_service.summarize_document(clauses)

        assert isinstance(result, DocumentSummaryResponse)
        assert mock_provider.call_count == 1
        # No pacing sleep should occur for single-chunk document
        mock_sleep.assert_not_called()


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
