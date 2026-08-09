import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.ai.base_provider import BaseAIProvider
from app.ai.exceptions import AIProviderError
from app.ai.prompt_builder import PromptBuilder
from app.schemas.ai import AITask
from app.schemas.clause_analysis import (
    ClauseAnalysisRequest,
    ClauseAnalysisResponse,
    ClauseRiskResult,
    RiskLevel,
)
from app.schemas.document import ClauseSegment
from app.services.ai import AIService
from app.services.clause_analysis import ClauseAnalysisService
from app.services.exceptions import ClauseAnalysisError, EmptyClauseListError
from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

class MockAIProvider(BaseAIProvider):
    """Minimal in-memory provider that returns a preset response string."""

    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.last_prompt: str | None = None

    async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        self.last_prompt = prompt
        return self.response_text


def _make_clause(idx: int = 1) -> ClauseSegment:
    return ClauseSegment(
        clause_id=f"clause_{idx}",
        clause_number=str(idx),
        text=f"Clause {idx} text: The party of the first part agrees to pay.",
    )


def _make_result_dict(
    clause_id: str = "clause_1",
    clause_number: str = "1",
    risk_level: str = "medium",
    explanation: str = "This clause imposes a financial obligation on the signatory.",
    recommendation: str = "Request a payment cap and late-fee waiver clause before signing.",
    confidence: float = 0.85,
) -> dict:
    return {
        "clause_id": clause_id,
        "clause_number": clause_number,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 1. PromptBuilder — builds correct prompt for CLAUSE_ANALYSIS
# ---------------------------------------------------------------------------

def test_prompt_builder_clause_analysis():
    """PromptBuilder builds a clause analysis prompt containing key instructions."""
    clauses_json = json.dumps([{"clause_id": "clause_1", "clause_number": "1", "text": "Sample."}])
    payload = {"clauses_json": clauses_json}
    prompt = PromptBuilder.build_prompt(AITask.CLAUSE_ANALYSIS, payload)

    assert "legal risk analyst" in prompt
    assert "clause_id" in prompt
    assert "risk_level" in prompt
    assert "explanation" in prompt
    assert "recommendation" in prompt
    assert "confidence" in prompt
    assert "clause_1" in prompt


# ---------------------------------------------------------------------------
# 2. PromptBuilder — raises ValueError on missing payload key
# ---------------------------------------------------------------------------

def test_prompt_builder_clause_analysis_missing_payload():
    """PromptBuilder raises ValueError when clauses_json is absent from payload."""
    with pytest.raises(ValueError) as exc_info:
        PromptBuilder.build_prompt(AITask.CLAUSE_ANALYSIS, {})
    assert "Missing required key 'clauses_json'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. Service — success path
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_clause_analysis_service_success():
    """ClauseAnalysisService returns valid ClauseAnalysisResponse on successful AI output."""
    result_dict = _make_result_dict()
    mock_json = json.dumps({"results": [result_dict]})
    mock_provider = MockAIProvider(response_text=mock_json)
    service = ClauseAnalysisService(ai_service=AIService(provider=mock_provider))

    response = await service.analyze_clauses([_make_clause(1)])

    assert isinstance(response, ClauseAnalysisResponse)
    assert response.total_clauses == 1
    assert len(response.results) == 1

    result = response.results[0]
    assert result.clause_id == "clause_1"
    assert result.clause_number == "1"
    assert result.risk_level == RiskLevel.MEDIUM
    assert isinstance(result.explanation, str) and result.explanation
    assert isinstance(result.recommendation, str) and result.recommendation
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# 4. Service — empty clause list raises EmptyClauseListError
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_clause_analysis_service_empty_clauses():
    """ClauseAnalysisService raises EmptyClauseListError when given an empty list."""
    service = ClauseAnalysisService(ai_service=AIService(provider=MockAIProvider()))
    with pytest.raises(EmptyClauseListError) as exc_info:
        await service.analyze_clauses([])
    assert "cannot be empty" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Service — AI provider error surfaces as ClauseAnalysisError
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_clause_analysis_service_ai_provider_error():
    """ClauseAnalysisService wraps AIProviderError into ClauseAnalysisError."""
    class FailingProvider(BaseAIProvider):
        async def generate(self, prompt: str, system_instruction: str | None = None) -> str:
            raise AIProviderError("Gemini API unavailable.")

    service = ClauseAnalysisService(ai_service=AIService(provider=FailingProvider()))
    with pytest.raises(ClauseAnalysisError) as exc_info:
        await service.analyze_clauses([_make_clause(1)])
    assert "Clause risk analysis failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. Service — invalid JSON from AI surfaces as ClauseAnalysisError
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_clause_analysis_service_invalid_json():
    """ClauseAnalysisService raises ClauseAnalysisError when AI returns malformed JSON."""
    mock_provider = MockAIProvider(response_text="not-valid-json")
    service = ClauseAnalysisService(ai_service=AIService(provider=mock_provider))
    with pytest.raises(ClauseAnalysisError) as exc_info:
        await service.analyze_clauses([_make_clause(1)])
    assert "Clause risk analysis failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. Service — schema validation failure (missing field) raises ClauseAnalysisError
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_clause_analysis_service_schema_validation_failure():
    """ClauseAnalysisService raises ClauseAnalysisError when AI omits a required field."""
    # recommendation is intentionally omitted to trigger Pydantic validation error
    bad_result = {
        "clause_id": "clause_1",
        "clause_number": "1",
        "risk_level": "low",
        "explanation": "Looks fine.",
        "confidence": 0.9,
    }
    mock_json = json.dumps({"results": [bad_result]})
    mock_provider = MockAIProvider(response_text=mock_json)
    service = ClauseAnalysisService(ai_service=AIService(provider=mock_provider))
    with pytest.raises(ClauseAnalysisError) as exc_info:
        await service.analyze_clauses([_make_clause(1)])
    assert "Clause risk analysis failed" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 8. Service — large document (50+ clauses) handled in a single request
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_clause_analysis_service_large_document():
    """ClauseAnalysisService handles 54 clauses in one request, returning all results."""
    clauses = [_make_clause(i) for i in range(1, 55)]
    results = [
        _make_result_dict(
            clause_id=f"clause_{i}",
            clause_number=str(i),
            risk_level="low",
            recommendation="This clause is standard; review for completeness.",
            confidence=0.75,
        )
        for i in range(1, 55)
    ]
    mock_json = json.dumps({"results": results})
    mock_provider = MockAIProvider(response_text=mock_json)
    service = ClauseAnalysisService(ai_service=AIService(provider=mock_provider))

    response = await service.analyze_clauses(clauses)

    assert isinstance(response, ClauseAnalysisResponse)
    assert response.total_clauses == 54
    assert len(response.results) == 54
    # Verify all clause_ids are present
    returned_ids = {r.clause_id for r in response.results}
    expected_ids = {f"clause_{i}" for i in range(1, 55)}
    assert returned_ids == expected_ids
    # Verify recommendation present on all results
    for result in response.results:
        assert isinstance(result.recommendation, str) and result.recommendation
    # Confirm single batched request (provider called once)
    assert mock_provider.last_prompt is not None
    assert "clause_1" in mock_provider.last_prompt
    assert "clause_54" in mock_provider.last_prompt


# ---------------------------------------------------------------------------
# 9. Service — confidence values are always within [0.0, 1.0]
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_clause_analysis_service_confidence_range():
    """All returned confidence scores are validated to be within [0.0, 1.0]."""
    results = [
        _make_result_dict(clause_id="clause_1", confidence=0.0),
        _make_result_dict(clause_id="clause_2", confidence=0.5),
        _make_result_dict(clause_id="clause_3", confidence=1.0),
    ]
    clauses = [_make_clause(i) for i in range(1, 4)]
    mock_json = json.dumps({"results": results})
    mock_provider = MockAIProvider(response_text=mock_json)
    service = ClauseAnalysisService(ai_service=AIService(provider=mock_provider))

    response = await service.analyze_clauses(clauses)

    for result in response.results:
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# 10. Endpoint — success (200 OK)
# ---------------------------------------------------------------------------

def test_analyze_endpoint_success():
    """POST /clauses/analyze returns 200 OK with valid ClauseAnalysisResponse JSON."""
    mock_response = ClauseAnalysisResponse(
        total_clauses=2,
        results=[
            ClauseRiskResult(
                clause_id="clause_1",
                clause_number="1",
                risk_level=RiskLevel.HIGH,
                explanation="This clause waives all liability.",
                recommendation="Negotiate or refuse this clause; seek legal advice.",
                confidence=0.92,
            ),
            ClauseRiskResult(
                clause_id="clause_2",
                clause_number="2",
                risk_level=RiskLevel.LOW,
                explanation="Standard notice provision.",
                recommendation="Review for completeness; generally acceptable.",
                confidence=0.78,
            ),
        ],
    )

    with patch.object(
        ClauseAnalysisService, "analyze_clauses", new_callable=AsyncMock
    ) as mock_analyze:
        mock_analyze.return_value = mock_response

        payload = {
            "clauses": [
                {"clause_id": "clause_1", "clause_number": "1", "text": "Waives all liability."},
                {"clause_id": "clause_2", "clause_number": "2", "text": "30 days notice required."},
            ]
        }
        response = client.post("/clauses/analyze", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total_clauses"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["risk_level"] == "high"
    assert data["results"][0]["recommendation"] == "Negotiate or refuse this clause; seek legal advice."
    assert data["results"][1]["risk_level"] == "low"
    assert data["results"][1]["recommendation"] == "Review for completeness; generally acceptable."


# ---------------------------------------------------------------------------
# 11. Endpoint — empty clauses (400 Bad Request)
# ---------------------------------------------------------------------------

def test_analyze_endpoint_empty_clauses():
    """POST /clauses/analyze returns 400 when the clause list is empty."""
    with patch.object(
        ClauseAnalysisService,
        "analyze_clauses",
        side_effect=EmptyClauseListError("Clause list cannot be empty for clause risk analysis."),
    ):
        response = client.post("/clauses/analyze", json={"clauses": []})

    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 12. Endpoint — AI generation failure (500 Internal Server Error)
# ---------------------------------------------------------------------------

def test_analyze_endpoint_ai_error():
    """POST /clauses/analyze returns 500 when ClauseAnalysisService raises ClauseAnalysisError."""
    with patch.object(
        ClauseAnalysisService,
        "analyze_clauses",
        side_effect=ClauseAnalysisError("Clause risk analysis failed: Gemini API unavailable."),
    ):
        payload = {
            "clauses": [{"clause_id": "clause_1", "clause_number": "1", "text": "Sample clause."}]
        }
        response = client.post("/clauses/analyze", json=payload)

    assert response.status_code == 500
    assert "Clause risk analysis failed" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 13. Schema — Pydantic rejects invalid risk_level string
# ---------------------------------------------------------------------------

def test_clause_risk_result_schema_validation():
    """ClauseRiskResult raises ValidationError when risk_level is not a valid RiskLevel."""
    with pytest.raises(ValidationError):
        ClauseRiskResult(
            clause_id="clause_1",
            clause_number="1",
            risk_level="critical",          # invalid value
            explanation="Something bad.",
            recommendation="Do not sign.",
            confidence=0.9,
        )


# ---------------------------------------------------------------------------
# 14. Schema — Pydantic rejects confidence values outside [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_clause_risk_result_confidence_out_of_range():
    """ClauseRiskResult raises ValidationError when confidence is outside [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        ClauseRiskResult(
            clause_id="clause_1",
            clause_number="1",
            risk_level=RiskLevel.LOW,
            explanation="Fine.",
            recommendation="Looks good.",
            confidence=1.5,                 # invalid: > 1.0
        )

    with pytest.raises(ValidationError):
        ClauseRiskResult(
            clause_id="clause_1",
            clause_number="1",
            risk_level=RiskLevel.LOW,
            explanation="Fine.",
            recommendation="Looks good.",
            confidence=-0.1,                # invalid: < 0.0
        )


# ---------------------------------------------------------------------------
# 15. Schema — Pydantic rejects missing recommendation field
# ---------------------------------------------------------------------------

def test_clause_risk_result_missing_recommendation():
    """ClauseRiskResult raises ValidationError when recommendation is absent."""
    with pytest.raises(ValidationError) as exc_info:
        ClauseRiskResult(
            clause_id="clause_1",
            clause_number="1",
            risk_level=RiskLevel.HIGH,
            explanation="This clause is very risky.",
            # recommendation intentionally omitted
            confidence=0.88,
        )
    assert "recommendation" in str(exc_info.value)
