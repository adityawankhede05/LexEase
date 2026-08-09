from enum import StrEnum
from pydantic import BaseModel, Field
from app.schemas.document import DocumentContext


class RiskLevel(StrEnum):
    """Enumeration of possible risk levels for a legal clause."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClauseRiskResult(BaseModel):
    """
    Risk analysis result for a single clause.

    All six fields are mandatory. Omitting any field will cause a
    validation failure in ResponseParser and surface as a ClauseAnalysisError.
    """
    clause_id: str
    clause_number: str | None
    risk_level: RiskLevel
    explanation: str
    recommendation: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ClauseAnalysisAIResponse(BaseModel):
    """
    Internal wrapper for the raw JSON array returned by the AI model.
    Used exclusively by ResponseParser; never returned to the HTTP client.
    """
    results: list[ClauseRiskResult]


class ClauseAnalysisRequest(DocumentContext):
    """
    HTTP request body for POST /clauses/analyze.
    Inherits `clauses: list[ClauseSegment]` from DocumentContext,
    following the same pattern as DocumentSummaryRequest.
    """
    pass


class ClauseAnalysisResponse(BaseModel):
    """
    HTTP response body for POST /clauses/analyze.
    """
    total_clauses: int
    results: list[ClauseRiskResult]
