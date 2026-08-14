from pydantic import BaseModel, Field


class DocumentQARequest(BaseModel):
    """
    HTTP request body for POST /documents/ask.
    """
    document_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=2000)


class DocumentQAResponse(BaseModel):
    """
    HTTP response body for POST /documents/ask.
    Also used by ResponseParser to validate raw JSON returned by AI.
    """
    answer: str
    source_clauses: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    cannot_answer: bool
