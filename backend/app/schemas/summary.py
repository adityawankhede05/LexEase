from pydantic import BaseModel
from app.schemas.document import DocumentContext

class DocumentSummaryRequest(DocumentContext):
    """
    Request model for document summarization, inheriting clauses from DocumentContext.
    """
    document_id: str | None = None

class DocumentSummaryResponse(BaseModel):
    """
    Response model for whole document summarization.
    """
    summary: str
    key_points: list[str]
    document_type: str | None = None
