from pydantic import BaseModel

class ClauseSegment(BaseModel):
    clause_id: str
    clause_number: str | None
    text: str

class DocumentUploadResponse(BaseModel):
    filename: str
    page_count: int
    character_count: int
    clauses: list[ClauseSegment]
