from enum import StrEnum
from pydantic import BaseModel

class AITask(StrEnum):
    SIMPLIFICATION = "simplification"
    DOCUMENT_SUMMARY = "document_summary"
    CLAUSE_ANALYSIS = "clause_analysis"

class ClauseSimplificationResponse(BaseModel):
    simplified_text: str
