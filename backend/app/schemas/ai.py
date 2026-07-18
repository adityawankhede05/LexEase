from enum import StrEnum
from pydantic import BaseModel

class AITask(StrEnum):
    SIMPLIFICATION = "simplification"

class ClauseSimplificationResponse(BaseModel):
    simplified_text: str
