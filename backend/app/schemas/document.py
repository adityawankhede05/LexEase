from pydantic import BaseModel

class DocumentUploadResponse(BaseModel):
    filename: str
    page_count: int
    character_count: int
    extracted_text: str
