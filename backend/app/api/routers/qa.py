from fastapi import APIRouter, HTTPException, status
from app.schemas.qa import DocumentQARequest, DocumentQAResponse
from app.services.exceptions import (
    DocumentContextNotFoundError,
    QAGenerationError,
)
from app.services.qa import DocumentQAService

router = APIRouter(prefix="/documents", tags=["Document Q&A"])


@router.post("/ask", response_model=DocumentQAResponse)
async def ask_document_question(request: DocumentQARequest):
    """
    Answers a question about an uploaded legal document using only its preprocessed clauses.

    Accepts document_id and question in the request body.
    Returns answer, source_clauses, confidence, and cannot_answer flag.
    """
    qa_service = DocumentQAService()
    try:
        return await qa_service.answer_question(
            document_id=request.document_id,
            question=request.question,
        )
    except DocumentContextNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except QAGenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
