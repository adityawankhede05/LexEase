from fastapi import APIRouter, HTTPException, status
from app.schemas.summary import DocumentSummaryRequest, DocumentSummaryResponse
from app.services.summary import DocumentSummaryService
from app.services.exceptions import EmptyClauseListError, SummaryGenerationError

router = APIRouter(prefix="/documents", tags=["Summarization"])

@router.post("/summarize", response_model=DocumentSummaryResponse)
async def summarize_document(request: DocumentSummaryRequest):
    """
    Summarizes a legal document represented as a list of preprocessed clause segments.
    """
    summary_service = DocumentSummaryService()
    try:
        return await summary_service.summarize_document(
            request.clauses, document_id=request.document_id
        )
    except EmptyClauseListError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except SummaryGenerationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
